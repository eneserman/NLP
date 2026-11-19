"""Train one neural course baseline from scratch (configs/train_scratch.yaml), choose on dev, decode test once.

Usage:
  python scripts/train_scratch.py --system ctc_scratch|aed_hybrid_scratch|rnnt_scratch

Steps:
  1. Character vocabulary from the normalized train transcripts (hrasr/vocab.py).
  2. Log-mel features of train, dev, test and the no-speech clips, cached in work/feats/.
  3. Training on train only. Every `eval_every` epochs: greedy decoding of dev, corpus WER; the
     checkpoint with the lowest dev WER is kept (work/models/<system>/best.pt).
  4. Decoding settings in `dev_search` (e.g. beam size, CTC weight) are chosen on dev with the best
     checkpoint.
  5. With the chosen checkpoint and settings: dev, test and no-speech predictions, written once.

Output:
  results/predictions/<system>.{dev,test,silence}.jsonl and <system>.meta.json
  results/train_logs/<system>.json      loss and dev WER per evaluation, the choices made on dev
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import CLEAN, ROOT, WORK  # noqa: E402
from hrasr.features import spec_augment  # noqa: E402
from hrasr.manifest import read_jsonl, write_jsonl  # noqa: E402
from hrasr.metrics import per_clip_counts  # noqa: E402
from hrasr.scratch_data import FeatureStore, collate, frame_batches  # noqa: E402
from hrasr.scratch_models import AEDModel, build  # noqa: E402
from hrasr.textnorm import normalize_for_scoring  # noqa: E402
from hrasr.vocab import CharVocab  # noqa: E402

SPLITS = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl", "silence": "silence_probe.jsonl"}


def load_cfg(path: Path, system: str) -> tuple[dict, dict]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if system not in cfg["models"]:
        raise SystemExit(f"{system} is not in {path}")
    m = copy.deepcopy(cfg["models"][system])
    train = {**cfg["train"], **m.get("train", {})}
    return cfg, {**m, "train": train}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    c = per_clip_counts(refs, hyps)
    return float(c["w_err"].sum() / c["w_n"].sum())


def lr_at(step: int, peak: float, warmup: int) -> float:
    s = step + 1
    return peak * min(s / warmup, (warmup / s) ** 0.5)


def decode_store(model, store: FeatureStore, vocab: CharVocab, device: str, max_frames: int, opts: dict,
                 nbest_only: bool = False):
    """-> ({id: text}, compute seconds); with nbest_only (AED): ({id: n-best}, seconds)."""
    model.eval()
    out, seconds = {}, 0.0
    lengths = [store.frames(k) for k in range(len(store))]
    for idx in frame_batches(lengths, max_frames, 64, None):
        x, xl = collate(store, idx)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            if nbest_only:
                res = model.nbest(x.to(device), xl.to(device), beam=opts.get("beam", 1))
            else:
                res = model.decode(x.to(device), xl.to(device), **opts)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        for k, r in zip(idx, res):
            out[store.ids[k]] = r if nbest_only else vocab.decode(r)
    return out, seconds


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "numpy"):
        out[pkg] = metadata.version(pkg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_scratch.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--epochs", type=int, default=None, help="override, for a short trial run")
    ap.add_argument("--limit-train", type=int, default=None, help="first N train clips, for a short trial run")
    ap.add_argument("--limit-eval", type=int, default=None, help="first N clips of dev/test/silence, for a trial run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prepare-only", action="store_true",
                    help="only build the feature cache (run once before training several systems in parallel)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    trial = bool(args.epochs or args.limit_train or args.limit_eval)
    if trial and out_dir.resolve() == (ROOT / "results").resolve():
        raise SystemExit("trial options need --out-dir outside results/")

    cfg, m = load_cfg(Path(args.config), args.system)
    tc = m["train"]
    seed_all(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = {s: sorted(read_jsonl(CLEAN / f), key=lambda r: r["id"]) for s, f in SPLITS.items()}
    if args.limit_train:
        rows["train"] = rows["train"][: args.limit_train]
    if args.limit_eval:
        for s in ("dev", "test", "silence"):
            rows[s] = rows[s][: args.limit_eval]
    vocab = CharVocab.from_texts(normalize_for_scoring(r["text"]) for r in rows["train"])
    targets = [vocab.encode(normalize_for_scoring(r["text"])) for r in rows["train"]]
    tag = f"{cfg['features']}" + (f"_trial{args.limit_train}_{args.limit_eval}" if trial else "")
    stores = {s: FeatureStore.cached(WORK / "feats" / tag / s, rows[s], WORK, cfg["features"], args.workers)
              for s in SPLITS}
    if args.prepare_only:
        print(f"features ready: {tag}", flush=True)
        return
    refs_dev = [r["text"] for r in rows["dev"]]

    model = build(m["kind"], len(vocab), m).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["peak_lr"], betas=(0.9, 0.98), weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(cfg["seed"])
    train_len = [stores["train"].frames(k) for k in range(len(stores["train"]))]
    epochs = args.epochs or tc["epochs"]
    ckpt_dir = WORK / "models" / (args.system + ("_trial" if trial else ""))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = {"system": args.system, "params": n_params, "vocab": vocab.to_dict(), "evals": []}
    best_wer, step, t_start = float("inf"), 0, time.time()
    print(f"{args.system}: {n_params / 1e6:.1f} M parameters, vocabulary {len(vocab)}, device {device}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for idx in frame_batches(train_len, tc["max_frames"], tc["max_items"], rng):
            x, xl, y, yl = collate(stores["train"], idx, targets)
            x, xl, y, yl = x.to(device), xl.to(device), y.to(device), yl.to(device)
            if tc["spec_augment"]:
                x = spec_augment(x, xl)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tc["peak_lr"], tc["warmup_steps"])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model.loss(x, xl, y, yl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            step += 1
        hours = (time.time() - t_start) / 3600
        last = epoch == epochs or hours >= tc["max_hours"]
        entry = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)) if losses else None,
                 "hours": round(hours, 3)}
        if epoch % tc["eval_every"] == 0 or last:
            hyps, _ = decode_store(model, stores["dev"], vocab, device, tc["max_frames"], {"beam": 1})
            wer = corpus_wer(refs_dev, [hyps[r["id"]] for r in rows["dev"]])
            entry["dev_wer_greedy"] = wer
            if wer < best_wer:
                best_wer = wer
                torch.save({"model": model.state_dict(), "epoch": epoch, "vocab": vocab.to_dict()}, ckpt_dir / "best.pt")
        log["evals"].append(entry)
        print(json.dumps(entry), flush=True)
        if last:
            break

    best = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    log["chosen_epoch"] = best["epoch"]
    log["chosen_epoch_dev_wer_greedy"] = best_wer

    # decoding settings chosen on dev
    grid = m.get("dev_search") or {}
    keys = sorted(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))] or [{}]
    search = []
    dev_nbest: dict[int, tuple[dict, float]] = {}   # beam -> (n-best per clip, compute seconds), reused below
    if isinstance(model, AEDModel):
        beams = sorted({c.get("beam", 1) for c in combos})
        for beam in beams:
            nb, sec = decode_store(model, stores["dev"], vocab, device, tc["max_frames"], {"beam": beam}, nbest_only=True)
            dev_nbest[beam] = (nb, sec)
            for c in (c for c in combos if c.get("beam", 1) == beam):
                hyps = [vocab.decode(AEDModel.choose(nb[r["id"]], c.get("ctc_lambda", 0.0))) for r in rows["dev"]]
                search.append({**c, "dev_wer": corpus_wer(refs_dev, hyps)})
    elif combos != [{}]:
        for c in combos:
            hyps, _ = decode_store(model, stores["dev"], vocab, device, tc["max_frames"], c)
            search.append({**c, "dev_wer": corpus_wer(refs_dev, [hyps[r["id"]] for r in rows["dev"]])})
    chosen = min(search, key=lambda s: s["dev_wer"]) if search else {}
    opts = {k: v for k, v in chosen.items() if k != "dev_wer"}
    if isinstance(model, AEDModel):
        opts.setdefault("beam", 1)
    log["dev_search"] = search
    log["chosen_decoding"] = opts

    # final predictions with everything frozen: dev, test (once), no-speech clips
    timing = {}
    pred_dir = out_dir / "predictions"
    for split in ("dev", "test", "silence"):
        if split == "dev" and isinstance(model, AEDModel) and opts["beam"] in dev_nbest:
            # the beam search of the dev search above is deterministic: reuse it instead of decoding dev again
            nb, seconds = dev_nbest[opts["beam"]]
            hyps = {i: vocab.decode(AEDModel.choose(nb[i], opts.get("ctc_lambda", 0.0))) for i in nb}
        else:
            hyps, seconds = decode_store(model, stores[split], vocab, device, tc["max_frames"], opts)
        write_jsonl(pred_dir / f"{args.system}.{split}.jsonl", [{"id": r["id"], "hyp": hyps[r["id"]]} for r in rows[split]])
        timing[split] = {"clips": len(rows[split]), "audio_seconds": round(sum(r["audio_length"] for r in rows[split]), 2),
                         "compute_seconds": round(seconds, 2)}
    enc = m["encoder"]
    setup = (f"Conformer {enc['n_layers']}x{enc['d_model']} ({n_params / 1e6:.1f} M parameters), characters, "
             f"trained from scratch on our train split, epoch {best['epoch']} chosen on dev")
    meta = {
        "kind": "baseline", "trained_by_us": True, "system": args.system, "family": m["family"], "setup": setup,
        "checkpoint": None, "revision": None, "language": "Croatian only (trained on our data)",
        "model_type": m["kind"], "encoder": enc, "decoder": m.get("decoder"), "transducer": m.get("transducer"),
        "decoding": opts, "features": cfg["features"], "train_settings": tc, "epochs_run": log["evals"][-1]["epoch"],
        "chosen_epoch": best["epoch"], "params": n_params,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "precision": "fp16 mixed (torch.autocast)" if device == "cuda" else "fp32",
        "batch_size": f"up to {tc['max_frames']} padded frames",
        "audio_seconds": timing["test"]["audio_seconds"], "compute_seconds": timing["test"]["compute_seconds"],
        "timing_split": "test", "timing_note": "model forward and decoding on precomputed log-mel features",
        "timing": timing, "seed": cfg["seed"], "versions": versions(), "trial": trial,
    }
    (pred_dir / f"{args.system}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "train_logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "train_logs" / f"{args.system}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"chosen_epoch": best["epoch"], "chosen_decoding": opts, "timing": timing}), flush=True)


if __name__ == "__main__":
    main()
