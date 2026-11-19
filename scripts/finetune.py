"""Our contribution: fine-tune a pretrained model on the train split (configs/finetune.yaml).

Usage:
  python scripts/finetune.py --system whisper_small_ft|xlsr_300m_ft

Steps:
  1. The base checkpoint at the pinned revision; train/dev/test/no-speech audio decoded once (ffmpeg).
  2. Training on train only. Every `eval_every_steps` steps and after the last step: greedy decoding of
     dev, corpus WER; the best weights are kept (work/models/<system>/best/).
  3. Decoding settings in `dev_search` are chosen on dev with the best weights.
  4. With everything frozen: dev, test (once) and no-speech predictions.

Output:
  results/predictions/<system>.{dev,test,silence}.jsonl and <system>.meta.json (kind "ours")
  results/train_logs/<system>.json     loss and dev WER per evaluation, the choices made on dev
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import platform
import random
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import CLEAN, ROOT, WORK  # noqa: E402
from hrasr.finetune import (WaveStore, build_xlsr_processor, corpus_wer, linear_schedule, whisper_labels,  # noqa: E402
                            xlsr_labels, xlsr_vocab)
from hrasr.inference import ctc_decode_valid  # noqa: E402
from hrasr.manifest import read_jsonl, write_jsonl  # noqa: E402
from hrasr.scratch_data import frame_batches  # noqa: E402

SPLITS = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl", "silence": "silence_probe.jsonl"}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "numpy", "huggingface_hub"):
        out[pkg] = metadata.version(pkg)
    return out


class Whisper:
    def __init__(self, local: str, m: dict, device: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(local)
        self.model = WhisperForConditionalGeneration.from_pretrained(local).to(device)
        self.model.generation_config.forced_decoder_ids = None
        self.m, self.device = m, device

    def batches(self, store: WaveStore, rng):
        idx = list(range(len(store)))
        rng.shuffle(idx)
        bs = self.m["train"]["batch_size"]
        return [idx[i:i + bs] for i in range(0, len(idx), bs)]

    def loss(self, store, idx):
        feats = self.proc.feature_extractor([store.get(k) for k in idx], sampling_rate=16000,
                                            return_tensors="pt").input_features.to(self.device)
        labels = whisper_labels(self.proc.tokenizer, [store.rows[k]["text"] for k in idx], self.m["language"]).to(self.device)
        return self.model(input_features=feats, labels=labels).loss

    @torch.no_grad()
    def decode(self, store, idx, num_beams: int = 1) -> list[str]:
        self.model.eval()
        feats = self.proc.feature_extractor([store.get(k) for k in idx], sampling_rate=16000,
                                            return_tensors="pt").input_features.to(self.device)
        out = self.model.generate(feats, language=self.m["language"], task="transcribe", num_beams=num_beams,
                                  do_sample=False, max_new_tokens=440)
        return list(self.proc.batch_decode(out, skip_special_tokens=True))

    def eval_batches(self, store):
        order = sorted(range(len(store)), key=lambda k: (-store.samples(k), k))
        bs = self.m["eval_batch_size"]
        return [order[i:i + bs] for i in range(0, len(order), bs)]

    def save(self, path: Path):
        self.model.save_pretrained(path)

    def load(self, path: Path):
        from transformers import WhisperForConditionalGeneration
        self.model = WhisperForConditionalGeneration.from_pretrained(path).to(self.device)
        self.model.generation_config.forced_decoder_ids = None


class XLSR:
    def __init__(self, local: str, m: dict, device: str, train_texts: list[str], work: Path):
        from transformers import Wav2Vec2ForCTC
        self.vocab = xlsr_vocab(train_texts)
        self.proc = build_xlsr_processor(self.vocab, work / "processor")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            local, vocab_size=len(self.vocab), pad_token_id=self.vocab["[PAD]"], ctc_zero_infinity=True,
            ignore_mismatched_sizes=True, **m["model"]).to(device)
        self.model.freeze_feature_encoder()
        self.m, self.device = m, device

    def _inputs(self, store, idx):
        x = self.proc.feature_extractor([store.get(k) for k in idx], sampling_rate=16000, padding=True,
                                        return_attention_mask=True, return_tensors="pt")
        return x.input_values.to(self.device), x.attention_mask.to(self.device)

    def batches(self, store, rng):
        tc = self.m["train"]
        return frame_batches([store.samples(k) for k in range(len(store))], tc["max_samples"], tc["max_items"], rng)

    def loss(self, store, idx):
        x, mask = self._inputs(store, idx)
        labels = xlsr_labels(self.proc.tokenizer, [store.rows[k]["text"] for k in idx]).to(self.device)
        return self.model(input_values=x, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def decode(self, store, idx, **_) -> list[str]:
        self.model.eval()
        x, mask = self._inputs(store, idx)
        return ctc_decode_valid(self.model, self.proc, x, mask)

    def eval_batches(self, store):
        return frame_batches([store.samples(k) for k in range(len(store))], self.m["eval_max_samples"], 64, None)

    def save(self, path: Path):
        self.model.save_pretrained(path)
        self.proc.save_pretrained(path)

    def load(self, path: Path):
        from transformers import Wav2Vec2ForCTC
        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(self.device)


def decode_store(sysobj, store: WaveStore, device: str, opts: dict) -> tuple[dict[str, str], float]:
    out, seconds = {}, 0.0
    for idx in sysobj.eval_batches(store):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            hyps = sysobj.decode(store, idx, **opts)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        for k, h in zip(idx, hyps):
            out[store.ids[k]] = h
    return out, seconds


def dev_wer(sysobj, store, device, opts) -> float:
    hyps, _ = decode_store(sysobj, store, device, opts)
    return corpus_wer([r["text"] for r in store.rows], [hyps[i] for i in store.ids])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--base-dir", default=None, help="local copy of the base model instead of the Hub (tests)")
    ap.add_argument("--max-steps", type=int, default=None, help="for a short trial run")
    ap.add_argument("--limit-train", type=int, default=None, help="first N train clips, for a trial run")
    ap.add_argument("--limit-eval", type=int, default=None, help="first N dev/test/silence clips, for a trial run")
    ap.add_argument("--prepare-only", action="store_true",
                    help="only decode the audio to work/wav16k (run once before fine-tuning two systems in parallel)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    trial = bool(args.max_steps or args.limit_train or args.limit_eval or args.base_dir)
    if trial and out_dir.resolve() == (ROOT / "results").resolve():
        raise SystemExit("trial options need --out-dir outside results/")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.system not in cfg["models"]:
        raise SystemExit(f"{args.system} is not in {args.config}")
    m = copy.deepcopy(cfg["models"][args.system])
    tc = m["train"]
    seed_all(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = {s: sorted(read_jsonl(CLEAN / f), key=lambda r: r["id"]) for s, f in SPLITS.items()}
    if args.limit_train:
        rows["train"] = rows["train"][: args.limit_train]
    if args.limit_eval:
        for s in ("dev", "test", "silence"):
            rows[s] = rows[s][: args.limit_eval]
    stores = {s: WaveStore(rows[s], WORK, WORK / "wav16k") for s in SPLITS}
    if args.prepare_only:
        print("audio ready: " + ", ".join(f"{s} {len(v)}" for s, v in stores.items()), flush=True)
        return

    if args.base_dir:
        local, sha = args.base_dir, "local-test-model"
    else:
        from huggingface_hub import snapshot_download
        local = snapshot_download(m["base"], revision=m["revision"],
                                  allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "vocab*", "merges*"])
        sha = m["revision"]
    work = WORK / "models" / (args.system + ("_trial" if trial else ""))
    work.mkdir(parents=True, exist_ok=True)
    if m["kind"] == "whisper":
        sysobj = Whisper(local, m, device)
    else:
        sysobj = XLSR(local, m, device, [r["text"] for r in rows["train"]], work)
    model = sysobj.model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc["peak_lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(cfg["seed"])
    steps_per_epoch = len(sysobj.batches(stores["train"], np.random.default_rng(0)))
    total = args.max_steps or tc["epochs"] * steps_per_epoch
    log = {"system": args.system, "base": m["base"], "revision": sha, "steps_per_epoch": steps_per_epoch,
           "total_steps": total, "evals": []}
    best_wer, step, losses, t0 = float("inf"), 0, [], time.time()
    print(f"{args.system}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M parameters, {total} steps", flush=True)

    def evaluate(final: bool):
        nonlocal best_wer
        wer = dev_wer(sysobj, stores["dev"], device, {})
        entry = {"step": step, "epoch": round(step / steps_per_epoch, 3), "train_loss": float(np.mean(losses)) if losses else None,
                 "dev_wer_greedy": wer, "hours": round((time.time() - t0) / 3600, 3)}
        if wer < best_wer:
            best_wer = wer
            sysobj.save(work / "best")
            entry["saved"] = True
        log["evals"].append(entry)
        losses.clear()
        print(json.dumps(entry), flush=True)

    done = False
    while not done:
        for idx in sysobj.batches(stores["train"], rng):
            model.train()
            for g in opt.param_groups:
                g["lr"] = linear_schedule(step, tc["peak_lr"], tc["warmup_steps"], total)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = sysobj.loss(stores["train"], idx)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            step += 1
            out_of_time = (time.time() - t0) / 3600 >= tc["max_hours"]
            if step >= total or out_of_time:
                evaluate(final=True)
                done = True
                break
            if step % tc["eval_every_steps"] == 0:
                evaluate(final=False)

    sysobj.load(work / "best")
    chosen_step = max((e for e in log["evals"] if e.get("saved")), key=lambda e: e["step"])["step"]
    log["chosen_step"], log["chosen_dev_wer_greedy"] = chosen_step, best_wer

    grid = m.get("dev_search") or {}
    keys = sorted(grid)
    search = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        opts = dict(zip(keys, vals))
        search.append({**opts, "dev_wer": dev_wer(sysobj, stores["dev"], device, opts)})
        print(json.dumps(search[-1]), flush=True)
    chosen = min(search, key=lambda s: s["dev_wer"]) if search else {}
    opts = {k: v for k, v in chosen.items() if k != "dev_wer"}
    log["dev_search"], log["chosen_decoding"] = search, opts

    timing = {}
    pred_dir = out_dir / "predictions"
    for split in ("dev", "test", "silence"):
        hyps, seconds = decode_store(sysobj, stores[split], device, opts)
        write_jsonl(pred_dir / f"{args.system}.{split}.jsonl", [{"id": r["id"], "hyp": hyps[r["id"]]} for r in rows[split]])
        timing[split] = {"clips": len(rows[split]), "audio_seconds": round(sum(r["audio_length"] for r in rows[split]), 2),
                         "compute_seconds": round(seconds, 2)}
    hours = sum(r["audio_length"] for r in rows["train"]) / 3600
    meta = {
        "kind": "ours", "system": args.system, "family": m["family"],
        "setup": (f"`{m['base']}` @ `{sha[:10]}` fine-tuned on our train split ({hours:.1f} h), "
                  f"step {chosen_step} of {step} chosen on dev"),
        "checkpoint": None, "base_checkpoint": m["base"], "base_revision": sha, "revision": None,
        "language": m["language"], "decoding": opts or {"decoding": "greedy"}, "train_settings": tc,
        "model_settings": m.get("model"), "chosen_step": chosen_step, "steps_run": step,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "precision": "fp16 mixed (torch.autocast)" if device == "cuda" else "fp32",
        "batch_size": tc.get("batch_size", f"up to {tc.get('max_samples')} padded samples"),
        "audio_seconds": timing["test"]["audio_seconds"], "compute_seconds": timing["test"]["compute_seconds"],
        "timing_split": "test", "timing_note": "feature extraction, model forward and decoding (audio already decoded)",
        "timing": timing, "seed": cfg["seed"], "versions": versions(), "trial": trial,
    }
    (pred_dir / f"{args.system}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "train_logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "train_logs" / f"{args.system}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if trial:
        shutil.rmtree(work / "best", ignore_errors=True)
    print(json.dumps({"chosen_step": chosen_step, "chosen_decoding": opts, "timing": timing}), flush=True)


if __name__ == "__main__":
    main()
