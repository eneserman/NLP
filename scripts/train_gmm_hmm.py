"""Train the MFCC + GMM-HMM baseline (configs/train_gmm_hmm.yaml), choose decoding settings on dev,
decode test once. CPU only.

Usage:
  python scripts/train_gmm_hmm.py [--workers 4]
  python scripts/train_gmm_hmm.py --from-model model.npz --trained-log run.log   # decoding only

The decoding setting is chosen on dev by `dev_select_by` (configs/train_gmm_hmm.yaml). With
--from-model, the model of an earlier run is decoded; its training iterations come from that run's
Kaggle log, and the dev WER of every setting it logged must come out again.

Output:
  results/predictions/gmm_hmm.{dev,test,silence}.jsonl and gmm_hmm.meta.json
  results/train_logs/gmm_hmm.json     per iteration: utterances aligned, average log-likelihood per frame;
                                      dev WER and CER of every decoding setting and the choice
  work/models/gmm_hmm/model.npz       the trained model (not in the repository)
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from importlib import metadata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import CLEAN, ROOT, WORK  # noqa: E402
from hrasr.gmm_hmm import GMMHMM, CharBigramLM  # noqa: E402
from hrasr.manifest import read_jsonl, write_jsonl  # noqa: E402
from hrasr.metrics import per_clip_counts  # noqa: E402
from hrasr.scratch_data import FeatureStore  # noqa: E402
from hrasr.textnorm import normalize_for_scoring  # noqa: E402
from hrasr.vocab import CharVocab  # noqa: E402

SPLITS = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl", "silence": "silence_probe.jsonl"}
SYSTEM = "gmm_hmm"
_G: dict = {}  # per-worker state


def _init(model_state, store_path, lm=None):
    _G["model"] = GMMHMM.from_state_dict(model_state)
    _G["store"] = FeatureStore.load(Path(store_path))
    _G["lm"] = lm


def _align(job):
    k, text, flat = job
    m, X = _G["model"], np.asarray(_G["store"].get(k), dtype=np.float64)
    units, optional = m.utterance_units(text)
    if flat:
        return k, m.flat_start_alignment(len(X), units, optional), 0.0
    ll = m.frame_loglik(X)
    path = m.align(ll, units, optional)
    score = float(ll[np.arange(len(X)), path].mean()) if path is not None else 0.0
    return k, path, score


def _decode(job):
    k, w, p = job
    X = np.asarray(_G["store"].get(k), dtype=np.float64)
    return k, _G["model"].decode(X, _G["lm"], w, p)


def corpus_wer(refs, hyps) -> float:
    return corpus_wer_cer(refs, hyps)[0]


def corpus_wer_cer(refs, hyps) -> tuple[float, float]:
    c = per_clip_counts(refs, hyps)
    return float(c["w_err"].sum() / c["w_n"].sum()), float(c["c_err"].sum() / c["c_n"].sum())


def from_kaggle_log(path: Path) -> dict:
    """What an earlier run of this script printed, read from its Kaggle log (a JSON list of
    {"stream_name", "time", "data"}): training iterations, the dev search and the chosen setting."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict = {"iterations": [], "dev_search": [], "chosen_decoding": None}
    lines = "".join(i.get("data", "") for i in items).splitlines()
    # Kaggle writes each line of a subprocess twice, one right after the other: keep one of each pair
    lines = [l for k, l in enumerate(lines) if k == 0 or l != lines[k - 1]]
    for line in lines:
        if line.startswith('{"gaussians"'):
            out["iterations"].append(json.loads(line))
        elif line.startswith('{"lm_weight"'):
            out["dev_search"].append(json.loads(line))
        elif line.startswith('{"chosen_decoding"'):
            out["chosen_decoding"] = json.loads(line)["chosen_decoding"]
    if not out["iterations"] or not out["dev_search"] or out["chosen_decoding"] is None:
        raise SystemExit(f"{path} is not the log of a complete run of scripts/train_gmm_hmm.py")
    return out


def decode_split(model, lm, store_path, n, w, p, workers):
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init, initargs=(model.state_dict(), str(store_path), lm)) as pool:
        res = dict(pool.map(_decode, [(k, w, p) for k in range(n)], chunksize=4))
    return res, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_gmm_hmm.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=None, help="first N train clips, for a trial run")
    ap.add_argument("--limit-eval", type=int, default=None, help="first N dev/test/silence clips, for a trial run")
    ap.add_argument("--from-model", default=None,
                    help="skip training: the model.npz that an earlier run of this script saved")
    ap.add_argument("--trained-log", default=None,
                    help="with --from-model: the Kaggle log of that run (its training iterations and dev search)")
    args = ap.parse_args()
    if bool(args.from_model) != bool(args.trained_log):
        raise SystemExit("--from-model and --trained-log go together")
    out_dir = Path(args.out_dir)
    trial = bool(args.limit_train or args.limit_eval)
    if trial and out_dir.resolve() == (ROOT / "results").resolve():
        raise SystemExit("trial options need --out-dir outside results/")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tc = cfg["train"]
    np.random.seed(cfg["seed"])

    rows = {s: sorted(read_jsonl(CLEAN / f), key=lambda r: r["id"]) for s, f in SPLITS.items()}
    if args.limit_train:
        rows["train"] = rows["train"][: args.limit_train]
    if args.limit_eval:
        for s in ("dev", "test", "silence"):
            rows[s] = rows[s][: args.limit_eval]
    texts = [normalize_for_scoring(r["text"]) for r in rows["train"]]
    vocab = CharVocab.from_texts(texts)
    tag = cfg["features"] + (f"_trial{args.limit_train}_{args.limit_eval}" if trial else "")
    paths = {s: WORK / "feats" / tag / s for s in SPLITS}
    for s in SPLITS:
        FeatureStore.cached(paths[s], rows[s], WORK, cfg["features"], args.workers)
    store = FeatureStore.load(paths["train"])

    model = GMMHMM(vocab.chars, dim=store.data.shape[1], var_floor=tc["var_floor"])
    log = {"system": SYSTEM, "units": model.units, "iterations": []}
    earlier = None
    if args.from_model:
        # Decoding only: the model and its training log come from an earlier run of this script.
        earlier = from_kaggle_log(Path(args.trained_log))
        with np.load(args.from_model) as d:
            model = GMMHMM.from_state_dict({k: d[k] for k in d.files})
        if model.units != log["units"]:
            raise SystemExit("the saved model has other units than this train split")
        log["iterations"] = earlier["iterations"]
        log["trained_in"] = {"model": str(args.from_model), "kaggle_log": Path(args.trained_log).as_posix(),
                             "earlier_dev_search": earlier["dev_search"],
                             "earlier_chosen_decoding": earlier["chosen_decoding"]}
    t_start = time.time()
    first = True
    for stage in ([] if args.from_model else tc["schedule"]):
        while model.means.shape[1] < stage["gaussians"]:
            model.split()
        for it in range(stage["iterations"]):
            jobs = [(k, texts[k], first) for k in range(len(store))]
            with Pool(args.workers, initializer=_init, initargs=(model.state_dict(), str(paths["train"]))) as pool:
                res = pool.map(_align, jobs, chunksize=8)
            ok = [(k, p, s) for k, p, s in res if p is not None]
            model.reestimate([store.get(k) for k, _, _ in ok], [p for _, p, _ in ok])
            entry = {"gaussians": int(model.means.shape[1]), "iteration": it + 1, "flat_start": first,
                     "aligned": len(ok), "skipped": len(res) - len(ok),
                     "avg_loglik_per_frame": (float(np.mean([s for _, _, s in ok])) if not first else None),
                     "hours": round((time.time() - t_start) / 3600, 3)}
            log["iterations"].append(entry)
            print(json.dumps(entry), flush=True)
            first = False
    ckpt = WORK / "models" / (SYSTEM + ("_trial" if trial else ""))
    ckpt.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt / "model.npz", **model.state_dict())

    lm = CharBigramLM(texts, vocab.chars, k=tc["lm_add_k"])
    refs_dev = [r["text"] for r in rows["dev"]]
    grid = cfg["dev_search"]
    select_by = cfg.get("dev_select_by", "wer")
    if select_by not in ("wer", "cer"):
        raise SystemExit(f"dev_select_by must be wer or cer, not {select_by}")
    search = []
    for w, p in itertools.product(grid["lm_weight"], grid["ins_penalty"]):
        hyps, _ = decode_split(model, lm, paths["dev"], len(rows["dev"]), w, p, args.workers)
        wer, cer = corpus_wer_cer(refs_dev, [hyps[k] for k in range(len(rows["dev"]))])
        search.append({"lm_weight": w, "ins_penalty": p, "dev_wer": wer, "dev_cer": cer})
        print(json.dumps(search[-1]), flush=True)
    if earlier is not None:
        # the loaded model must be the one of that run: (almost) the same dev WER for every setting it logged.
        # Not exactly: another CPU may sum in another order, which can flip a rare tie in the Viterbi search.
        for old in earlier["dev_search"]:
            new = next((s for s in search if s["lm_weight"] == old["lm_weight"] and s["ins_penalty"] == old["ins_penalty"]), None)
            if new is not None and abs(new["dev_wer"] - old["dev_wer"]) > 0.002:
                raise SystemExit(f"{args.from_model} does not give the logged dev WER: {new} vs {old}")
    chosen = min(search, key=lambda s: (s[f"dev_{select_by}"], s["lm_weight"], s["ins_penalty"]))
    log["dev_search"], log["dev_select_by"] = search, select_by
    log["chosen_decoding"] = {k: chosen[k] for k in ("lm_weight", "ins_penalty")}

    timing = {}
    pred_dir = out_dir / "predictions"
    for split in ("dev", "test", "silence"):
        hyps, seconds = decode_split(model, lm, paths[split], len(rows[split]), chosen["lm_weight"], chosen["ins_penalty"], args.workers)
        write_jsonl(pred_dir / f"{SYSTEM}.{split}.jsonl", [{"id": r["id"], "hyp": hyps[k]} for k, r in enumerate(rows[split])])
        timing[split] = {"clips": len(rows[split]), "audio_seconds": round(sum(r["audio_length"] for r in rows[split]), 2),
                         "compute_seconds": round(seconds, 2)}
    n_gauss = int(model.means.shape[1])
    meta = {
        "kind": "baseline", "trained_by_us": True, "system": SYSTEM, "family": "Classical: MFCC + GMM-HMM",
        "setup": (f"MFCC 39-dim, {len(model.units) - 1} character HMMs + silence, 3 states, {n_gauss} Gaussians per state, "
                  f"character bigram LM; trained from scratch on our train split"),
        "checkpoint": None, "revision": None, "language": "Croatian only (trained on our data)",
        "decoding": log["chosen_decoding"], "decoding_chosen_by": f"dev {select_by.upper()}",
        "trained_in": log.get("trained_in", {}).get("model") if earlier is not None else None,
        "train_settings": tc, "features": cfg["features"],
        "gpu": "cpu", "precision": "fp64 (numpy)", "batch_size": 1, "workers": args.workers,
        "audio_seconds": timing["test"]["audio_seconds"], "compute_seconds": timing["test"]["compute_seconds"],
        "timing_split": "test", "timing_note": f"Viterbi decoding on precomputed MFCC, {args.workers} CPU processes",
        "timing": timing, "seed": cfg["seed"], "trial": trial,
        "versions": {"python": platform.python_version(), "numpy": metadata.version("numpy")},
    }
    (pred_dir / f"{SYSTEM}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "train_logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "train_logs" / f"{SYSTEM}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"chosen_decoding": log["chosen_decoding"], "timing": timing}), flush=True)


if __name__ == "__main__":
    main()
