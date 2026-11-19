"""Train the MFCC + GMM-HMM baseline (configs/train_gmm_hmm.yaml), choose decoding settings on dev,
decode test once. CPU only.

Usage:
  python scripts/train_gmm_hmm.py [--workers 4]

Output:
  results/predictions/gmm_hmm.{dev,test,silence}.jsonl and gmm_hmm.meta.json
  results/train_logs/gmm_hmm.json     per iteration: utterances aligned, average log-likelihood per frame;
                                      dev WER of every decoding setting and the choice
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
    c = per_clip_counts(refs, hyps)
    return float(c["w_err"].sum() / c["w_n"].sum())


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
    args = ap.parse_args()
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
    t_start = time.time()
    first = True
    for stage in tc["schedule"]:
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
    search = []
    for w, p in itertools.product(grid["lm_weight"], grid["ins_penalty"]):
        hyps, _ = decode_split(model, lm, paths["dev"], len(rows["dev"]), w, p, args.workers)
        search.append({"lm_weight": w, "ins_penalty": p, "dev_wer": corpus_wer(refs_dev, [hyps[k] for k in range(len(rows["dev"]))])})
        print(json.dumps(search[-1]), flush=True)
    chosen = min(search, key=lambda s: s["dev_wer"])
    log["dev_search"], log["chosen_decoding"] = search, {k: chosen[k] for k in ("lm_weight", "ins_penalty")}

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
        "decoding": log["chosen_decoding"], "train_settings": tc, "features": cfg["features"],
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
