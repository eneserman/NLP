"""WER and CER.

* Both strings are normalized with `normalize_for_scoring` first (one policy for all systems).
* Corpus scores add up edit counts over all clips and divide by all reference words or
  characters (slide W9-12). This is not the mean of per-clip percentages.
* CER counts spaces as characters, like jiwer and slide W9-10.
* A 95% bootstrap interval is computed by resampling clips.
"""
from __future__ import annotations

import jiwer
import numpy as np

from hrasr.textnorm import normalize_for_scoring


def per_clip_counts(refs: list[str], hyps: list[str]) -> dict[str, np.ndarray]:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    refs = [normalize_for_scoring(r) for r in refs]
    hyps = [normalize_for_scoring(h) for h in hyps]
    if any(not r for r in refs):
        raise ValueError("A reference is empty after normalization; the cleaned data should not allow this.")
    w_err, w_n, c_err, c_n = [], [], [], []
    sub = dele = ins = 0
    for r, h in zip(refs, hyps):
        wo = jiwer.process_words(r, h)
        co = jiwer.process_characters(r, h)
        w_err.append(wo.substitutions + wo.deletions + wo.insertions)
        w_n.append(wo.substitutions + wo.deletions + wo.hits)
        c_err.append(co.substitutions + co.deletions + co.insertions)
        c_n.append(co.substitutions + co.deletions + co.hits)
        sub += wo.substitutions
        dele += wo.deletions
        ins += wo.insertions
    return {
        "w_err": np.array(w_err), "w_n": np.array(w_n),
        "c_err": np.array(c_err), "c_n": np.array(c_n),
        "S": sub, "D": dele, "I": ins,
    }


def bootstrap_ci(err: np.ndarray, n: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(iters, len(err)))
    vals = err[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def score(refs: list[str], hyps: list[str], iters: int = 1000, seed: int = 0) -> dict:
    c = per_clip_counts(refs, hyps)
    wer = c["w_err"].sum() / c["w_n"].sum()
    cer = c["c_err"].sum() / c["c_n"].sum()
    wlo, whi = bootstrap_ci(c["w_err"], c["w_n"], iters, seed)
    clo, chi = bootstrap_ci(c["c_err"], c["c_n"], iters, seed)
    return {
        "clips": len(refs),
        "ref_words": int(c["w_n"].sum()),
        "wer": float(wer), "wer_ci95": [wlo, whi],
        "cer": float(cer), "cer_ci95": [clo, chi],
        "substitutions": int(c["S"]), "deletions": int(c["D"]), "insertions": int(c["I"]),
    }
