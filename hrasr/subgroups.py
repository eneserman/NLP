"""Subgroups of the reference clips, for scores per group.

Every group is defined from the reference data only (never from a system's output), so all
systems are scored on exactly the same clips in each group.

* speaking_rate: words per second of the normalized reference (`normalize_for_scoring`) divided
  by the clip length. The borders are the 1/3 and 2/3 quantiles of the scored split, so the three
  groups (slow, medium, fast) have about the same number of clips. The borders are reported.
* gender: the speaker's gender from the source metadata (`speaker_gender`).
* clip_length: fixed bins of the clip length in seconds (clips are 1-30 s after cleaning).
"""
from __future__ import annotations

import numpy as np

from hrasr.metrics import score
from hrasr.textnorm import normalize_for_scoring

LENGTH_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, float("inf")))
RATE_GROUPS = ("slow", "medium", "fast")


def words_per_second(row: dict) -> float:
    return len(normalize_for_scoring(row["text"]).split()) / row["audio_length"]


def rate_borders(rows: list[dict]) -> tuple[float, float]:
    rates = np.array([words_per_second(r) for r in rows])
    lo, hi = np.quantile(rates, [1 / 3, 2 / 3])
    return float(lo), float(hi)


def length_label(seconds: float) -> str:
    for lo, hi in LENGTH_BINS:
        if lo <= seconds < hi:
            return f"{lo:g}-{hi:g} s" if hi != float("inf") else f">= {lo:g} s"
    raise ValueError(f"negative clip length {seconds}")


def assign_groups(rows: list[dict]) -> tuple[dict[str, dict[str, str]], dict]:
    """Returns ({dimension: {clip id: group}}, {"speaking_rate_borders_wps": [lo, hi]})."""
    lo, hi = rate_borders(rows)
    groups: dict[str, dict[str, str]] = {"speaking_rate": {}, "gender": {}, "clip_length": {}}
    for r in rows:
        wps = words_per_second(r)
        groups["speaking_rate"][r["id"]] = "slow" if wps < lo else ("medium" if wps < hi else "fast")
        groups["gender"][r["id"]] = r.get("speaker_gender") or "unknown"
        groups["clip_length"][r["id"]] = length_label(r["audio_length"])
    return groups, {"speaking_rate_borders_wps": [round(lo, 3), round(hi, 3)]}


def group_order(dimension: str, labels: set[str]) -> list[str]:
    if dimension == "speaking_rate":
        return [g for g in RATE_GROUPS if g in labels]
    if dimension == "clip_length":
        order = [length_label(lo) for lo, _ in LENGTH_BINS]
        return [g for g in order if g in labels]
    return sorted(labels)


def subgroup_scores(refs: dict[str, str], hyps: dict[str, str], groups: dict[str, dict[str, str]],
                    iters: int = 1000, seed: int = 0) -> dict:
    """WER/CER per group with `hrasr.metrics.score` (same normalization and corpus-level counts)."""
    out: dict = {}
    for dim, by_id in groups.items():
        out[dim] = {}
        for g in group_order(dim, set(by_id.values())):
            ids = sorted(i for i, lab in by_id.items() if lab == g)
            out[dim][g] = score([refs[i] for i in ids], [hyps[i] for i in ids], iters, seed)
    return out
