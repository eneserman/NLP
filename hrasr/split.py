"""Speaker-disjoint and session-disjoint train / dev / test split.

Course rule (W9 slide "Assign related recordings together before splitting", Project 3 split
announcement): group clips by speaker and by recording session, then assign whole groups.

The groups are chosen in hrasr/selection.py, before the audio is downloaded, so that test and dev
get whole speakers from separate sessions (see the role column: test, dev or train). This module
applies that plan to the cleaned clips and enforces it:

1. A speaker is in exactly one split.
2. A session that has a test clip gives no clip to dev or train; a session that has a dev clip
   gives no clip to train. Any clip that breaks this is removed and logged in
   data/clean/split_log.csv (with the plan from selection this list is expected to be empty).
3. If any speaker or session is still in two splits, the script stops with an error.
"""
from __future__ import annotations

import pandas as pd

PRIORITY = ("test", "dev", "train")


def assign_splits(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Returns (kept clips with a `split` column, statistics, removed clips with a reason)."""
    df = df.copy()
    unknown = set(df["role"]) - set(PRIORITY)
    if unknown:
        raise ValueError(f"unknown roles {unknown}")
    df["split"] = df["role"]
    df["speaker_gender"] = df["speaker_gender"].fillna("unknown").replace("", "unknown")

    removed = []
    for i, higher in enumerate(PRIORITY[:-1]):
        sessions = set(df.loc[df["split"] == higher, "session"])
        lower = df["split"].isin(PRIORITY[i + 1:]) & df["session"].isin(sessions)
        part = df[lower].assign(reason=f"session_shared_with_{higher}")
        removed.append(part[["id", "speaker_name", "session", "split", "audio_length", "reason"]])
        df = df[~lower]
    removed_df = pd.concat(removed, ignore_index=True)

    check_disjoint(df)
    stats = split_stats(df)
    stats["removed_for_session_overlap"] = {
        "clips": int(len(removed_df)), "hours": round(float(removed_df["audio_length"].sum()) / 3600, 3)}
    return df, stats, removed_df


def check_disjoint(df: pd.DataFrame) -> None:
    for key in ("speaker_name", "session"):
        per = df.groupby(key)["split"].nunique()
        bad = per[per > 1]
        if len(bad):
            raise AssertionError(f"{key} values in more than one split: {list(bad.index)[:10]}")


def split_stats(df: pd.DataFrame) -> dict:
    stats: dict = {}
    for name, g in df.groupby("split"):
        stats[name] = {
            "speakers": int(g["speaker_name"].nunique()),
            "sessions": int(g["session"].nunique()),
            "utterances": int(len(g)),
            "hours": round(float(g["audio_length"].sum()) / 3600.0, 3),
            "speakers_by_gender": {str(k): int(v) for k, v in
                                   g.groupby("speaker_gender")["speaker_name"].nunique().items()},
        }
    train_sent = set(df.loc[df["split"] == "train", "text_norm"])
    for name in ("dev", "test"):
        if name in stats:
            sents = df.loc[df["split"] == name, "text_norm"]
            hit = int(sents.isin(train_sent).sum())
            stats[name]["sentences_also_in_train"] = hit
            stats[name]["sentences_also_in_train_pct"] = round(100.0 * hit / max(len(sents), 1), 2)
    return stats
