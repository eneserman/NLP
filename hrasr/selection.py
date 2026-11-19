"""Choose which clips to download from the 3,061-hour corpus.

Goals:
* Many different speakers (hour caps per speaker).
* An "eval" pool for development and test: speakers who are NOT in ParlaSpeech-HR v1.0 (the
  training data of several published baselines) and who first speak on or after a cutoff date.
  Test is filled first, then dev, alternating genders. A dev or test speaker is never in train.
* Session separation: a dev clip never comes from a test session, and a train clip never comes
  from a dev or test session.
* Targets are counted in audio that is expected to survive cleaning. The metadata-only rules
  (hrasr/cleaning.py, check_audio=False) predict this. Clips that fail these rules are still
  downloaded from the same speakers (at most max_raw_factor times the speaker's expected-clean
  audio), so the raw data keeps its real problems and step 5 removes them with a logged reason.
* Low download cost: audio is stored in Parquet row groups, so clips are taken from a limited
  set of row groups instead of single clips scattered over the whole corpus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hrasr.cleaning import FEATURE_COLUMNS, clip_reasons, session_of, word_features
from hrasr.textnorm import clean_reference


def _norm_name(name) -> str:
    return " ".join(str(name).split()).casefold()


def add_columns(meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    meta = meta.copy()
    meta["speaker_gender"] = meta["speaker_gender"].fillna("unknown").replace("", "unknown")
    meta["session"] = [session_of(i, d) for i, d in zip(meta["id"], meta["date"])]
    tf = cfg["source"]["text_field"]
    if "wa_n" not in meta.columns:  # a table with full word lists instead of step-1 features
        feats = [word_features(w, d) for w, d in zip(meta["words"], meta["audio_length"])]
        for col, key in zip(FEATURE_COLUMNS, ("n", "lead", "trail", "inner", "relative")):
            meta[col] = [f[key] if f else (0 if key == "n" else None) for f in feats]
    meta["_good"] = [
        not clip_reasons(r, clean_reference(r.get(tf) or ""), None, cfg, check_audio=False)
        for r in meta[["speaker_name", "audio_length", tf, *FEATURE_COLUMNS]].to_dict("records")
    ]
    meta["_good_s"] = meta["audio_length"].where(meta["_good"], 0.0)
    return meta


def speaker_table(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta.groupby("speaker_name")
    return pd.DataFrame({
        "good_hours": g["_good_s"].sum() / 3600.0,
        "first_seen": g["date"].min(),
        "gender": g["speaker_gender"].agg(lambda s: s.mode().iat[0]),
    })


def _interleave(names: list[str], gender: pd.Series) -> list[str]:
    queues: dict[str, list[str]] = {}
    for n in names:
        queues.setdefault(gender[n], []).append(n)
    order: list[str] = []
    while any(queues.values()):
        for g in sorted(queues):
            if queues[g]:
                order.append(queues[g].pop(0))
    return order


def select(meta: pd.DataFrame, cfg: dict, excluded_eval_speakers: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    s = cfg["selection"]
    rng = np.random.default_rng(cfg["seed"])
    n_all = len(meta)
    meta = meta[meta["speaker_name"].fillna("").astype(str).str.len() > 0]
    report: dict = {"rows_in_corpus": n_all, "rows_without_speaker": n_all - len(meta)}
    meta = add_columns(meta, cfg)
    report["predicted_share_removed_by_metadata_rules"] = round(1 - float(meta["_good"].mean()), 4)

    spk = speaker_table(meta)
    excluded = {_norm_name(n) for n in (excluded_eval_speakers or set())}
    in_v1 = np.array([_norm_name(n) in excluded for n in spk.index], dtype=bool)
    cutoff = s["new_speaker_first_seen_on_or_after"]
    pool = spk[(~in_v1) & (spk["first_seen"] >= cutoff) & (spk["good_hours"] >= s["min_hours_per_eval_speaker"])]
    report.update({"speakers_in_corpus": int(len(spk)), "speakers_excluded_as_in_v1": int(in_v1.sum()),
                   "eval_pool_size": int(len(pool))})
    order = _interleave(list(rng.permutation(pool.index.to_numpy())), spk["gender"])

    # Sessions in which eval-pool speakers talk are split at random (fixed seed) into test sessions
    # and dev sessions. Test speakers take clips only from test sessions, dev speakers only from dev
    # sessions, so the two splits never share a session. A train clip may not come from a session
    # that gives a clip to dev or test.
    eval_rows = meta[meta["speaker_name"].isin(set(pool.index))]
    sessions = sorted(set(eval_rows["session"]))
    perm = rng.permutation(len(sessions))
    session_pool = {sessions[j]: ("test" if k % 2 == 0 else "dev") for k, j in enumerate(perm)}

    chosen: dict[int, str] = {}
    used = pd.Series(0.0, index=spk.index)
    raw_used = pd.Series(0.0, index=spk.index)
    eval_cap_s = s["max_hours_per_eval_speaker"] * 3600
    raw_factor = s["max_raw_factor"]
    by_speaker = {k: g for k, g in eval_rows.groupby("speaker_name")}
    eval_set: set[str] = set()
    eval_totals = {"test": 0.0, "dev": 0.0}
    queue = list(order)
    for split_name in ("test", "dev"):
        target_s = s[f"{split_name}_hours"] * s["oversample"] * 3600
        while queue and eval_totals[split_name] < target_s:
            name = queue.pop(0)
            rows = by_speaker[name]
            rows = rows[rows["session"].map(session_pool) == split_name]
            if rows.empty:
                continue
            # Densest sessions and row groups first: fewer sessions used, fewer row groups downloaded.
            rows = rows.assign(
                _ds=rows.groupby("session")["_good_s"].transform("sum"),
                _dr=rows.groupby(["file", "row_group"])["_good_s"].transform("sum"),
            ).sort_values(["_ds", "_dr", "file", "row_group"], ascending=[False, False, True, True], kind="stable")
            total, raw, groups = 0.0, 0.0, set()
            for idx, good_s, dur, key in zip(rows.index, rows["_good_s"], rows["audio_length"],
                                             zip(rows["file"], rows["row_group"])):
                if good_s and total + good_s <= eval_cap_s:
                    chosen[idx] = split_name
                    total += good_s
                    raw += dur
                    groups.add(key)
            if not total:
                continue
            # Keep some of this speaker's problem clips (same row groups) in the raw data.
            for idx, good_s, dur, key in zip(rows.index, rows["_good_s"], rows["audio_length"],
                                             zip(rows["file"], rows["row_group"])):
                if not good_s and key in groups and raw + dur <= total * raw_factor:
                    chosen[idx] = split_name
                    raw += dur
            used[name], raw_used[name] = total, raw
            eval_totals[split_name] += total
            eval_set.add(name)
    eval_total = sum(eval_totals.values())
    eval_target_s = (s["dev_hours"] + s["test_hours"]) * s["oversample"] * 3600
    eval_sessions = set(meta.loc[list(chosen), "session"])

    train_target_s = s["train_hours"] * s["oversample"] * 3600
    train_cap_s = s["max_hours_per_train_speaker"] * 3600
    train_total = 0.0

    def harvest(group: pd.DataFrame) -> None:
        nonlocal train_total
        group = group.loc[rng.permutation(group.index.to_numpy())]
        for idx, name, good_s, sess in zip(group.index, group["speaker_name"], group["_good_s"], group["session"]):
            if train_total >= train_target_s:
                return
            if idx in chosen or name in eval_set or sess in eval_sessions:
                continue
            dur = group.at[idx, "audio_length"]
            if good_s:
                if used[name] + good_s > train_cap_s:
                    continue
            elif raw_used[name] + dur > raw_factor * max(used[name], 60.0):
                continue  # problem clip: keep raw audio <= max_raw_factor x the speaker's clean audio
            chosen[idx] = "train"
            used[name] += good_s
            raw_used[name] += dur
            train_total += good_s

    groups = {k: g for k, g in meta.groupby(["file", "row_group"])}
    touched = sorted({(meta.at[i, "file"], meta.at[i, "row_group"]) for i in chosen})
    for key in touched:
        harvest(groups[key])
    rest = [k for k in sorted(groups) if k not in set(touched)]
    for j in rng.permutation(len(rest)):
        if train_total >= train_target_s:
            break
        harvest(groups[rest[j]])

    sel = meta.loc[sorted(chosen)].copy()
    sel["role"] = [chosen[i] for i in sel.index]
    sel["predicted_clean"] = sel["_good"]
    sel = sel.drop(columns=["_good", "_good_s"])
    report.update({
        "eval_speakers": len(eval_set),
        "test_speakers": int(sel.loc[sel["role"] == "test", "speaker_name"].nunique()),
        "dev_speakers": int(sel.loc[sel["role"] == "dev", "speaker_name"].nunique()),
        "test_expected_clean_hours": round(eval_totals["test"] / 3600, 3),
        "dev_expected_clean_hours": round(eval_totals["dev"] / 3600, 3),
        "train_speakers": int(sel.loc[sel["role"] == "train", "speaker_name"].nunique()),
        "train_expected_clean_hours": round(train_total / 3600, 3),
        "selected_clips": int(len(sel)),
        "selected_hours_raw": round(float(sel["audio_length"].sum()) / 3600, 3),
        "selected_clips_predicted_to_fail_cleaning": int((~sel["predicted_clean"]).sum()),
        "row_groups_to_download": int(sel.groupby(["file", "row_group"]).ngroups),
        "eval_sessions": len(eval_sessions),
    })
    if eval_total < 0.95 * eval_target_s:
        report["warning"] = (f"Only {eval_total/3600:.2f} h of expected-clean dev+test audio; target was "
                             f"{eval_target_s/3600:.2f} h. Lower the cutoff date or the eval hours.")
    return sel, report
