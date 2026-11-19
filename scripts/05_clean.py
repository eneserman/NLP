"""Step 5: turn the raw manifest into cleaned clips.

Fixes (the clip stays):   formatting noise in the transcript -> data/clean/text_fixes.csv
Removals (the clip goes): every rule in hrasr/cleaning.py     -> data/clean/cleaning_log.csv
Summary:                                                         data/clean/cleaning_report.json

The kept clips go to work/clean_kept.parquet; step 6 splits them.
After a fresh clone, the audio is unpacked from data/audio/ first if work/audio/ is missing,
so this step runs from the committed raw data alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.audio import decode, rms_dbfs  # noqa: E402
from hrasr.cleaning import REASONS, AudioInfo, clip_reasons, session_of  # noqa: E402
from hrasr.config import AUDIO_SHARDS, CLEAN, RAW, WORK, load_config  # noqa: E402
from hrasr.manifest import read_jsonl  # noqa: E402
from hrasr.textnorm import clean_reference, normalize_for_scoring  # noqa: E402

WARN_SHARE = 0.30


def probe(path: Path, sr: int) -> AudioInfo | None:
    try:
        wave = decode(path, sr)
    except RuntimeError:
        return None
    return AudioInfo(duration_s=len(wave) / sr, rms_dbfs=rms_dbfs(wave))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    sr = cfg["audio"]["sample_rate"]
    tf = cfg["source"]["text_field"]

    raw = list(read_jsonl(RAW / "manifest.jsonl.gz"))
    if raw and not (WORK / raw[0]["audio"]).exists():
        if not list(AUDIO_SHARDS.glob("shard_*.tar")):
            raise SystemExit("Audio not found in work/audio/ or data/audio/.")
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name("unpack_audio.py"))], check=True)
    with ThreadPoolExecutor(args.workers) as ex:
        infos = list(tqdm(ex.map(lambda r: probe(WORK / r["audio"], sr), raw), total=len(raw), desc="decode"))

    pred = {r["id"]: r.get("predicted_clean", True) for r in raw}
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    kept, removed, fixes = [], [], []
    any_reason: Counter = Counter()
    first_reason: Counter = Counter()
    diff_text_fields = 0

    for row, info in zip(raw, infos):
        original = row.get(tf) or ""
        if "text_normalised" in row and row.get("text_normalised") not in (None, original):
            diff_text_fields += 1
        text = clean_reference(original)
        reasons = clip_reasons(row, text, info, cfg)
        if row["id"] in seen_ids:
            reasons.insert(0, "duplicate_id")
        elif row["audio_sha1"] in seen_audio:
            reasons.insert(0, "duplicate_audio")
        seen_ids.add(row["id"])
        seen_audio.add(row["audio_sha1"])

        if reasons:
            any_reason.update(reasons)
            first_reason[reasons[0]] += 1
            removed.append({"id": row["id"], "speaker_name": row.get("speaker_name") or "",
                            "duration_s": row["audio_length"], "reason": reasons[0],
                            "all_reasons": ";".join(reasons), "text": original})
            continue
        if text != original:
            fixes.append({"id": row["id"], "before": original, "after": text})
        kept.append({
            "id": row["id"], "audio": row["audio"], "text": text, "text_norm": normalize_for_scoring(text),
            "audio_length": info.duration_s if info else row["audio_length"],
            "speaker_name": row["speaker_name"], "speaker_gender": row.get("speaker_gender") or "",
            "date": row["date"], "role": row["role"],
            "session": row.get("session") or session_of(row["id"], row["date"]),
        })

    CLEAN.mkdir(parents=True, exist_ok=True)
    with open(CLEAN / "cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "speaker_name", "duration_s", "reason", "all_reasons", "text"])
        w.writeheader()
        w.writerows(removed)
    with open(CLEAN / "text_fixes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "before", "after"])
        w.writeheader()
        w.writerows(fixes)
    pd.DataFrame(kept).to_parquet(WORK / "clean_kept.parquet", index=False)

    h_raw = sum(r["audio_length"] for r in raw) / 3600
    h_kept = sum(r["audio_length"] for r in kept) / 3600
    report = {
        "raw_clips": len(raw), "raw_hours": round(h_raw, 3),
        "kept_clips": len(kept), "kept_hours": round(h_kept, 3),
        "removed_clips": len(removed),
        "removed_hours": round(sum(r["duration_s"] for r in removed) / 3600, 3),
        "removed_by_first_reason": dict(first_reason.most_common()),
        "clips_with_reason_any": dict(any_reason.most_common()),
        "reason_definitions": REASONS,
        "transcripts_fixed": len(fixes),
        "selection_prediction_check": {
            "predicted_clean_but_removed": sum(1 for r in removed if pred.get(r["id"], False)),
            "predicted_removed_but_kept": sum(1 for k in kept if not pred.get(k["id"], True)),
        },
        "rows_where_text_differs_from_text_normalised": diff_text_fields,
        "config": cfg["cleaning"],
    }
    warnings = [f"rule '{k}' flags {v/len(raw):.0%} of clips; check the threshold"
                for k, v in any_reason.items() if len(raw) and v / len(raw) > WARN_SHARE]
    if warnings:
        report["warnings"] = warnings
    (CLEAN / "cleaning_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("reason_definitions", "config")}, indent=2, ensure_ascii=False))
    for w_ in warnings:
        print("WARNING:", w_, file=sys.stderr)


if __name__ == "__main__":
    main()
