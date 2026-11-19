"""Step 6: speaker- and session-disjoint train / dev / test split (see hrasr/split.py).

Output:
  data/clean/{train,dev,test}.jsonl   the cleaned dataset
  data/clean/split_stats.json         speakers, sessions, utterances, hours per split;
                                      dev/test sentences that also appear word for word in train
  data/clean/split_log.csv            clips removed because their session is in a higher split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import CLEAN, WORK, load_config  # noqa: E402
from hrasr.manifest import write_jsonl  # noqa: E402
from hrasr.split import assign_splits  # noqa: E402

FIELDS = ["id", "audio", "text", "text_norm", "audio_length", "speaker_name", "speaker_gender", "date", "session"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = pd.read_parquet(WORK / "clean_kept.parquet")
    df, stats, removed = assign_splits(df, cfg)
    removed.to_csv(CLEAN / "split_log.csv", index=False)
    stats["seed"] = cfg["seed"]
    stats["rule"] = ("speaker-disjoint and session-disjoint; dev/test speakers are not in ParlaSpeech-HR v1.0 "
                     "and first speak on or after " + cfg["selection"]["new_speaker_first_seen_on_or_after"])
    for name in ("train", "dev", "test"):
        part = df[df["split"] == name].sort_values("id")
        write_jsonl(CLEAN / f"{name}.jsonl", part[FIELDS].to_dict("records"))
    (CLEAN / "split_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
