"""Step 3: choose the clips to download (rules: hrasr/selection.py).

Output:
  work/selection.parquet              chosen clips with role "train" or "eval"
  data/raw/selection_report.json      hours, speakers, sessions, row groups to download
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import RAW, WORK, load_config  # noqa: E402
from hrasr.selection import select  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    v1_file = RAW / "v1_speakers.json"
    if not v1_file.exists():
        raise SystemExit("data/raw/v1_speakers.json is missing: run scripts/02_fetch_v1_speakers.py first.")
    v1 = set(json.loads(v1_file.read_text(encoding="utf-8"))["speakers"])

    meta = pd.read_parquet(WORK / "metadata.parquet")
    sel, report = select(meta, cfg, excluded_eval_speakers=v1)
    sel.to_parquet(WORK / "selection.parquet", index=False)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "selection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if "warning" in report:
        print("WARNING:", report["warning"], file=sys.stderr)


if __name__ == "__main__":
    main()
