"""Step 1: read the metadata of every clip in the corpus, but no audio.

The word alignment of each clip is reduced to five numbers (wa_* columns: number of words and the
gaps before, after and between words), so that the table of ~900k clips fits in memory. The full
rows of the chosen clips, word alignment included, are copied into the raw manifest in step 4.

Output:
  work/metadata.parquet   one row per clip in the full corpus (cache, rebuilt by this script)
  data/raw/SOURCE.json    exact source: repo, resolved commit SHA, file list, totals
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.cleaning import word_features  # noqa: E402
from hrasr.config import RAW, WORK, load_config  # noqa: E402
from hrasr.source import Source  # noqa: E402

WANTED = ["id", "text", "text_normalised", "words", "audio_length", "date",
          "speaker_name", "speaker_gender", "speaker_birth", "speaker_party"]
REQUIRED = {"id", "audio_length", "date", "speaker_name", "speaker_gender", "words"}


def read_file(src: Source, rel: str, text_field: str) -> pd.DataFrame:
    pf = src.parquet(rel)
    names = set(pf.schema_arrow.names)
    missing = (REQUIRED | {text_field}) - names
    if missing:
        raise KeyError(f"{rel}: missing columns {sorted(missing)}; found {sorted(names)}")
    cols = [c for c in WANTED if c in names]
    parts = []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        feats = [word_features(w, d) for w, d in zip(df["words"], df["audio_length"])]
        df["wa_n"] = [f["n"] if f else 0 for f in feats]
        df["wa_lead"] = [f["lead"] if f else None for f in feats]
        df["wa_trail"] = [f["trail"] if f else None for f in feats]
        df["wa_inner"] = [f["inner"] if f else None for f in feats]
        df["wa_relative"] = [f["relative"] if f else None for f in feats]
        df = df.drop(columns=["words"])
        df["file"] = rel
        df["row_group"] = rg
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--local-dir", default=None, help="read Parquet files from a folder instead of the Hub")
    ap.add_argument("--limit-files", type=int, default=0, help="for a quick test run only")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)

    src = Source(cfg["source"]["repo_id"], cfg["source"]["revision"], args.local_dir)
    files = src.files()
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit("No Parquet files found.")

    with ThreadPoolExecutor(args.workers) as ex:
        dfs = list(tqdm(ex.map(lambda f: read_file(src, f, cfg["source"]["text_field"]), files),
                        total=len(files), desc="metadata"))
    meta = pd.concat(dfs, ignore_index=True)
    meta["date"] = meta["date"].astype(str)

    WORK.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(WORK / "metadata.parquet", index=False)
    info = {
        **src.describe(),
        "files": files,
        "rows": int(len(meta)),
        "hours": round(float(meta["audio_length"].sum()) / 3600.0, 2),
        "speakers": int(meta["speaker_name"].nunique()),
        "date_range": [str(meta["date"].min()), str(meta["date"].max())],
        "limited_test_run": bool(args.limit_files),
        "license": "CC BY-SA 4.0 (ParlaSpeech-HR 2.0, http://hdl.handle.net/11356/1914)",
    }
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "SOURCE.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
