"""Step 4: download the chosen clips and write the raw manifest.

Each chosen row is copied from the source Parquet file as it is (transcripts, word alignment and
all speaker metadata), plus our selection columns (role, session, predicted_clean) and the
location of the row in the source (source_file, source_row_group).

Audio is stored as 16 kHz mono Opus (32 kbit/s) to keep the repository under GitHub's size
limits. The original audio bytes are hashed (SHA-1) before encoding, so exact duplicates can
be found in step 5. Re-running the script skips clips that are already on disk.

Output:
  work/audio/<clip>.opus           audio (packed into data/audio/ by step 8)
  data/raw/manifest.jsonl.gz       every downloaded clip with its original transcript and metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.audio import encode_opus  # noqa: E402
from hrasr.config import RAW, WORK, load_config  # noqa: E402
from hrasr.manifest import write_jsonl  # noqa: E402
from hrasr.source import Source  # noqa: E402

AUDIO_DIR = WORK / "audio"
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def audio_name(clip_id: str) -> str:
    return _SAFE.sub("_", clip_id) + ".opus"


def jsonable(v):
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def fetch_group(src: Source, rel: str, rg: int, ids: set[str], cfg: dict) -> dict[str, dict]:
    """All columns of the chosen rows of one row group; audio is encoded to Opus on disk."""
    pf = src.parquet(rel)
    tbl = pf.read_row_group(rg)
    out: dict[str, dict] = {}
    col = tbl.column("audio").combine_chunks()
    if pa.types.is_struct(col.type):  # Hugging Face Audio feature: struct<bytes, path>
        names = [f.name for f in col.type]
        blobs = col.field("bytes")
        paths = col.field("path") if "path" in names else None
    else:  # plain binary column
        blobs, paths = col, None
    other = [c for c in tbl.column_names if c != "audio"]
    ids_col = tbl.column("id").to_pylist()
    rows = tbl.select(other).to_pylist()
    for i, cid in enumerate(ids_col):
        if cid not in ids:
            continue
        data = blobs[i].as_py()
        name = audio_name(cid)
        dest = AUDIO_DIR / name
        rec = {k: jsonable(v) for k, v in rows[i].items()}
        rec.update({"audio": f"audio/{name}", "audio_sha1": hashlib.sha1(data).hexdigest(),
                    "source_audio_name": paths[i].as_py() if paths is not None else None})
        if not dest.exists():
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(encode_opus(data, cfg["audio"]["sample_rate"], cfg["audio"]["bitrate"]))
            tmp.rename(dest)
        out.setdefault(cid, []).append(rec)
    return out


def fetch_with_retry(src: Source, rel: str, rg: int, ids: set[str], cfg: dict, tries: int = 4) -> dict[str, list[dict]]:
    for attempt in range(tries):
        try:
            return fetch_group(src, rel, rg, ids, cfg)
        except (OSError, RuntimeError) as err:  # network errors and ffmpeg failures
            if attempt == tries - 1:
                raise
            print(f"retry {attempt + 1} for {rel} row group {rg}: {err}", file=sys.stderr)
            time.sleep(5 * 2 ** attempt)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)

    source_info = json.loads((RAW / "SOURCE.json").read_text(encoding="utf-8"))
    revision = source_info.get("resolved_sha") or cfg["source"]["revision"]
    src = Source(cfg["source"]["repo_id"], revision, args.local_dir)

    sel = pd.read_parquet(WORK / "selection.parquet")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(rel, int(rg), set(g["id"])) for (rel, rg), g in sel.groupby(["file", "row_group"])]

    found: dict[tuple[str, int, str], list[dict]] = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_with_retry, src, rel, rg, ids, cfg): (rel, rg) for rel, rg, ids in jobs}
        for fut in tqdm(futs, desc="row groups"):
            rel, rg = futs[fut]
            for cid, recs in fut.result().items():
                found[(rel, rg, cid)] = recs

    missing = set(sel["id"]) - {cid for _, _, cid in found}
    if missing:
        print(f"WARNING: {len(missing)} selected clips were not found in their row group.", file=sys.stderr)

    ours = ["role", "session", "predicted_clean"]

    def rows():
        # One output row per source row; a duplicated source ID stays duplicated (step 5 removes it).
        for (rel, rg, cid), g in sel.groupby(["file", "row_group", "id"], sort=True):
            for rec in found.get((rel, int(rg), cid), []):
                row = dict(rec)
                row.update({k: jsonable(g.iloc[0][k]) for k in ours})
                row["source_file"], row["source_row_group"] = rel, int(rg)
                yield row

    n = write_jsonl(RAW / "manifest.jsonl.gz", rows())
    print(f"raw manifest: {n} clips, {sel.loc[~sel['id'].isin(missing), 'audio_length'].sum()/3600:.2f} h")


if __name__ == "__main__":
    main()
