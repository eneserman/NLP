"""Step 8: pack the audio of the raw manifest and of the no-speech probe into tar shards under
90 MB (GitHub refuses files over 100 MiB).

Output:
  data/audio/shard_000.tar, shard_001.tar, ...
  data/audio/INDEX.csv     which shard holds which audio file
"""
from __future__ import annotations

import argparse
import csv
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import AUDIO_SHARDS, CLEAN, RAW, WORK, load_config  # noqa: E402
from hrasr.manifest import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    limit = cfg["audio"]["shard_max_mb"] * 1024 * 1024

    names = {r["audio"] for r in read_jsonl(RAW / "manifest.jsonl.gz")}
    probe = CLEAN / "silence_probe.jsonl"
    if probe.exists():
        names |= {r["audio"] for r in read_jsonl(probe)}
    names = sorted(names)
    AUDIO_SHARDS.mkdir(parents=True, exist_ok=True)
    for old in AUDIO_SHARDS.glob("shard_*.tar"):
        old.unlink()

    index, shard, size, tar = [], 0, 0, None
    for name in names:
        path = WORK / name
        n = path.stat().st_size + 1024  # tar header and padding
        if tar is None or size + n > limit:
            if tar:
                tar.close()
            tar = tarfile.open(AUDIO_SHARDS / f"shard_{shard:03d}.tar", "w")
            shard, size = shard + 1, 0
        tar.add(path, arcname=name)
        size += n
        index.append((name, f"shard_{shard - 1:03d}.tar"))
    if tar:
        tar.close()
    with open(AUDIO_SHARDS / "INDEX.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio", "shard"])
        w.writerows(index)
    print(f"{len(names)} files in {shard} shards")


if __name__ == "__main__":
    main()
