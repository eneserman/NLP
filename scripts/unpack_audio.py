"""Extract data/audio/shard_*.tar into work/audio/. Run this once after cloning the repository;
every training and evaluation script reads audio from work/audio/."""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import AUDIO_SHARDS, WORK  # noqa: E402


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    try:
        tar.extractall(dest, filter="data")  # Python >= 3.11.4 / 3.12
    except TypeError:
        root = dest.resolve()
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not m.isfile() or root not in target.parents:
                raise RuntimeError(f"unexpected tar member: {m.name}")
        tar.extractall(dest)


def main() -> None:
    shards = sorted(AUDIO_SHARDS.glob("shard_*.tar"))
    if not shards:
        raise SystemExit("No shards in data/audio/.")
    WORK.mkdir(parents=True, exist_ok=True)
    for s in shards:
        with tarfile.open(s) as tar:
            safe_extract(tar, WORK)
    print(f"extracted {len(shards)} shards into {WORK / 'audio'}")


if __name__ == "__main__":
    main()
