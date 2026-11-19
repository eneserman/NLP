"""Read and write JSON Lines manifests (plain or gzip)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator


def _open(path: Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with _open(Path(path), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
