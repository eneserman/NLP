"""Paths and config loading."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# The environment variables only exist so that tests can run the pipeline in a temporary folder.
DATA = Path(os.environ.get("HRASR_DATA", ROOT / "data"))
RAW = DATA / "raw"
CLEAN = DATA / "clean"
AUDIO_SHARDS = DATA / "audio"
# Local working folder for large temporary files (full metadata scan, unpacked audio).
# It is not part of the dataset; every file in it is rebuilt by a script. See README.
WORK = Path(os.environ.get("HRASR_WORK", ROOT / "work"))


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "data.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
