#!/usr/bin/env bash
# Raw corpus -> cleaned, split dataset. The same seed and config give the same result.
# Usage: bash scripts/build_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/01_fetch_metadata.py
python scripts/02_fetch_v1_speakers.py
python scripts/03_select_subset.py
python scripts/04_fetch_audio.py
python scripts/05_clean.py
python scripts/06_split.py
python scripts/07_make_silence_probe.py
python scripts/08_pack_audio.py
