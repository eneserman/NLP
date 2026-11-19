"""Step 2: list every speaker in ParlaSpeech-HR v1.0.

Several published Croatian baselines were trained on v1.0 (classla wav2vec2, NVIDIA stt_hr_*).
A dev/test speaker who is in v1.0 would make those baselines look better than they are, so
step 3 never puts these speakers in dev or test.

The v1.0 metadata file (CLARIN.SI, CC BY-SA 4.0) is read line by line; only speaker names are kept.

Output:
  data/raw/v1_speakers.json   {"source": url, "lines": n, "speakers": [...sorted names...]}
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import RAW, load_config  # noqa: E402

MIN_EXPECTED = 100  # v1.0 documents 309 speakers; far fewer means the parsing went wrong


def find_names(obj, out: set[str]) -> None:
    """Collect values of keys called 'speaker_name' or 'name' inside any 'speaker*' object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if kl == "speaker_name" and isinstance(v, str) and v.strip():
                out.add(v.strip())
            elif kl.startswith("speaker") and isinstance(v, dict):
                for kk, vv in v.items():
                    if kk.lower() in ("name", "speaker_name") and isinstance(vv, str) and vv.strip():
                        out.add(vv.strip())
                find_names(v, out)
            elif isinstance(v, (dict, list)):
                find_names(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_names(v, out)


def parse_lines(lines) -> tuple[set[str], int]:
    names: set[str] = set()
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        n += 1
        find_names(json.loads(line), names)
    return names, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--local-file", default=None, help="read a local copy of the v1.0 jsonl instead")
    args = ap.parse_args()
    cfg = load_config(args.config)
    url = cfg["selection"]["v1_metadata_url"]

    if args.local_file:
        with open(args.local_file, encoding="utf-8") as f:
            names, n = parse_lines(f)
        source = args.local_file
    else:
        with urllib.request.urlopen(url, timeout=120) as resp:
            names, n = parse_lines(line.decode("utf-8") for line in resp)
        source = url
    if len(names) < MIN_EXPECTED and not args.local_file:
        raise SystemExit(f"Only {len(names)} speaker names found in {n} lines; check the file format.")

    RAW.mkdir(parents=True, exist_ok=True)
    out = {"source": source, "lines": n, "speakers": sorted(names)}
    (RAW / "v1_speakers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(names)} v1.0 speakers from {n} lines")


if __name__ == "__main__":
    main()
