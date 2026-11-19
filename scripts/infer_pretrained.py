"""Run a published checkpoint from configs/baselines.yaml on dev, test and the no-speech clips.

Usage:
  python scripts/infer_pretrained.py --system whisper_small [--splits dev test silence]
                                     [--batch-size 16] [--limit 20 --out-dir /tmp/trial]

Output (default folder results/predictions/):
  <system>.<split>.jsonl     {"id": ..., "hyp": ...} per clip; split "silence" = data/clean/silence_probe.jsonl
  <system>.meta.json         kind, family, checkpoint, revision, language, decoding, GPU, precision,
                             batch size, audio_seconds and compute_seconds (of the test split, used for
                             RTFx by scripts/score.py), and the timing of every split

--limit takes the first N clips of each split (sorted by id). It is only for a short trial run and
must write to another folder: scripts/score.py refuses predictions that miss clips.
Audio is read from work/audio/ (run scripts/unpack_audio.py first).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.config import CLEAN, ROOT, WORK, load_config  # noqa: E402
from hrasr.inference import download, make_backend, prepare_wavs, run_split  # noqa: E402
from hrasr.manifest import read_jsonl, write_jsonl  # noqa: E402

SPLIT_FILES = {"dev": "dev.jsonl", "test": "test.jsonl", "silence": "silence_probe.jsonl"}


def baseline_entry(system: str) -> dict:
    listed = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"))["baselines"]
    entry = next((b for b in listed if b["system"] == system), None)
    if entry is None:
        raise SystemExit(f"{system} is not in configs/baselines.yaml")
    if entry.get("trained_by_us") or entry.get("status") != "planned" or "backend" not in entry:
        raise SystemExit(f"{system} is not a published checkpoint that can be run (status {entry.get('status')})")
    return entry


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "nemo_toolkit", "huggingface_hub"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "silence"], choices=list(SPLIT_FILES))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--revision", default=None, help="model commit; default: the current one, resolved to a SHA")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "predictions"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.limit and out_dir.resolve() == (ROOT / "results" / "predictions").resolve():
        raise SystemExit("--limit is for trial runs; use --out-dir outside results/predictions")

    import torch
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    entry = baseline_entry(args.system)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local, sha = download(entry, args.revision)
    backend = make_backend(entry, local, device)

    timing = {}
    for split in args.splits:
        rows = sorted(read_jsonl(CLEAN / SPLIT_FILES[split]), key=lambda r: r["id"])
        if args.limit:
            rows = rows[: args.limit]
        wavs = prepare_wavs(rows, WORK, WORK / "wav16k", cfg["audio"]["sample_rate"])
        preds, seconds = run_split(backend, rows, wavs, args.batch_size)
        write_jsonl(out_dir / f"{args.system}.{split}.jsonl", preds)
        audio = sum(r["audio_length"] for r in rows)
        timing[split] = {"clips": len(rows), "audio_seconds": round(audio, 2), "compute_seconds": round(seconds, 2)}
        print(f"{args.system} {split}: {len(rows)} clips, {audio:.0f} s audio, {seconds:.1f} s compute")

    main_split = "test" if "test" in timing else next(iter(timing))
    meta = {
        "kind": "baseline",
        "system": args.system,
        "family": entry["family"],
        "checkpoint": entry["checkpoint"],
        "revision": sha,
        "language": entry["language"],
        "backend": entry["backend"],
        **backend.settings(),
        "gpu": gpu_name(),
        "precision": backend.precision,
        "batch_size": args.batch_size,
        "audio_seconds": timing[main_split]["audio_seconds"],
        "compute_seconds": timing[main_split]["compute_seconds"],
        "timing_split": main_split,
        "timing": timing,
        "seed": cfg["seed"],
        "versions": versions(),
        "limit": args.limit,
    }
    (out_dir / f"{args.system}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
