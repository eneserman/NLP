"""Step 7: make a small no-speech test set (W9 slide 44: "Test silence instead of silently dropping it").

The reference of every clip is the empty string, so it is trusted by construction. Systems are
not scored with WER here (the denominator would be zero); scripts/score.py reports the number of
words each system emits and how many clips get any output (hallucination on non-speech).

Clips (fixed seed): digital silence, quiet white noise (-60 dBFS) and louder low-frequency
noise (-45 dBFS, like room hum), 20 of each, 2-10 s long.

Output:
  work/audio/silence/<id>.opus      (packed into data/audio/ by step 8)
  data/clean/silence_probe.jsonl
"""
from __future__ import annotations

import argparse
import io
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrasr.audio import encode_opus  # noqa: E402
from hrasr.config import CLEAN, WORK, load_config  # noqa: E402
from hrasr.manifest import write_jsonl  # noqa: E402

PER_KIND = 20


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def at_dbfs(x: np.ndarray, dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2))
    return x * (10 ** (dbfs / 20) / rms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    sr = cfg["audio"]["sample_rate"]
    rng = np.random.default_rng(cfg["seed"])
    out_dir = WORK / "audio" / "silence"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for kind in ("digital_silence", "white_noise_-60dBFS", "lowfreq_noise_-45dBFS"):
        for k in range(PER_KIND):
            dur = float(np.round(rng.uniform(2.0, 10.0), 2))
            n = int(dur * sr)
            if kind == "digital_silence":
                x = np.zeros(n)
            elif kind.startswith("white"):
                x = at_dbfs(rng.standard_normal(n), -60.0)
            else:
                # Brown noise (integrated white noise) with its slow drift removed: mostly low frequencies.
                x = np.cumsum(rng.standard_normal(n))
                x = at_dbfs(x - np.convolve(x, np.ones(800) / 800, mode="same"), -45.0)
            cid = f"silence_{kind.split('_')[0]}_{k:02d}"
            name = f"audio/silence/{cid}.opus"
            (WORK / name).write_bytes(encode_opus(wav_bytes(x, sr), sr, cfg["audio"]["bitrate"]))
            rows.append({"id": cid, "audio": name, "text": "", "kind": kind, "audio_length": dur})
    CLEAN.mkdir(parents=True, exist_ok=True)
    write_jsonl(CLEAN / "silence_probe.jsonl", rows)
    print(f"{len(rows)} no-speech clips, {sum(r['audio_length'] for r in rows):.1f} s")


if __name__ == "__main__":
    main()
