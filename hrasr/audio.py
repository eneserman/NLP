"""Audio helpers. All decoding and encoding goes through ffmpeg, so every system reads
exactly the same samples (slide W9-14: resample for real, do not relabel the rate)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def _run(cmd: list[str], data: bytes | None = None) -> bytes:
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return proc.stdout


def encode_opus(audio_bytes: bytes, sample_rate: int = 16000, bitrate: str = "32k") -> bytes:
    """Any input format -> mono Opus in an Ogg container at `sample_rate`."""
    return _run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(sample_rate), "-c:a", "libopus", "-b:a", bitrate,
         "-map_metadata", "-1", "-f", "ogg", "pipe:1"],
        audio_bytes,
    )


def decode(src: bytes | str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode to mono float32 in [-1, 1] at `sample_rate`."""
    if isinstance(src, (str, Path)):
        cmd_in, data = ["-i", str(src)], None
    else:
        cmd_in, data = ["-i", "pipe:0"], src
    raw = _run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *cmd_in,
         "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"],
        data,
    )
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def rms_dbfs(wave: np.ndarray) -> float:
    if wave.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")
