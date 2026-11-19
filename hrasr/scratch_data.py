"""Data for the models we train ourselves.

* Features are computed once per split and kept in one float16 array (frames x dims) plus offsets:
  audio is decoded with hrasr.audio.decode (the same ffmpeg path as every other system), then
  log-mel (neural models) or MFCC (GMM-HMM), then per-utterance CMVN.
* Batches are built by total frames (similar lengths together), shuffled with a fixed seed.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from hrasr.audio import decode
from hrasr.features import cmvn, log_mel, mfcc


def utterance_features(wave: np.ndarray, kind: str) -> np.ndarray:
    w = torch.from_numpy(wave)
    f = log_mel(w) if kind == "logmel" else mfcc(w)
    return cmvn(f).numpy().astype(np.float16)


class FeatureStore:
    """All features of one split: data (N_frames, D) float16, offsets (N+1,), ids."""

    def __init__(self, data: np.ndarray, offsets: np.ndarray, ids: list[str]):
        self.data, self.offsets, self.ids = data, offsets, ids
        self.index = {i: k for k, i in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def frames(self, k: int) -> int:
        return int(self.offsets[k + 1] - self.offsets[k])

    def get(self, k: int) -> np.ndarray:
        return self.data[self.offsets[k]:self.offsets[k + 1]]

    @classmethod
    def build(cls, rows: list[dict], audio_root: Path, kind: str, workers: int = 4, sr: int = 16000) -> "FeatureStore":
        def one(r):
            return utterance_features(decode(audio_root / r["audio"], sr), kind)
        with ThreadPoolExecutor(workers) as ex:
            feats = list(ex.map(one, rows))
        offsets = np.zeros(len(feats) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(f) for f in feats])
        return cls(np.concatenate(feats, 0) if feats else np.zeros((0, 1), np.float16), offsets, [r["id"] for r in rows])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.data)
        np.save(path.with_suffix(".offsets.npy"), self.offsets)
        path.with_suffix(".ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureStore":
        return cls(np.load(path.with_suffix(".npy"), mmap_mode="r"), np.load(path.with_suffix(".offsets.npy")),
                   json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8")))

    @classmethod
    def cached(cls, path: Path, rows: list[dict], audio_root: Path, kind: str, workers: int = 4) -> "FeatureStore":
        if path.with_suffix(".ids.json").exists():
            store = cls.load(path)
            if store.ids == [r["id"] for r in rows]:
                return store
        store = cls.build(rows, audio_root, kind, workers)
        store.save(path)
        return cls.load(path)


def frame_batches(lengths: list[int], max_frames: int, max_items: int, rng: np.random.Generator | None) -> list[list[int]]:
    """Indices grouped so that (longest item x items) <= max_frames (padded size), similar lengths together.
    With rng: batch order is shuffled (training). Without: longest first (decoding)."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > max_frames or len(cur) >= max_items):
            batches.append(cur)
            cur, m = [], lengths[i]
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(store: FeatureStore, idx: list[int], targets: list[list[int]] | None = None):
    """-> x (B, T, D) float32, lengths (B,), and if targets: y (B, U) int64, y_len (B,)."""
    feats = [torch.from_numpy(np.asarray(store.get(i), dtype=np.float32)) for i in idx]
    lengths = torch.tensor([len(f) for f in feats])
    x = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    if targets is None:
        return x, lengths
    ys = [torch.tensor(targets[i], dtype=torch.long) for i in idx]
    y_len = torch.tensor([len(y) for y in ys])
    y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True) if ys else torch.zeros(0, 0, dtype=torch.long)
    return x, lengths, y, y_len
