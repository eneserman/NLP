"""Our contribution: fine-tuning pretrained models on our train split (configs/finetune.yaml).

* whisper: `openai/whisper-small`, all weights trained, targets = the cleaned orthographic transcript
  (`text`, with case and punctuation, as the Hugging Face Audio Course recommends; scoring normalizes
  both sides). Decoder prompt: <|startoftranscript|><|hr|><|transcribe|><|notimestamps|>.
* xlsr: `facebook/wav2vec2-xls-r-300m` + a new CTC head over our characters (targets = the
  scoring-normalized text, word boundary "|"); the convolutional feature encoder stays frozen.

Shared here: audio decoded once with hrasr.audio.decode (via hrasr.inference) and read from disk,
length-bucketed batches, the learning-rate schedule and the dev WER.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import torch

from hrasr.inference import prepare_wavs, read_wav
from hrasr.metrics import per_clip_counts
from hrasr.textnorm import normalize_for_scoring


class WaveStore:
    """Audio of a split: every clip decoded once to a 16-bit WAV file (hrasr.inference.prepare_wavs),
    read from disk when a batch needs it. Nothing is held in memory, so two training processes on the
    two GPUs share the operating system's file cache instead of holding 2 x 7 GB of audio."""

    def __init__(self, rows: list[dict], audio_root: Path, cache: Path, sr: int = 16000):
        wavs = prepare_wavs(rows, audio_root, cache, sr)
        self.rows = rows
        self.ids = [r["id"] for r in rows]
        self.paths = [wavs[i] for i in self.ids]
        self.lengths = []
        for p in self.paths:
            with wave.open(str(p), "rb") as w:
                self.lengths.append(w.getnframes())

    def __len__(self) -> int:
        return len(self.ids)

    def samples(self, k: int) -> int:
        return self.lengths[k]

    def get(self, k: int) -> np.ndarray:
        return read_wav(self.paths[k])


def linear_schedule(step: int, peak: float, warmup: int, total: int) -> float:
    """Linear warmup to `peak`, then linear decay to 0 at `total` steps."""
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, (total - step) / max(1, total - warmup))


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    c = per_clip_counts(refs, hyps)
    return float(c["w_err"].sum() / c["w_n"].sum())


# ------------------------------------------------------------------------------- whisper
def whisper_labels(tokenizer, texts: list[str], language: str, max_len: int = 448) -> torch.Tensor:
    """Token ids with the decoder prompt, the transcript and <|endoftext|>; padding = -100.
    The first token (<|startoftranscript|>) is removed: the model adds it when it shifts the labels."""
    tokenizer.set_prefix_tokens(language=language, task="transcribe", predict_timestamps=False)
    ids = [tokenizer(t).input_ids[:max_len] for t in texts]
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if any(x[0] != sot for x in ids):
        raise ValueError("expected labels to start with <|startoftranscript|>")
    ids = [x[1:] for x in ids]
    out = torch.full((len(ids), max(len(x) for x in ids)), -100, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x)
    return out


# ------------------------------------------------------------------------------- xls-r
def xlsr_vocab(texts: list[str]) -> dict[str, int]:
    """Characters of the normalized train text; space -> "|" (word delimiter); [PAD] = CTC blank, [UNK]."""
    chars = sorted({ch for t in texts for ch in normalize_for_scoring(t)} - {" "})
    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    for c in chars:
        vocab[c] = len(vocab)
    return vocab


def build_xlsr_processor(vocab: dict[str, int], out_dir: Path):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    tok = Wav2Vec2CTCTokenizer(str(out_dir / "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                  return_attention_mask=True)
    proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
    proc.save_pretrained(out_dir)
    return proc


def xlsr_labels(tokenizer, texts: list[str]) -> torch.Tensor:
    ids = [tokenizer(normalize_for_scoring(t)).input_ids for t in texts]
    out = torch.full((len(ids), max(1, max(len(x) for x in ids))), -100, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return out
