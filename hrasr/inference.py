"""Inference with published checkpoints (configs/baselines.yaml, trained_by_us: false).

* Every system reads the same samples: each clip is decoded once with `hrasr.audio.decode`
  (ffmpeg, 16 kHz mono) and written as a 16-bit WAV file; all backends read these files.
* Checkpoints are pinned to a Hugging Face commit SHA. The SHA goes into the system's .meta.json.
* compute_seconds is the wall time of the model calls only (no model loading, no audio decoding),
  with the GPU synchronized before the clock is read.

Backends:
  nemo        NeMo ASR models (CTC, transducer, hybrid, Canary, Parakeet), `ASRModel.transcribe`
  hf_ctc      Hugging Face wav2vec 2.0 + CTC, greedy (argmax) decoding
  hf_whisper  Hugging Face Whisper, language forced, task "transcribe"
"""
from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from hrasr.audio import decode


# ---------------------------------------------------------------- audio ----------------------------
def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    # Inverse of hrasr.audio.decode (int16 / 32768), so decoded samples are stored without change.
    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit mono WAV")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def prepare_wavs(rows: list[dict], audio_root: Path, cache: Path, sr: int = 16000) -> dict[str, Path]:
    """Decode every clip once (same ffmpeg path for all systems) -> {clip id: wav path}."""
    out = {}
    for r in rows:
        dst = cache / f"{r['id']}.wav"
        if not dst.exists():
            write_wav(dst, decode(audio_root / r["audio"], sr), sr)
        out[r["id"]] = dst
    return out


def length_sorted(rows: list[dict]) -> list[dict]:
    """Longest first: batches hold clips of similar length (less padding). Output order does not matter,
    predictions are stored by clip id."""
    return sorted(rows, key=lambda r: (-r["audio_length"], r["id"]))


def batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def hyp_text(h) -> str:
    """NeMo returns str, Hypothesis objects, or (best, all) tuples depending on the model type."""
    if isinstance(h, (list, tuple)) and h and not isinstance(h, str):
        h = h[0]
    text = getattr(h, "text", h)
    return "" if text is None else str(text)


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------- backends ------------------------
class Backend:
    precision = "fp32"

    def transcribe(self, paths: list[Path]) -> list[str]:
        raise NotImplementedError

    def settings(self) -> dict:
        return {}


def ctc_decode_valid(model, proc, input_values, attention_mask) -> list[str]:
    """Greedy CTC decoding of a padded batch. Only the output frames of each clip's own audio are
    decoded: frames computed from padding would otherwise add characters to the shorter clips."""
    import torch
    with torch.inference_mode():
        logits = model(input_values, attention_mask=attention_mask).logits
        n = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).tolist()
    ids = logits.argmax(dim=-1).cpu().numpy()
    return [ctc_ids_to_text(row[:k].tolist(), proc.tokenizer) for row, k in zip(ids, n)]


def ctc_ids_to_text(ids: list[int], tok) -> str:
    """Best-path CTC: merge repeats, remove the blank (the pad token), drop other special tokens
    (<unk>, <s>, </s>), word delimiter -> space. Done here, not with tokenizer.decode, because some
    transformers versions drop the pad token before merging repeats when skip_special_tokens=True."""
    blank = tok.pad_token_id
    merged = [t for i, t in enumerate(ids) if i == 0 or t != ids[i - 1]]
    special = set(tok.all_special_ids) - {tok.convert_tokens_to_ids(tok.word_delimiter_token)}
    pieces = tok.convert_ids_to_tokens([t for t in merged if t != blank and t not in special])
    text = "".join(pieces).replace(tok.word_delimiter_token, " ")
    return " ".join(text.split())


class HFCTC(Backend):
    def __init__(self, model_dir: str, device: str):
        from transformers import AutoModelForCTC, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForCTC.from_pretrained(model_dir).to(device).eval()
        self.device = device

    def transcribe(self, paths):
        import torch
        waves = [read_wav(p) for p in paths]
        feats = self.proc(waves, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        with torch.inference_mode():
            return ctc_decode_valid(self.model, self.proc, feats.input_values.to(self.device),
                                    feats.attention_mask.to(self.device))

    def settings(self):
        return {"decoding": "greedy CTC (argmax), no language model"}


class HFWhisper(Backend):
    def __init__(self, model_dir: str, device: str, language: str, num_beams: int = 1):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device).eval()
        self.device, self.language, self.num_beams = device, language, num_beams

    def transcribe(self, paths):
        import torch
        waves = [read_wav(p) for p in paths]
        feats = self.proc(waves, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with torch.inference_mode():
            out = self.model.generate(feats, language=self.language, task="transcribe",
                                      num_beams=self.num_beams, do_sample=False, max_new_tokens=440)
        return list(self.proc.batch_decode(out, skip_special_tokens=True))

    def settings(self):
        return {"decoding": f"beam search, num_beams={self.num_beams}" if self.num_beams > 1 else "greedy",
                "task": "transcribe", "max_new_tokens": 440, "timestamps": False}


class NeMo(Backend):
    def __init__(self, nemo_file: str, device: str, entry: dict):
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.restore_from(nemo_file, map_location=device).eval()
        self.entry = entry
        self.kwargs: dict = {}
        if entry.get("decoder"):
            self.model.change_decoding_strategy(decoder_type=entry["decoder"])
        if entry["system"].startswith("canary"):
            self.kwargs = {"source_lang": entry["language"], "target_lang": entry["language"]}

    def transcribe(self, paths):
        out = self.model.transcribe([str(p) for p in paths], batch_size=len(paths), verbose=False, **self.kwargs)
        if isinstance(out, tuple):  # older hybrid/RNNT models return (best, all)
            out = out[0]
        return [hyp_text(h) for h in out]

    def settings(self):
        cfg = getattr(self.model.cfg, "decoding", None)
        strategy = str(cfg.get("strategy")) if cfg is not None and hasattr(cfg, "get") else "model default"
        s = {"decoding": f"NeMo model default ({strategy})", "transcribe_kwargs": self.kwargs}
        if self.entry.get("decoder"):
            s["decoder_type"] = self.entry["decoder"]
        return s


def download(entry: dict, revision: str | None) -> tuple[str, str]:
    """-> (local path, resolved commit SHA). NeMo backends get the .nemo file, others a snapshot folder."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    info = HfApi().model_info(entry["checkpoint"], revision=revision)
    sha = info.sha
    if entry["backend"] == "nemo":
        name = next(s.rfilename for s in info.siblings if s.rfilename.endswith(".nemo"))
        return hf_hub_download(entry["checkpoint"], name, revision=sha), sha
    path = snapshot_download(entry["checkpoint"], revision=sha,
                             allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "vocab*", "merges*"])
    return path, sha


def make_backend(entry: dict, local: str, device: str) -> Backend:
    if entry["backend"] == "hf_ctc":
        return HFCTC(local, device)
    if entry["backend"] == "hf_whisper":
        return HFWhisper(local, device, entry["language"], int(entry.get("num_beams", 1)))
    if entry["backend"] == "nemo":
        return NeMo(local, device, entry)
    raise ValueError(f"unknown backend {entry['backend']}")


def run_split(backend: Backend, rows: list[dict], wavs: dict[str, Path], batch_size: int) -> tuple[list[dict], float]:
    """-> (predictions [{"id", "hyp"}], compute seconds)."""
    preds, seconds = [], 0.0
    for b in batches(length_sorted(rows), batch_size):
        _sync()
        t0 = time.perf_counter()
        hyps = backend.transcribe([wavs[r["id"]] for r in b])
        _sync()
        seconds += time.perf_counter() - t0
        if len(hyps) != len(b):
            raise RuntimeError(f"backend returned {len(hyps)} outputs for {len(b)} clips")
        preds += [{"id": r["id"], "hyp": h} for r, h in zip(b, hyps)]
    return sorted(preds, key=lambda p: p["id"]), seconds
