import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from hrasr.config import ROOT
from hrasr.inference import Backend, batches, hyp_text, length_sorted, prepare_wavs, read_wav, run_split, write_wav

sys.path.insert(0, str(ROOT / "scripts"))
import infer_pretrained  # noqa: E402


def test_wav_roundtrip_is_exact_at_16_bit(tmp_path):
    x = (np.arange(-32768, 32768, 7, dtype=np.int16).astype(np.float32) / 32768.0)
    write_wav(tmp_path / "a.wav", x, 16000)
    assert np.array_equal(read_wav(tmp_path / "a.wav"), x)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_prepare_wavs_decodes_once_with_ffmpeg(tmp_path):
    from hrasr.audio import decode, encode_opus
    from hrasr.inference import write_wav as ww
    sig = (0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    ww(tmp_path / "src.wav", sig, 16000)
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "c1.opus").write_bytes(encode_opus((tmp_path / "src.wav").read_bytes()))
    wavs = prepare_wavs([{"id": "c1", "audio": "audio/c1.opus"}], tmp_path, tmp_path / "cache")
    got = read_wav(wavs["c1"])
    ref = decode(tmp_path / "audio" / "c1.opus")
    assert np.allclose(got, ref, atol=1 / 32768)   # same samples as every other reader of the shards
    mtime = wavs["c1"].stat().st_mtime_ns
    prepare_wavs([{"id": "c1", "audio": "audio/c1.opus"}], tmp_path, tmp_path / "cache")
    assert wavs["c1"].stat().st_mtime_ns == mtime  # cached, not decoded again


def test_length_sorted_and_batches():
    rows = [{"id": i, "audio_length": L} for i, L in [("a", 2.0), ("b", 9.0), ("c", 5.0), ("d", 9.0)]]
    assert [r["id"] for r in length_sorted(rows)] == ["b", "d", "c", "a"]
    assert batches(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


def test_hyp_text_handles_nemo_outputs():
    assert hyp_text("dobar dan") == "dobar dan"
    assert hyp_text(SimpleNamespace(text="hvala")) == "hvala"
    assert hyp_text([SimpleNamespace(text="prvi"), SimpleNamespace(text="drugi")]) == "prvi"
    assert hyp_text(SimpleNamespace(text=None)) == ""


class EchoBackend(Backend):
    """Returns the file stem; records batch sizes."""
    def __init__(self):
        self.sizes = []

    def transcribe(self, paths):
        self.sizes.append(len(paths))
        return [p.stem for p in paths]


def test_run_split_keeps_ids_and_times(tmp_path):
    pytest.importorskip("torch")
    rows = [{"id": f"c{i}", "audio_length": float(i)} for i in range(5)]
    wavs = {r["id"]: tmp_path / f"{r['id']}.wav" for r in rows}
    be = EchoBackend()
    preds, seconds = run_split(be, rows, wavs, batch_size=2)
    assert be.sizes == [2, 2, 1]
    assert preds == [{"id": f"c{i}", "hyp": f"c{i}"} for i in range(5)]   # every clip, its own output, sorted by id
    assert seconds >= 0


def test_run_split_rejects_wrong_output_count(tmp_path):
    pytest.importorskip("torch")

    class Bad(Backend):
        def transcribe(self, paths):
            return ["x"]

    rows = [{"id": "a", "audio_length": 1.0}, {"id": "b", "audio_length": 1.0}]
    with pytest.raises(RuntimeError):
        run_split(Bad(), rows, {"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}, batch_size=2)


def test_hf_ctc_backend_with_tiny_random_model(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from hrasr.inference import HFCTC
    vocab = {"<pad>": 0, "|": 1, "<unk>": 2, "a": 3, "b": 4, "č": 5}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    tok = transformers.Wav2Vec2CTCTokenizer(str(tmp_path / "vocab.json"), word_delimiter_token="|")
    fe = transformers.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                               do_normalize=True, return_attention_mask=True)
    transformers.Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok).save_pretrained(tmp_path)
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                      intermediate_size=37, conv_dim=(8, 8), conv_stride=(5, 4), conv_kernel=(10, 8),
                                      num_feat_extract_layers=2, feat_extract_norm="layer", pad_token_id=0,
                                      do_stable_layer_norm=True)
    transformers.Wav2Vec2ForCTC(cfg).save_pretrained(tmp_path)
    rng = np.random.default_rng(0)
    paths = []
    for i, n in enumerate([16000, 8000]):
        write_wav(tmp_path / f"w{i}.wav", (0.1 * rng.standard_normal(n)).astype(np.float32), 16000)
        paths.append(tmp_path / f"w{i}.wav")
    be = HFCTC(str(tmp_path), "cpu")
    out = be.transcribe(paths)
    assert len(out) == 2 and all(isinstance(t, str) for t in out)
    assert set("".join(out)) <= set("abč ")         # only characters of the vocabulary
    # the padded frames of the shorter clip are not decoded: batched == alone
    assert out[1] == be.transcribe(paths[1:])[0]
    # greedy decoding: argmax, merge repeats, drop blank (<pad>) and <unk>, "|" -> space
    single = be.transcribe(paths[:1])[0]
    feats = be.proc([read_wav(paths[0])], sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        ids = be.model(feats.input_values).logits.argmax(-1)[0].tolist()
    inv = {i: c for c, i in vocab.items()}
    merged = [t for i, t in enumerate(ids) if i == 0 or t != ids[i - 1]]
    expected = " ".join("".join(inv[t] for t in merged if t not in (0, 2)).replace("|", " ").split())
    assert single == expected


def test_ctc_ids_to_text_keeps_repeats_separated_by_blank():
    transformers = pytest.importorskip("transformers")
    import json as _json
    import tempfile
    from pathlib import Path
    from hrasr.inference import ctc_ids_to_text
    vocab = {"<pad>": 0, "|": 1, "<unk>": 2, "a": 3, "b": 4}
    d = Path(tempfile.mkdtemp())
    (d / "v.json").write_text(_json.dumps(vocab), encoding="utf-8")
    tok = transformers.Wav2Vec2CTCTokenizer(str(d / "v.json"), word_delimiter_token="|")
    # a a <pad> a | | b <unk> b  ->  "aa bb"
    assert ctc_ids_to_text([3, 3, 0, 3, 1, 1, 4, 2, 4], tok) == "aa bb"
    assert ctc_ids_to_text([0, 0, 1, 0], tok) == ""


def test_every_runnable_baseline_has_backend_and_language():
    listed = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"))["baselines"]
    runnable = [b for b in listed if not b.get("trained_by_us") and b.get("status") == "planned"]
    assert {b["system"] for b in runnable} == {
        "nemo_hr_conformer_ctc", "nemo_hr_conformer_transducer", "nemo_hr_fastconformer_hybrid",
        "w2v2_classla_hr", "whisper_small", "canary_1b_v2", "parakeet_tdt_v3"}
    for b in runnable:
        assert b["backend"] in {"nemo", "hf_ctc", "hf_whisper"} and b["language"] and "/" in b["checkpoint"]
    langs = {b["system"]: b["language"] for b in runnable}
    assert langs["whisper_small"] == "hr" and langs["canary_1b_v2"] == "hr"


def test_infer_script_refuses_unrunnable_systems_and_trial_into_results():
    with pytest.raises(SystemExit):
        infer_pretrained.baseline_entry("qwen3_asr_1_7b")      # not_supported
    with pytest.raises(SystemExit):
        infer_pretrained.baseline_entry("gmm_hmm")              # trained by us
    assert infer_pretrained.baseline_entry("whisper_small")["checkpoint"] == "openai/whisper-small"
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "infer_pretrained.py"), "--system", "whisper_small",
                          "--limit", "2"], capture_output=True, text=True)
    assert out.returncode != 0 and "trial" in (out.stderr + out.stdout)
