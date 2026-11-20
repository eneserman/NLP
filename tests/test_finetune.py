import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from hrasr.config import ROOT  # noqa: E402
from hrasr.finetune import WaveStore, linear_schedule, whisper_labels, xlsr_labels, xlsr_vocab  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import finetune  # noqa: E402

WHISPER_REV = "973afd24965f72e36ca33b3055d56a652f456b4d"
TEXTS = ["Dobar dan.", "Hvala lijepa, Ivan."]


def fake_store(n_samples=(16000, 12000), texts=TEXTS, seed=0):
    """A WaveStore over WAV files that are already decoded (prepare_wavs finds them in its cache,
    so no ffmpeg is needed)."""
    import tempfile
    from pathlib import Path
    from hrasr.inference import write_wav
    rng = np.random.default_rng(seed)
    cache = Path(tempfile.mkdtemp())
    rows = [{"id": f"c{i}", "audio": f"audio/c{i}.opus", "text": t, "audio_length": n / 16000}
            for i, (n, t) in enumerate(zip(n_samples, texts))]
    for r, n in zip(rows, n_samples):
        write_wav(cache / f"{r['id']}.wav", (0.1 * rng.standard_normal(n)).astype(np.float32), 16000)
    return WaveStore(rows, cache / "no-audio-needed", cache)


def test_wave_store_reads_lengths_and_samples():
    s = fake_store((8000, 6000))
    assert len(s) == 2 and [s.samples(0), s.samples(1)] == [8000, 6000]
    x = s.get(1)
    assert x.dtype == np.float32 and len(x) == 6000 and float(np.abs(x).max()) <= 1.0


def test_linear_schedule():
    assert linear_schedule(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert linear_schedule(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert linear_schedule(55, 1.0, 10, 100) == pytest.approx(0.5)
    assert linear_schedule(100, 1.0, 10, 100) == 0.0


def test_xlsr_vocab_and_labels(tmp_path):
    from hrasr.finetune import build_xlsr_processor
    v = xlsr_vocab(["Čaša vode.", "Dobar dan!"])
    assert list(v)[:3] == ["[PAD]", "[UNK]", "|"] and "č" in v and " " not in v and "." not in v
    proc = build_xlsr_processor(v, tmp_path)
    y = xlsr_labels(proc.tokenizer, ["Dobar dan!", "Čaša."])
    assert proc.tokenizer.decode([i for i in y[0].tolist() if i >= 0]) == "dobar dan"
    assert int((y[1] == -100).sum()) == len("dobar dan") - len("čaša")


@pytest.fixture(scope="module")
def whisper_proc(tmp_path_factory):
    try:
        from transformers import WhisperProcessor
        proc = WhisperProcessor.from_pretrained("openai/whisper-small", revision=WHISPER_REV)
    except Exception as e:  # no internet
        pytest.skip(f"Whisper tokenizer not available: {e}")
    d = tmp_path_factory.mktemp("whisper_proc")
    proc.save_pretrained(d)
    return proc, d


def test_whisper_labels_have_croatian_prompt_and_eot(whisper_proc):
    proc, _ = whisper_proc
    y = whisper_labels(proc.tokenizer, TEXTS, "hr")
    tok = proc.tokenizer
    first = tok.convert_ids_to_tokens(y[0, :3].tolist())
    assert first == ["<|hr|>", "<|transcribe|>", "<|notimestamps|>"]       # <|startoftranscript|> removed
    ids0 = [i for i in y[0].tolist() if i >= 0]
    assert ids0[-1] == tok.eos_token_id
    assert tok.decode(ids0, skip_special_tokens=True) == "Dobar dan."
    assert int((y[0] == -100).sum()) > 0 and int((y[1] == -100).sum()) == 0


def _tiny_whisper(proc_dir, out):
    from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(proc_dir)
    tok = proc.tokenizer
    cfg = WhisperConfig(vocab_size=len(tok), d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=80,
                        max_source_positions=1500, max_target_positions=448,
                        decoder_start_token_id=tok.convert_tokens_to_ids("<|startoftranscript|>"),
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id)
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg)
    from transformers import GenerationConfig
    full = GenerationConfig.from_pretrained("openai/whisper-small", revision=WHISPER_REV)  # language and task ids
    model.generation_config = full
    model.save_pretrained(out)
    proc.save_pretrained(out)


def _train(sysobj, store, steps, lr, target_ratio=None):
    """Train on the two clips; stop early once loss < target_ratio * first loss (if given)."""
    opt = torch.optim.Adam(sysobj.model.parameters(), lr=lr)
    first = None
    for _ in range(steps):
        sysobj.model.train()
        loss = sysobj.loss(store, [0, 1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
        if target_ratio is not None and float(loss.detach()) < target_ratio * first:
            break
    return first, float(loss.detach())


def test_tiny_whisper_learns_the_transcripts(whisper_proc, tmp_path):
    _, proc_dir = whisper_proc
    _tiny_whisper(proc_dir, tmp_path / "tiny")
    m = {"language": "hr", "train": {"batch_size": 2}, "eval_batch_size": 2}
    sysobj = finetune.Whisper(str(tmp_path / "tiny"), m, "cpu")
    store = fake_store()
    first, last = _train(sysobj, store, 150, 3e-3)
    assert last < 0.05 * first
    assert sysobj.decode(store, [0, 1]) == TEXTS
    assert sysobj.decode(store, [0, 1], num_beams=2) == TEXTS
    sysobj.save(tmp_path / "best")
    sysobj.load(tmp_path / "best")
    assert sysobj.decode(store, [1]) == TEXTS[1:]


def test_tiny_xlsr_learns_the_transcripts(tmp_path):
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
    cfg = Wav2Vec2Config(vocab_size=32, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 8), num_feat_extract_layers=2,
                         feat_extract_norm="layer", do_stable_layer_norm=True, num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=4)
    torch.manual_seed(0)
    Wav2Vec2ForCTC(cfg).save_pretrained(tmp_path / "base")
    m = {"model": {"layerdrop": 0.0, "mask_time_prob": 0.0, "hidden_dropout": 0.0, "attention_dropout": 0.0,
                   "feat_proj_dropout": 0.0, "ctc_loss_reduction": "mean"},
         "train": {"max_samples": 10 ** 7, "max_items": 8}, "eval_max_samples": 10 ** 7}
    sysobj = finetune.XLSR(str(tmp_path / "base"), m, "cpu", TEXTS, tmp_path / "work")
    assert not any(p.requires_grad for p in sysobj.model.wav2vec2.feature_extractor.parameters())   # frozen
    store = fake_store((8000, 6000))
    first, last = _train(sysobj, store, 3000, 3e-3, target_ratio=0.0005)
    assert last < 0.0005 * first
    # batched decoding (the shorter clip is padded) gives the same text as decoding each clip alone
    assert sysobj.decode(store, [0, 1]) == ["dobar dan", "hvala lijepa ivan"]
    assert sysobj.decode(store, [1]) == ["hvala lijepa ivan"]
    sysobj.save(tmp_path / "best")
    sysobj.load(tmp_path / "best")
    assert sysobj.decode(store, [0]) == ["dobar dan"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_finetune_script_end_to_end(whisper_proc, tmp_path):
    from hrasr.audio import encode_opus
    from hrasr.inference import write_wav
    from hrasr.manifest import write_jsonl
    _, proc_dir = whisper_proc
    _tiny_whisper(proc_dir, tmp_path / "tiny")
    data, work = tmp_path / "data", tmp_path / "work"
    (work / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)

    def clips(prefix, n, text):
        rows = []
        for i in range(n):
            sig = (0.1 * rng.standard_normal(16000 + 800 * i)).astype(np.float32)
            write_wav(tmp_path / "x.wav", sig, 16000)
            (work / f"audio/{prefix}{i}.opus").write_bytes(encode_opus((tmp_path / "x.wav").read_bytes()))
            rows.append({"id": f"{prefix}{i}", "audio": f"audio/{prefix}{i}.opus", "text": text, "audio_length": len(sig) / 16000})
        return rows
    for split in ("train", "dev", "test"):
        write_jsonl(data / "clean" / f"{split}.jsonl", clips(split, 2, "Dobar dan."))
    write_jsonl(data / "clean" / "silence_probe.jsonl", clips("si", 1, ""))
    import re
    cfg = re.sub(r"eval_every_steps: \d+", "eval_every_steps: 2", (ROOT / "configs" / "finetune.yaml").read_text(encoding="utf-8"))
    # 2 train clips, batch 1, 2 batches per optimizer step: one step per epoch, so 5 steps = 5 epochs
    cfg = re.sub(r"batch_size: \d+", "batch_size: 1", cfg)
    assert "grad_accum: 2" in cfg
    (tmp_path / "cfg.yaml").write_text(cfg, encoding="utf-8")
    env = {**os.environ, "HRASR_DATA": str(data), "HRASR_WORK": str(work), "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune.py"), "--system", "whisper_small_ft",
                          "--config", str(tmp_path / "cfg.yaml"), "--out-dir", str(tmp_path / "results"),
                          "--base-dir", str(tmp_path / "tiny"), "--max-steps", "5"], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    meta = json.loads((tmp_path / "results" / "predictions" / "whisper_small_ft.meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "ours" and meta["trial"] and meta["chosen_step"] in (2, 4, 5)
    log = json.loads((tmp_path / "results" / "train_logs" / "whisper_small_ft.json").read_text(encoding="utf-8"))
    assert [e["step"] for e in log["evals"]] == [2, 4, 5] and {s["num_beams"] for s in log["dev_search"]} == {1, 5}
    assert log["grad_accum"] == 2 and log["steps_per_epoch"] == 1 and [e["epoch"] for e in log["evals"]] == [2.0, 4.0, 5.0]
    assert "x 2 accumulated" in meta["batch_size"]
    for split, n in (("dev", 2), ("test", 2), ("silence", 1)):
        assert len((tmp_path / "results" / "predictions" / f"whisper_small_ft.{split}.jsonl").read_text(encoding="utf-8").splitlines()) == n

    # decoding only (--from-trained): weights of an earlier run + its Kaggle log; the logged dev WER is checked
    def kaggle_log(wer):
        line = "[whisper_small_ft] " + json.dumps({"step": 5, "dev_wer_greedy": wer, "saved": True}) + chr(10)
        (tmp_path / "train.log").write_text(json.dumps([{"stream_name": "stdout", "time": 1.0, "data": line}]), encoding="utf-8")

    def decode_only():
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune.py"), "--system", "whisper_small_ft",
                               "--config", str(tmp_path / "cfg.yaml"), "--out-dir", str(tmp_path / "results2"),
                               "--base-dir", str(tmp_path / "tiny"), "--from-trained", str(tmp_path / "tiny"),
                               "--trained-log", str(tmp_path / "train.log")], env=env, capture_output=True, text=True)
    kaggle_log(9.9)
    bad = decode_only()
    assert bad.returncode != 0 and "do not give the logged dev WER" in bad.stderr + bad.stdout
    actual = float(re.search(r"\(([0-9.]+) vs 9\.9\)", bad.stderr + bad.stdout).group(1))
    kaggle_log(actual)
    ok = decode_only()
    assert ok.returncode == 0, ok.stderr[-3000:]
    log2 = json.loads((tmp_path / "results2" / "train_logs" / "whisper_small_ft.json").read_text(encoding="utf-8"))
    meta2 = json.loads((tmp_path / "results2" / "predictions" / "whisper_small_ft.meta.json").read_text(encoding="utf-8"))
    assert log2["chosen_step"] == 5 and log2["chosen_dev_wer_greedy_recomputed"] == pytest.approx(actual, abs=0.002)
    assert meta2["trained_in"]["weights"] == str(tmp_path / "tiny") and meta2["chosen_step"] == 5
    for split, n in (("dev", 2), ("test", 2), ("silence", 1)):
        assert len((tmp_path / "results2" / "predictions" / f"whisper_small_ft.{split}.jsonl").read_text(encoding="utf-8").splitlines()) == n


def test_gradient_checkpointing_trains_and_generates_with_cache(whisper_proc, tmp_path):
    """The settings of configs/finetune.yaml: gradient checkpointing on (config.use_cache off for training);
    training still lowers the loss, and generation still uses the key/value cache."""
    _, proc_dir = whisper_proc
    _tiny_whisper(proc_dir, tmp_path / "tiny")
    m = {"language": "hr", "train": {"batch_size": 2, "gradient_checkpointing": True}, "eval_batch_size": 2}
    sysobj = finetune.Whisper(str(tmp_path / "tiny"), m, "cpu")
    assert sysobj.model.is_gradient_checkpointing and sysobj.model.config.use_cache is False
    store = fake_store()
    sysobj.model.train()
    (sysobj.loss(store, [0, 1]) / 2).backward()          # two accumulated batches, as in scripts/finetune.py
    (sysobj.loss(store, [1, 0]) / 2).backward()
    assert [n for n, p in sysobj.model.named_parameters() if p.requires_grad and p.grad is None] == []
    sysobj.model.zero_grad()
    first, last = _train(sysobj, store, 150, 3e-3)
    assert last < 0.05 * first
    calls = []
    orig = sysobj.model.generate
    sysobj.model.generate = lambda *a, **k: (calls.append(k.get("use_cache")), orig(*a, **k))[1]
    assert sysobj.decode(store, [0, 1]) == TEXTS and calls == [True]


def test_xlsr_gradient_checkpointing_flag(tmp_path):
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
    cfg = Wav2Vec2Config(vocab_size=32, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 8), num_feat_extract_layers=2,
                         feat_extract_norm="layer", do_stable_layer_norm=True, num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=4)
    Wav2Vec2ForCTC(cfg).save_pretrained(tmp_path / "base")
    m = {"model": {"layerdrop": 0.0, "mask_time_prob": 0.0}, "train": {"max_samples": 10 ** 7, "max_items": 8,
         "gradient_checkpointing": True}, "eval_max_samples": 10 ** 7}
    sysobj = finetune.XLSR(str(tmp_path / "base"), m, "cpu", TEXTS, tmp_path / "work")
    assert sysobj.model.is_gradient_checkpointing
    loss = sysobj.loss(fake_store((8000, 6000)), [0, 1])
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in sysobj.model.lm_head.parameters())
    # every trainable parameter gets a gradient through the checkpointed layers
    assert [n for n, p in sysobj.model.named_parameters() if p.requires_grad and p.grad is None] == []


def test_evals_from_kaggle_log_reads_only_that_systems_eval_lines(tmp_path):
    lines = ['[whisper_small_ft] whisper_small_ft: 242 M parameters, 4281 steps\n',
             '[xlsr_300m_ft] {"step": 2000, "dev_wer_greedy": 0.15, "saved": true}\n',
             '[whisper_small_ft] {"step": 1500, "dev_wer_greedy": 0.14, "saved": true}\n',
             '[whisper_small_ft] {"num_beams": 1, "dev_wer": 0.12}\n',
             '[whisper_small_ft] {"step": 3000, "dev_wer_greedy": 0.12, "saved": true}\n']
    # Kaggle splits the output in chunks that do not follow line ends
    blob = "".join(lines)
    chunks = [blob[i:i + 37] for i in range(0, len(blob), 37)]
    (tmp_path / "k.log").write_text(json.dumps([{"stream_name": "stdout", "time": 1.0, "data": c} for c in chunks]),
                                    encoding="utf-8")
    ev = finetune.evals_from_kaggle_log(tmp_path / "k.log", "whisper_small_ft")
    assert [e["step"] for e in ev] == [1500, 3000] and ev[1]["dev_wer_greedy"] == 0.12
    with pytest.raises(SystemExit):
        finetune.evals_from_kaggle_log(tmp_path / "k.log", "ctc_scratch")


def test_whisper_eval_batches_shrink_with_the_beam(whisper_proc, tmp_path):
    _, proc_dir = whisper_proc
    _tiny_whisper(proc_dir, tmp_path / "tiny")
    sysobj = finetune.Whisper(str(tmp_path / "tiny"), {"language": "hr", "train": {}, "eval_batch_size": 4}, "cpu")
    store = fake_store((8000, 6000, 7000, 5000, 9000), texts=TEXTS * 3)
    assert [len(b) for b in sysobj.eval_batches(store)] == [4, 1]
    assert [len(b) for b in sysobj.eval_batches(store, {"num_beams": 2})] == [2, 2, 1]
    assert [len(b) for b in sysobj.eval_batches(store, {"num_beams": 5})] == [1] * 5
    assert sorted(k for b in sysobj.eval_batches(store, {"num_beams": 5}) for k in b) == list(range(5))
