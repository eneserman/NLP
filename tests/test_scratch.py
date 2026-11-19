import itertools
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hrasr.config import ROOT  # noqa: E402
from hrasr.conformer import ConformerEncoder  # noqa: E402
from hrasr.features import cmvn, deltas, log_mel, mel_filterbank, mfcc, num_frames, spec_augment  # noqa: E402
from hrasr.scratch_data import FeatureStore, collate, frame_batches  # noqa: E402
from hrasr.scratch_models import AEDModel, CTCModel, RNNTModel, ctc_greedy  # noqa: E402
from hrasr.transducer import greedy_decode, rnnt_loss  # noqa: E402
from hrasr.vocab import CharVocab  # noqa: E402

TINY = {"n_mels": 80, "d_model": 32, "n_layers": 1, "heads": 2, "ff": 64, "kernel": 3, "dropout": 0.0}


# ---------------------------------------------------------------- features -------------------------
def test_log_mel_and_mfcc_shapes():
    w = torch.randn(16000)
    assert log_mel(w).shape == (num_frames(16000), 80)
    assert mfcc(w).shape == (num_frames(16000), 39)


def test_mel_filterbank_peaks_increase():
    fb = mel_filterbank(40)
    assert fb.shape == (40, 201) and float(fb.min()) >= 0
    peaks = fb.argmax(1)
    assert bool((peaks[1:] >= peaks[:-1]).all())


def test_deltas_of_a_ramp_are_its_slope():
    x = torch.arange(20, dtype=torch.float32)[:, None] * 3.0
    d = deltas(x)
    assert torch.allclose(d[2:-2], torch.full((16, 1), 3.0))


def test_cmvn():
    x = cmvn(torch.randn(500, 4) * 5 + 2)
    assert torch.allclose(x.mean(0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(x.std(0), torch.ones(4), atol=1e-3)


def test_spec_augment_masks_only_inside_the_utterance():
    torch.manual_seed(0)
    x = torch.ones(3, 100, 80)
    lengths = torch.tensor([100, 60, 30])
    y = spec_augment(x, lengths, n_freq=2, f=10, n_time=3, t_frac=0.1)
    assert y.shape == x.shape and float(y.min()) == 0.0
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    for b, L in enumerate(lengths.tolist()):
        full_time_masked = (y[b].sum(1) == 0).nonzero().flatten()
        assert all(t < L for t in full_time_masked.tolist())


# ---------------------------------------------------------------- vocab / data ----------------------
def test_vocab_roundtrip_and_unknown():
    v = CharVocab.from_texts(["dobar dan", "čaša"])
    assert v.itos[:2] == ["<blank>", "<unk>"] and " " in v.chars
    assert v.decode(v.encode("dan čaša")) == "dan čaša"
    assert v.decode(v.encode("dan x")) == "dan"            # x is unknown -> dropped
    assert CharVocab.from_dict(v.to_dict()).itos == v.itos
    assert v.sos == len(v) and v.eos == len(v) + 1


def test_frame_batches_respect_the_budget():
    lengths = [100, 300, 50, 300, 200, 10]
    for rng in (None, np.random.default_rng(0)):
        b = frame_batches(lengths, max_frames=600, max_items=3, rng=rng)
        assert sorted(i for x in b for i in x) == list(range(6))
        for x in b:
            assert max(lengths[i] for i in x) * len(x) <= 600 and len(x) <= 3


def test_collate_pads_and_keeps_lengths():
    store = FeatureStore(np.arange(14, dtype=np.float16).reshape(7, 2), np.array([0, 3, 7]), ["a", "b"])
    x, xl, y, yl = collate(store, [1, 0], [[2, 3], [4]])
    assert x.shape == (2, 4, 2) and xl.tolist() == [4, 3] and yl.tolist() == [1, 2]
    assert float(x[1, 3].abs().sum()) == 0.0


# ---------------------------------------------------------------- RNN-T loss -------------------------
def brute_force_rnnt(lp, y, T, U):
    """Sum over all alignments: T blanks and U labels, each path ends with a blank in frame T-1."""
    total = []
    for emit_frames in itertools.combinations_with_replacement(range(T), U):
        s, u = 0.0, 0
        frames = list(emit_frames)
        for t in range(T):
            while u < U and frames[u] == t:
                s += lp[t, u, y[u]]
                u += 1
            s += lp[t, u, 0]
        total.append(s)
    return -float(torch.logsumexp(torch.tensor(total), 0))


def test_rnnt_loss_matches_sum_over_alignments():
    torch.manual_seed(0)
    B, T, U, V = 3, 4, 3, 5
    lp = torch.randn(B, T, U + 1, V, dtype=torch.float64).log_softmax(-1)
    y = torch.randint(1, V, (B, U))
    t_len, u_len = torch.tensor([4, 3, 2]), torch.tensor([3, 1, 2])
    got = rnnt_loss(lp, y, t_len, u_len, reduction="none")
    for b in range(B):
        exp = brute_force_rnnt(lp[b], y[b].tolist(), int(t_len[b]), int(u_len[b]))
        assert float(got[b]) == pytest.approx(exp, rel=1e-5)


def test_rnnt_loss_gradient():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 3, 4, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1, 2], [3, 1]])

    def f(z):
        return rnnt_loss(z.log_softmax(-1), y, torch.tensor([3, 2]), torch.tensor([2, 1])).to(torch.float64)
    # rnnt_loss computes in float32; compare autograd with finite differences loosely
    loss = f(logits)
    loss.backward()
    eps = 1e-3
    for idx in [(0, 0, 0, 1), (0, 2, 2, 0), (1, 1, 1, 3)]:
        z = logits.detach().clone()
        z[idx] += eps
        up = float(f(z))
        z[idx] -= 2 * eps
        down = float(f(z))
        assert float(logits.grad[idx]) == pytest.approx((up - down) / (2 * eps), abs=2e-3)


# ---------------------------------------------------------------- decoding helpers ----------------------
def test_ctc_greedy_merges_repeats_and_drops_blanks():
    seq = torch.tensor([[0, 3, 3, 0, 3, 4, 4, 0, 0]])
    lp = torch.nn.functional.one_hot(seq, 6).float().log()
    assert ctc_greedy(lp, torch.tensor([9])) == [[3, 3, 4]]
    assert ctc_greedy(lp, torch.tensor([3])) == [[3]]


class CountingPredictor:
    """State = number of symbols emitted so far."""
    def initial(self, B, device):
        n = torch.zeros(1, B, 1)
        return n[0], (n, n.clone())

    def step(self, tokens, state):
        n = state[0] + 1
        return n[0], (n, n.clone())


def test_greedy_transducer_emits_until_blank_wins():
    target = [3, 1, 2]

    def joint(enc_t, pred_out):          # emits target[n] while n < len(target), blank afterwards
        n = pred_out[:, 0].long()
        logits = torch.full((enc_t.shape[0], 5), -10.0)
        for b in range(enc_t.shape[0]):
            k = int(n[b])
            logits[b, target[k] if k < len(target) else 0] = 10.0
        return logits

    enc = torch.zeros(2, 4, 1)
    hyps = greedy_decode(enc, torch.tensor([4, 1]), CountingPredictor(), joint, max_symbols=2)
    assert hyps[0] == target            # 2 symbols in frame 0, 1 in frame 1
    assert hyps[1] == [3, 1]            # only one frame: at most max_symbols


# ---------------------------------------------------------------- encoder -------------------------
def test_encoder_output_does_not_depend_on_padding():
    torch.manual_seed(0)
    for stride in (2, 4):
        enc = ConformerEncoder(**TINY, time_stride=stride).eval()
        a, b = torch.randn(1, 37, 80), torch.randn(1, 64, 80)
        alone, la, _ = enc(a, torch.tensor([37]))
        batch = torch.zeros(2, 64, 80)
        batch[0, :37], batch[1] = a[0], b[0]
        both, lb, _ = enc(batch, torch.tensor([37, 64]))
        n = int(la[0])
        assert int(lb[0]) == n and n == math.ceil(37 / stride)
        assert torch.allclose(alone[0, :n], both[0, :n], atol=1e-5)


# ---------------------------------------------------------------- tiny models learn -------------------
def _toy():
    torch.manual_seed(0)
    x = torch.randn(2, 60, 80)
    lengths = torch.tensor([60, 44])
    y = torch.tensor([[3, 4, 5, 3], [5, 3, 0, 0]])
    yl = torch.tensor([4, 2])
    return x, lengths, y, yl


def _fit(model, steps=300, lr=3e-3):
    x, xl, y, yl = _toy()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = None
    for _ in range(steps):
        model.train()
        loss = model.loss(x, xl, y, yl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    model.eval()
    return first, float(loss.detach()), x, xl


@pytest.mark.parametrize("cls,stride", [(CTCModel, 2), (RNNTModel, 4)])
def test_ctc_and_transducer_memorize_two_utterances(cls, stride):
    torch.manual_seed(0)
    kw = {} if cls is CTCModel else {"pred_embed": 16, "pred_hidden": 32, "joint_hidden": 32, "dropout": 0.0}
    model = cls(6, {**TINY, "time_stride": stride}, **kw)
    first, last, x, xl = _fit(model)
    assert last < 0.1 * first
    assert model.decode(x, xl) == [[3, 4, 5, 3], [5, 3]]


def test_aed_memorizes_and_beam_search_and_ctc_rescoring():
    torch.manual_seed(0)
    model = AEDModel(6, {**TINY, "time_stride": 2}, dec_layers=1, dec_ff=64, dec_heads=2, dropout=0.0,
                     ctc_weight=0.3, label_smoothing=0.0)
    first, last, x, xl = _fit(model, steps=300)
    assert last < 0.1 * first
    assert model.decode(x, xl, beam=1) == [[3, 4, 5, 3], [5, 3]]
    nb = model.nbest(x, xl, beam=3)
    assert all(1 <= len(n) for n in nb)
    for n in nb:
        for seq, att, ctc in n:
            assert all(2 <= t < 6 for t in seq) and att <= 0 and ctc <= 0
    assert model.decode(x, xl, beam=3, ctc_lambda=0.3) == [[3, 4, 5, 3], [5, 3]]
    # length-normalized: -1.0 / 2 = -0.5 < -1.2 / 3 = -0.4
    assert AEDModel.choose([([3], -1.0, -50.0), ([4, 4], -1.2, -1.0)], 0.0) == [4, 4]
    assert AEDModel.choose([([3], -1.0, -50.0), ([4, 4], -1.2, -1.0)], 0.5) == [4, 4]
    assert AEDModel.choose([([3], -0.5, -50.0), ([4, 4], -1.8, -1.0)], 0.0) == [3]       # -0.25 > -0.6
    assert AEDModel.choose([([3], -0.5, -50.0), ([4, 4], -1.8, -1.0)], 0.5) == [4, 4]    # CTC changes the choice
    assert AEDModel.choose([([3], -0.1, -0.1), ([4, 4], -3.0, -3.0)], 0.5) == [3]


# ---------------------------------------------------------------- the training script, end to end ---------
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("system", ["ctc_scratch", "aed_hybrid_scratch", "rnnt_scratch"])
def test_train_scratch_script_end_to_end(tmp_path, system):
    from hrasr.audio import encode_opus
    from hrasr.inference import write_wav
    from hrasr.manifest import write_jsonl
    data, work = tmp_path / "data", tmp_path / "work"
    (work / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)
    words = ["dobar dan", "hvala lijepa", "zakon je prihvaćen", "nije tako"]

    def clips(prefix, n, texts):
        rows = []
        for i in range(n):
            sig = (0.1 * rng.standard_normal(16000 + 800 * i)).astype(np.float32)
            write_wav(tmp_path / "x.wav", sig, 16000)
            name = f"audio/{prefix}{i}.opus"
            (work / name).write_bytes(encode_opus((tmp_path / "x.wav").read_bytes()))
            rows.append({"id": f"{prefix}{i}", "audio": name, "text": texts[i % len(texts)].capitalize() + ".",
                         "audio_length": len(sig) / 16000, "speaker_gender": "F"})
        return rows
    write_jsonl(data / "clean" / "train.jsonl", clips("tr", 6, words))
    write_jsonl(data / "clean" / "dev.jsonl", clips("dv", 3, words))
    write_jsonl(data / "clean" / "test.jsonl", clips("te", 3, words))
    write_jsonl(data / "clean" / "silence_probe.jsonl", [{**r, "text": ""} for r in clips("si", 2, words)])
    cfg = (ROOT / "configs" / "train_scratch.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("d_model: 256", "d_model: 32").replace("n_layers: 12", "n_layers: 1").replace("ff: 1024", "ff: 64")
    cfg = cfg.replace("dec_layers: 6", "dec_layers: 1").replace("dec_ff: 1024", "dec_ff: 64")
    cfg = cfg.replace("pred_hidden: 320", "pred_hidden: 32").replace("joint_hidden: 320", "joint_hidden: 32")
    (tmp_path / "cfg.yaml").write_text(cfg, encoding="utf-8")
    env = {**os.environ, "HRASR_DATA": str(data), "HRASR_WORK": str(work), "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_scratch.py"), "--system", system,
                          "--config", str(tmp_path / "cfg.yaml"), "--out-dir", str(tmp_path / "results"),
                          "--epochs", "2", "--workers", "1"], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    pred = tmp_path / "results" / "predictions"
    for split, n in (("dev", 3), ("test", 3), ("silence", 2)):
        rows = [json.loads(line) for line in (pred / f"{system}.{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == n and all(isinstance(r["hyp"], str) for r in rows)
    meta = json.loads((pred / f"{system}.meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "baseline" and meta["trained_by_us"] and meta["trial"] and meta["chosen_epoch"] in (1, 2)
    assert meta["audio_seconds"] > 0 and "setup" in meta
    log = json.loads((tmp_path / "results" / "train_logs" / f"{system}.json").read_text(encoding="utf-8"))
    assert log["evals"][-1]["epoch"] == 2 and "dev_wer_greedy" in log["evals"][-1]
    if system == "aed_hybrid_scratch":
        assert {s["ctc_lambda"] for s in log["dev_search"]} == {0.0, 0.3, 0.5}
    # trial options may not write into the real results folder
    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_scratch.py"), "--system", system, "--epochs", "1"],
                         env=env, capture_output=True, text=True)
    assert bad.returncode != 0
