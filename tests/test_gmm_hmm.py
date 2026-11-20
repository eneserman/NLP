import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from hrasr.config import ROOT, load_config
from hrasr.gmm_hmm import N_STATES, SIL, GMMHMM, CharBigramLM

GRID = load_config(ROOT / "configs" / "train_gmm_hmm.yaml")["dev_search"]
N_SETTINGS = len(GRID["lm_weight"]) * len(GRID["ins_penalty"])

CHARS = ["a", "b", "c", " "]


def make_true_model(dim=4, seed=0):
    """Every state has its own well-separated mean."""
    m = GMMHMM(CHARS, dim=dim)
    rng = np.random.default_rng(seed)
    m.means = rng.normal(0, 4, size=(m.S, 1, dim))
    m.vars = np.full((m.S, 1, dim), 0.3)
    return m


def sample(m, text, rng, frames_per_state=3, pause=True):
    """Frames for `text`: each state of each unit emits a few frames; silence at the edges and between words."""
    seq = [SIL] if pause else []
    for i, w in enumerate(text.split()):
        if i and pause:
            seq.append(SIL)
        seq += list(w)
    if pause:
        seq.append(SIL)
    X, path = [], []
    for u in seq:
        for s in range(N_STATES):
            st = m.uidx[u] * N_STATES + s
            for _ in range(frames_per_state):
                X.append(rng.normal(m.means[st, 0], np.sqrt(m.vars[st, 0])))
                path.append(st)
    return np.array(X), np.array(path)


def test_frame_loglik_matches_direct_formula():
    m = make_true_model()
    m.split()
    m.logw = np.log(np.tile([[0.3, 0.7]], (m.S, 1)))
    x = np.random.default_rng(1).normal(size=(5, 4))
    got = m.frame_loglik(x)
    s, t = 7, 2
    comps = [math.log(w) - 0.5 * np.sum(np.log(2 * np.pi * v) + (x[t] - mu) ** 2 / v)
             for w, mu, v in zip(np.exp(m.logw[s]), m.means[s], m.vars[s])]
    assert got[t, s] == pytest.approx(np.logaddexp(*comps))


def test_bigram_lm_is_normalized():
    lm = CharBigramLM(["ab a", "cab"], ["a", "b", "c", " "])
    for ctx in lm.ctx:
        assert sum(math.exp(lm(ctx, t)) for t in lm.tokens) == pytest.approx(1.0)
    assert lm("<s>", "a") > lm("<s>", "b")      # "a" starts a sentence once, "b" never


def test_alignment_recovers_the_path_and_skips_missing_pauses():
    m = make_true_model()
    rng = np.random.default_rng(2)
    for pause in (True, False):
        X, path = sample(m, "ab ca", rng, pause=pause)
        units, optional = m.utterance_units("ab ca")
        got = m.align(m.frame_loglik(X), units, optional)
        assert np.array_equal(got, path)
    assert m.align(m.frame_loglik(X[:5]), units, optional) is None    # too few frames for the required states


def test_flat_start_covers_required_states_in_order():
    m = make_true_model()
    units, optional = m.utterance_units("ab")
    def runs(p):
        return [s for i, s in enumerate(p.tolist()) if i == 0 or s != p[i - 1]]
    p = m.flat_start_alignment(12, units, optional)
    sil, a, b = m.uidx[SIL], m.uidx["a"], m.uidx["b"]
    req = [u * N_STATES + s for u in (sil, a, b, sil) for s in range(N_STATES)]     # edge silences + characters
    assert runs(p) == req and len(p) == 12
    units2, optional2 = m.utterance_units("a b")                                     # no silence between words
    assert runs(m.flat_start_alignment(12, units2, optional2)) == req
    assert m.flat_start_alignment(3, units, optional) is None


def test_split_doubles_gaussians():
    m = make_true_model()
    m.split()
    assert m.means.shape[1] == 2 and np.allclose(np.exp(m.logw).sum(1), 1.0)


def test_training_from_flat_start_learns_to_decode():
    true = make_true_model()
    rng = np.random.default_rng(3)
    texts = ["ab ca", "ba", "cab ab", "a b c", "cc ab", "bac", "ca ba", "abc"] * 3
    pauses = [bool(rng.integers(0, 2)) or i % 3 == 0 for i in range(len(texts))]
    data = [sample(true, t, rng, frames_per_state=int(rng.integers(2, 5)), pause=p) for t, p in zip(texts, pauses)]
    feats = [x for x, _ in data]
    m = GMMHMM(CHARS, dim=4)
    first = True
    for gaussians, iters in ((1, 6), (2, 2)):
        while m.means.shape[1] < gaussians:
            m.split()
        for _ in range(iters):
            paths = []
            for X, t in zip(feats, texts):
                units, optional = m.utterance_units(t)
                paths.append(m.flat_start_alignment(len(X), units, optional) if first
                             else m.align(m.frame_loglik(X), units, optional))
            keep = [i for i, p in enumerate(paths) if p is not None]
            m.reestimate([feats[i] for i in keep], [paths[i] for i in keep])
            first = False
    # after training, the alignment of every utterance equals the true state sequence
    for (X, truth), t in zip(data, texts):
        units, optional = m.utterance_units(t)
        assert np.array_equal(m.align(m.frame_loglik(X), units, optional), truth)
    lm = CharBigramLM(texts, ["a", "b", "c", " "])
    hyps = [m.decode(X, lm, lm_weight=1.0, ins_penalty=0.0) for X in feats]
    # with a pause the word boundary is heard: exact text; without one only the LM can place it
    assert all(h == t for h, t, p in zip(hyps, texts, pauses) if p)
    assert all(h.replace(" ", "") == t.replace(" ", "") for h, t in zip(hyps, texts))


def test_state_dict_roundtrip():
    m = make_true_model()
    m2 = GMMHMM.from_state_dict(m.state_dict())
    assert m2.units == m.units and np.array_equal(m2.means, m.means)
    x = np.random.default_rng(0).normal(size=(3, 4))
    assert np.allclose(m2.frame_loglik(x), m.frame_loglik(x))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_train_gmm_hmm_script_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from hrasr.audio import encode_opus
    from hrasr.inference import write_wav
    from hrasr.manifest import write_jsonl
    data, work = tmp_path / "data", tmp_path / "work"
    (work / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)

    def clips(prefix, n):
        rows = []
        for i in range(n):
            sig = (0.1 * rng.standard_normal(16000 + 1600 * i)).astype(np.float32)
            write_wav(tmp_path / "x.wav", sig, 16000)
            (work / f"audio/{prefix}{i}.opus").write_bytes(encode_opus((tmp_path / "x.wav").read_bytes()))
            rows.append({"id": f"{prefix}{i}", "audio": f"audio/{prefix}{i}.opus", "text": "Dobar dan.",
                         "audio_length": len(sig) / 16000})
        return rows
    for split, n in (("train", 4), ("dev", 2), ("test", 2)):
        write_jsonl(data / "clean" / f"{split}.jsonl", clips(split, n))
    write_jsonl(data / "clean" / "silence_probe.jsonl", [{**r, "text": ""} for r in clips("si", 1)])
    cfg = (ROOT / "configs" / "train_gmm_hmm.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("iterations: 5", "iterations: 2").replace("iterations: 3", "iterations: 1")
    (tmp_path / "cfg.yaml").write_text(cfg, encoding="utf-8")
    env = {**os.environ, "HRASR_DATA": str(data), "HRASR_WORK": str(work), "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_gmm_hmm.py"), "--config", str(tmp_path / "cfg.yaml"),
                          "--out-dir", str(tmp_path / "results"), "--workers", "2", "--limit-train", "4"],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    pred = tmp_path / "results" / "predictions"
    assert len((pred / "gmm_hmm.test.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    meta = json.loads((pred / "gmm_hmm.meta.json").read_text(encoding="utf-8"))
    assert meta["trained_by_us"] and meta["decoding"]["lm_weight"] in GRID["lm_weight"]
    log = json.loads((tmp_path / "results" / "train_logs" / "gmm_hmm.json").read_text(encoding="utf-8"))
    assert len(log["dev_search"]) == N_SETTINGS and log["iterations"][-1]["gaussians"] == 16


def test_reads_an_earlier_run_from_its_kaggle_log(tmp_path):
    pytest.importorskip("torch")          # scripts/train_gmm_hmm.py imports the feature code (torch)
    import importlib
    sys.path.insert(0, str(ROOT / "scripts"))
    t = importlib.import_module("train_gmm_hmm")
    lines = ('{"gaussians": 1, "iteration": 1, "flat_start": true, "aligned": 4}\n'
             '{"gaussians": 1, "iteration": 1, "flat_start": true, "aligned": 4}\n'   # Kaggle writes lines twice
             '{"lm_weight": 5.0, "ins_penalty": 0.0, "dev_wer": 0.9}\n'
             '{"lm_weight": 5.0, "ins_penalty": 0.0, "dev_wer": 0.9}\n'
             'some other output\n'
             '{"chosen_decoding": {"lm_weight": 5.0, "ins_penalty": 0.0}, "timing": {}}\n')
    chunks = [lines[i:i + 23] for i in range(0, len(lines), 23)]         # Kaggle cuts lines anywhere
    (tmp_path / "k.log").write_text(json.dumps([{"stream_name": "stdout", "time": 1, "data": c} for c in chunks]), encoding="utf-8")
    run = t.from_kaggle_log(tmp_path / "k.log")
    assert len(run["iterations"]) == 1 and len(run["dev_search"]) == 1
    assert run["iterations"][0]["aligned"] == 4 and run["dev_search"][0]["dev_wer"] == 0.9
    assert run["chosen_decoding"] == {"lm_weight": 5.0, "ins_penalty": 0.0}
    (tmp_path / "bad.log").write_text(json.dumps([{"stream_name": "stdout", "time": 1, "data": "nothing\n"}]), encoding="utf-8")
    with pytest.raises(SystemExit):
        t.from_kaggle_log(tmp_path / "bad.log")
    assert t.corpus_wer_cer(["ab cd"], ["ab cx"]) == pytest.approx((0.5, 0.2))


def test_train_then_decode_only_without_ffmpeg(tmp_path):
    """scripts/train_gmm_hmm.py end to end on prepared MFCC (the feature cache is filled first, so no
    audio decoding is needed): training, choice by dev CER, then --from-model with the run's log."""
    pytest.importorskip("torch")
    import re
    from hrasr.manifest import write_jsonl
    from hrasr.scratch_data import FeatureStore, utterance_features
    data, work = tmp_path / "data", tmp_path / "work"
    rng = np.random.default_rng(0)
    rows = {}
    for split, n in (("train", 4), ("dev", 2), ("test", 2), ("silence", 1)):
        rows[split] = [{"id": f"{split}{i}", "audio": f"audio/{split}{i}.opus", "text": "" if split == "silence" else "Dobar dan.",
                        "audio_length": 1.0 + 0.1 * i} for i in range(n)]
        name = "silence_probe" if split == "silence" else split
        write_jsonl(data / "clean" / f"{name}.jsonl", rows[split])
        feats = [utterance_features((0.1 * rng.standard_normal(16000 + 1600 * i)).astype(np.float32), "mfcc") for i in range(n)]
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(f) for f in feats])
        FeatureStore(np.concatenate(feats), offsets, [r["id"] for r in rows[split]]).save(work / "feats" / "mfcc_trial4_None" / split)
    cfg = (ROOT / "configs" / "train_gmm_hmm.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("iterations: 5", "iterations: 2").replace("iterations: 3", "iterations: 1")
    (tmp_path / "cfg.yaml").write_text(cfg, encoding="utf-8")
    env = {**os.environ, "HRASR_DATA": str(data), "HRASR_WORK": str(work), "PYTHONPATH": str(ROOT)}
    script = [sys.executable, str(ROOT / "scripts" / "train_gmm_hmm.py"), "--config", str(tmp_path / "cfg.yaml"),
              "--workers", "1", "--limit-train", "4"]
    out = subprocess.run(script + ["--out-dir", str(tmp_path / "r1")], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    log1 = json.loads((tmp_path / "r1" / "train_logs" / "gmm_hmm.json").read_text(encoding="utf-8"))
    assert log1["dev_select_by"] == "cer" and len(log1["dev_search"]) == N_SETTINGS
    best = min(log1["dev_search"], key=lambda s: (s["dev_cer"], s["lm_weight"], s["ins_penalty"]))
    assert log1["chosen_decoding"] == {"lm_weight": best["lm_weight"], "ins_penalty": best["ins_penalty"]}
    # the run's printed output, as Kaggle stores it, is the training log of a decoding-only run
    (tmp_path / "run1.log").write_text(json.dumps([{"stream_name": "stdout", "time": 1, "data": out.stdout}]), encoding="utf-8")
    model = work / "models" / "gmm_hmm_trial" / "model.npz"
    out2 = subprocess.run(script + ["--out-dir", str(tmp_path / "r2"), "--from-model", str(model),
                                    "--trained-log", str(tmp_path / "run1.log")], env=env, capture_output=True, text=True)
    assert out2.returncode == 0, out2.stderr[-3000:]
    log2 = json.loads((tmp_path / "r2" / "train_logs" / "gmm_hmm.json").read_text(encoding="utf-8"))
    assert log2["iterations"] == log1["iterations"] and log2["chosen_decoding"] == log1["chosen_decoding"]
    assert [s["dev_wer"] for s in log2["dev_search"]] == pytest.approx([s["dev_wer"] for s in log1["dev_search"]])
    for split, n in (("dev", 2), ("test", 2), ("silence", 1)):
        a = (tmp_path / "r1" / "predictions" / f"gmm_hmm.{split}.jsonl").read_text(encoding="utf-8")
        b = (tmp_path / "r2" / "predictions" / f"gmm_hmm.{split}.jsonl").read_text(encoding="utf-8")
        assert len(b.splitlines()) == n and a == b                     # same model, same predictions
    meta2 = json.loads((tmp_path / "r2" / "predictions" / "gmm_hmm.meta.json").read_text(encoding="utf-8"))
    assert meta2["decoding_chosen_by"] == "dev CER" and meta2["trained_in"] == str(model)
    # a model that does not match the log is refused
    tampered = json.loads((tmp_path / "run1.log").read_text(encoding="utf-8"))
    tampered[0]["data"] = re.sub(r'"dev_wer": [0-9.]+', '"dev_wer": 0.123456', tampered[0]["data"], count=1)
    (tmp_path / "bad.log").write_text(json.dumps(tampered), encoding="utf-8")
    bad = subprocess.run(script + ["--out-dir", str(tmp_path / "r3"), "--from-model", str(model),
                                   "--trained-log", str(tmp_path / "bad.log")], env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "does not give the logged dev WER" in bad.stderr + bad.stdout
