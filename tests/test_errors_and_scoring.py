import json
import os
import subprocess
import sys

import pytest

from hrasr.config import ROOT
from hrasr.errors import category_recall, tag_reference
from hrasr.manifest import write_jsonl


def test_tags_names_not_sentence_starts():
    tags = dict(tag_reference("Hvala. Riječ ima Ivan Horvat iz HDZ-a, a on nije tu."))
    assert "name" not in tags["hvala"] and "name" not in tags["riječ"]  # sentence starts
    assert {"name"} <= tags["ivan"] and {"name"} <= tags["horvat"] and {"name"} <= tags["hdz"]
    assert "negation" in tags["nije"]


def test_category_recall():
    refs = ["Danas govori Ivan Horvat, on nije za."]
    hyps = ["danas govori ivan horvath on je za"]
    c = category_recall(refs, hyps)
    assert c["name"]["ref_words"] == 2 and c["name"]["recall"] == 0.5
    assert c["negation"]["recall"] == 0.0
    assert c["most_missed_names"] == [("horvat", 1)]


def test_score_script_end_to_end(tmp_path):
    data = tmp_path / "data"
    (data / "clean").mkdir(parents=True)
    write_jsonl(data / "clean" / "test.jsonl", [
        {"id": "a", "text": "Hvala lijepa, Ivan Horvat."}, {"id": "b", "text": "Zakon nije prihvaćen."}])
    write_jsonl(data / "clean" / "silence_probe.jsonl", [
        {"id": "s1", "text": "", "audio_length": 3.0}, {"id": "s2", "text": "", "audio_length": 4.0}])
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "baselines.yaml").write_text((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    pred = root / "results" / "predictions"
    pred.mkdir(parents=True)
    write_jsonl(pred / "whisper_small.test.jsonl", [{"id": "a", "hyp": "hvala lijepa ivan horvat"}, {"id": "b", "hyp": "zakon je prihvaćen"}])
    write_jsonl(pred / "whisper_small.silence.jsonl", [{"id": "s1", "hyp": ""}, {"id": "s2", "hyp": "hvala"}])
    (pred / "whisper_small.meta.json").write_text(json.dumps({"kind": "baseline", "checkpoint": "openai/whisper-small",
                                                              "revision": "abc", "audio_seconds": 10, "compute_seconds": 2}))
    # run a copy of score.py whose ROOT is the temporary repo
    script = (ROOT / "scripts" / "score.py").read_text(encoding="utf-8")
    script = script.replace('RESULTS = ROOT / "results"', f'ROOT = __import__("pathlib").Path({str(root)!r})\nRESULTS = ROOT / "results"')
    (root / "score.py").write_text(script, encoding="utf-8")
    env = {**os.environ, "HRASR_DATA": str(data), "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, str(root / "score.py")], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    md = (root / "results" / "RESULTS.test.md").read_text(encoding="utf-8")
    s = json.loads((root / "results" / "scores.test.json").read_text(encoding="utf-8"))["whisper_small"]
    assert s["wer"] == pytest.approx(1 / 7)          # "nije" -> "je": one substitution out of 7 words
    assert s["rtfx"] == 5.0
    assert s["silence"] == {"clips": 2, "clips_with_output": 1, "words_emitted": 1, "seconds": 7.0}
    assert "not run: Croatian is not one of the 30 supported languages." in md
    assert "No-speech clips" in md and "Important words" in md
