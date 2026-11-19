import pytest

from hrasr.metrics import score
from hrasr.subgroups import assign_groups, length_label, subgroup_scores, words_per_second

# 6 clips: 1, 2, 3, 4, 5, 6 words per second (text has 6 words, lengths 6 s ... 1 s)
SIX = "jedan dva tri četiri pet šest"
ROWS = [{"id": f"c{k}", "text": SIX, "audio_length": 6.0 / k, "speaker_gender": g}
        for k, g in zip(range(1, 7), ["F", "M", "F", "M", "F", "M"])]


def test_words_per_second_uses_normalized_words():
    assert words_per_second({"text": "Hvala, gospodine predsjedniče!", "audio_length": 1.5}) == pytest.approx(2.0)


def test_rate_tertiles_and_borders():
    groups, info = assign_groups(ROWS)
    lo, hi = info["speaking_rate_borders_wps"]
    assert lo == pytest.approx(2 + 2 / 3, abs=1e-3) and hi == pytest.approx(4 + 1 / 3, abs=1e-3)
    rate = groups["speaking_rate"]
    assert [rate[f"c{k}"] for k in range(1, 7)] == ["slow", "slow", "medium", "medium", "fast", "fast"]


def test_gender_and_length_groups():
    groups, _ = assign_groups(ROWS + [{"id": "x", "text": "a b", "audio_length": 25.0, "speaker_gender": ""},
                                      {"id": "y", "text": "a b", "audio_length": 3.0, "speaker_gender": "-"}])
    assert groups["gender"]["c1"] == "F" and groups["gender"]["c2"] == "M" and groups["gender"]["x"] == "unknown"
    assert groups["gender"]["y"] == "unknown"            # ParlaSpeech-HR writes "-" for an unknown gender
    assert groups["clip_length"]["c1"] == "5-10 s"      # 6 s
    assert groups["clip_length"]["c6"] == "0-5 s"       # 1 s
    assert groups["clip_length"]["x"] == ">= 20 s"
    assert [length_label(s) for s in (4.99, 5.0, 19.99, 20.0)] == ["0-5 s", "5-10 s", "10-20 s", ">= 20 s"]


def test_subgroup_scores_match_direct_scoring():
    refs = {r["id"]: r["text"] for r in ROWS}
    hyps = {i: t for i, t in refs.items()}
    hyps["c5"] = "jedan dva tri"          # 3 deletions in a fast, female clip
    hyps["c1"] = "jedan dva tri četiri pet šest sedam"   # 1 insertion in a slow, female clip
    groups, _ = assign_groups(ROWS)
    s = subgroup_scores(refs, hyps, groups, iters=50)
    assert list(s["speaking_rate"]) == ["slow", "medium", "fast"]
    assert s["speaking_rate"]["fast"]["wer"] == pytest.approx(3 / 12)
    assert s["speaking_rate"]["medium"]["wer"] == 0.0
    assert s["speaking_rate"]["slow"]["wer"] == pytest.approx(1 / 12)
    female = sorted(i for i, g in groups["gender"].items() if g == "F")
    assert s["gender"]["F"]["wer"] == score([refs[i] for i in female], [hyps[i] for i in female], 50)["wer"]
    assert s["gender"]["F"]["wer"] == pytest.approx(4 / 18)   # c1 and c5 are female clips
    assert s["gender"]["M"]["wer"] == 0.0
    # the groups come from the references only: a different system gets the same clips per group
    assert sum(v["clips"] for v in s["clip_length"].values()) == len(ROWS)
