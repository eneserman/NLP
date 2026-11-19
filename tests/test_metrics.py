import pytest

from hrasr.metrics import score


def test_course_example_slide_33_34():
    # W9 slides 33-34: expected WER 0.3333 and CER 0.2273
    s = score(["the cat sat on the mat"], ["the cat sit on the"])
    assert (s["substitutions"], s["deletions"], s["insertions"]) == (1, 1, 0)
    assert s["wer"] == pytest.approx(0.3333, abs=1e-4)
    assert s["cer"] == pytest.approx(0.2273, abs=1e-4)


def test_corpus_wer_weights_words_not_clips():
    # clip 1: 1 error / 1 word; clip 2: 0 errors / 9 words -> corpus WER 0.1 (mean clip WER would be 0.5)
    s = score(["a", "b c d e f g h i j"], ["x", "b c d e f g h i j"])
    assert s["wer"] == pytest.approx(0.1)


def test_normalization_applied_to_both_sides():
    s = score(["Hvala lijepa, Predsjedniče!"], ["hvala lijepa predsjedniče"])
    assert s["wer"] == 0 and s["cer"] == 0


def test_empty_hypothesis_counts_deletions():
    s = score(["dobar dan"], [""])
    assert s["wer"] == 1.0 and s["deletions"] == 2


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        score([" , "], ["nešto"])


def test_ci_contains_point_estimate():
    refs = [f"riječ {i} jedan dva" for i in range(50)]
    hyps = [r if i % 3 else "pogrešno" for i, r in enumerate(refs)]
    s = score(refs, hyps)
    assert s["wer_ci95"][0] <= s["wer"] <= s["wer_ci95"][1]
