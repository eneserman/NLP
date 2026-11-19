import yaml

from hrasr.cleaning import AudioInfo, clip_reasons, word_times
from hrasr.config import ROOT

CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))


def good_row(**kw):
    text = "poštovani zastupnici danas raspravljamo o prijedlogu zakona"  # 51 letters
    row = {
        "speaker_name": "Horvat, Ana", "audio_length": 4.0, "text": text,
        "words": [{"time_s": 200, "time_e": 900}, {"time_s": 1000, "time_e": 3800}],
    }
    row.update(kw)
    return row, text


def test_good_clip_is_kept():
    row, text = good_row()
    assert clip_reasons(row, text, AudioInfo(4.0, -20.0), CFG) == []


def test_missing_speaker():
    row, text = good_row(speaker_name=None)
    assert "missing_speaker" in clip_reasons(row, text, AudioInfo(4.0, -20.0), CFG)


def test_broken_audio():
    row, text = good_row()
    assert "decode_error" in clip_reasons(row, text, None, CFG)
    assert "duration_mismatch" in clip_reasons(row, text, AudioInfo(2.0, -20.0), CFG)
    assert "silence" in clip_reasons(row, text, AudioInfo(4.0, -70.0), CFG)


def test_text_problems():
    row, _ = good_row()
    assert "digits" in clip_reasons(row, "članak 5 zakona o radu i drugi zakoni", AudioInfo(4.0, -20.0), CFG)
    assert "annotation" in clip_reasons(row, "hvala (Upadica: ne) hvala lijepa svima", AudioInfo(4.0, -20.0), CFG)
    assert "empty_text" in clip_reasons(row, "", AudioInfo(4.0, -20.0), CFG)


def test_speaking_rate_outliers():
    row, _ = good_row()
    assert "speaking_rate" in clip_reasons(row, "da", AudioInfo(4.0, -20.0), CFG)
    long = "riječ " * 30
    assert "speaking_rate" in clip_reasons(row, long, AudioInfo(4.0, -20.0), CFG)


def test_alignment_gap_edges_and_inside():
    row, text = good_row(words=[{"time_s": 3000, "time_e": 3500}, {"time_s": 3600, "time_e": 3900}])
    assert "alignment_gap" in clip_reasons(row, text, AudioInfo(4.0, -20.0), CFG)  # 3 s before the first word
    row, text = good_row(audio_length=10.0, words=[{"time_s": 0, "time_e": 1000}, {"time_s": 5000, "time_e": 9900}])
    assert "alignment_gap" in clip_reasons(row, text, AudioInfo(10.0, -20.0), CFG)  # 4 s between words


def test_word_times_units():
    times, rel = word_times([{"time_s": 100, "time_e": 900}], 1.0)
    assert times == [(0.1, 0.9)] and rel
    times, rel = word_times([{"time_s": 0.1, "time_e": 0.9}], 1.0)
    assert times == [(0.1, 0.9)] and rel
    _, rel = word_times([{"time_s": 500000, "time_e": 500900}], 1.0)
    assert not rel  # offsets in the full session video


def test_metadata_only_mode_skips_audio_rules():
    row, text = good_row()
    assert clip_reasons(row, text, None, CFG, check_audio=False) == []
    row, _ = good_row()
    assert clip_reasons(row, "članak 5 zakona o radu i drugi zakoni", None, CFG, check_audio=False) == ["digits"]


def test_session_of():
    from hrasr.cleaning import session_of
    assert session_of("ParlaMint-HR_2021-01-02-0.u12_0-100", "2021-01-02") == "ParlaMint-HR_2021-01-02-0"
    assert session_of("odd-id", "2021-01-02") == "date:2021-01-02"
