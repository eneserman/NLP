"""Cleaning rules. Each rule finds a real problem and returns a reason code.

The reason codes and their meaning are listed in README, section "Cleaning".
`scripts/05_clean.py` applies the rules and writes every removed clip to
data/clean/cleaning_log.csv with its reason.
"""
from __future__ import annotations

from dataclasses import dataclass

from hrasr import textnorm

REASONS = {
    "missing_speaker": "No speaker name, so the clip cannot be put in a speaker-disjoint split.",
    "duplicate_id": "The same clip ID appears more than once.",
    "duplicate_audio": "The audio bytes are identical to an earlier clip.",
    "decode_error": "ffmpeg could not decode the audio.",
    "duration_mismatch": "Decoded length differs from the metadata length (broken or cut audio).",
    "duration_out_of_range": "Clip is shorter or longer than the allowed range.",
    "empty_text": "Transcript is empty after cleaning.",
    "annotation": "Transcript contains brackets or symbols that mark editorial notes.",
    "digits": "Transcript contains digits; the spoken form of a number is ambiguous.",
    "non_latin_script": "Transcript contains letters from another script.",
    "speaking_rate": "Characters per second are outside the normal range (likely misalignment).",
    "no_word_alignment": "The word-level alignment is missing.",
    "alignment_gap": "Long untranscribed gap at the start, end, or between words (likely missing words).",
    "silence": "Audio is almost silent.",
}


def session_of(clip_id: str, date: str = "") -> str:
    """Recording session of a clip. IDs look like ParlaMint-HR_2021-01-02-0.u12_<offsets>;
    the part before ".u" names the sitting. If the ID has another form, the date is used."""
    if ".u" in clip_id:
        return clip_id.split(".u", 1)[0]
    return f"date:{date}"


FEATURE_COLUMNS = ("wa_n", "wa_lead", "wa_trail", "wa_inner", "wa_relative")


def word_features(words, duration_s: float) -> dict | None:
    """Short summary of the word alignment: number of words, gap before the first word, gap after
    the last word, longest gap between words, and whether times are relative to the clip."""
    wt = word_times(words, duration_s)
    if wt is None:
        return None
    times, relative = wt
    inner = max((times[i + 1][0] - times[i][1] for i in range(len(times) - 1)), default=0.0)
    return {"n": len(times), "lead": times[0][0], "trail": duration_s - times[-1][1],
            "inner": inner, "relative": relative}


def alignment_features(row: dict, duration_s: float) -> dict | None:
    """From the `words` list if the row has it, else from the wa_* columns made in step 1."""
    if "words" in row:
        return word_features(row["words"], duration_s)
    if row.get("wa_n") is None or row.get("wa_n") != row.get("wa_n") or row.get("wa_n") == 0:
        return None
    return {"n": row["wa_n"], "lead": row["wa_lead"], "trail": row["wa_trail"],
            "inner": row["wa_inner"], "relative": bool(row["wa_relative"])}


@dataclass
class AudioInfo:
    duration_s: float
    rms_dbfs: float


def word_times(words, duration_s: float) -> tuple[list[tuple[float, float]], bool] | None:
    """Word (start, end) times in seconds, and whether they are relative to the clip start.

    The corpus documents millisecond offsets; seconds are accepted too. If the times are
    larger than the clip, they are offsets in the full session video, so only the gaps
    between words can be checked.
    """
    if words is None or len(words) == 0:  # Parquet gives a numpy array, JSON a list
        return None
    times = [(float(w["time_s"]), float(w["time_e"])) for w in words]
    if max(e for _, e in times) > duration_s * 20 + 1:
        times = [(s / 1000.0, e / 1000.0) for s, e in times]
    relative = max(e for _, e in times) <= duration_s + 1.0
    return times, relative


def text_reasons(text: str, cfg: dict) -> list[str]:
    reasons = []
    if not text:
        return ["empty_text"]
    if textnorm.has_annotation(text):
        reasons.append("annotation")
    if textnorm.has_digits(text):
        reasons.append("digits")
    if textnorm.has_non_latin_letters(text):
        reasons.append("non_latin_script")
    if not textnorm.normalize_for_scoring(text):
        reasons.append("empty_text")
    return reasons


def clip_reasons(row: dict, text: str, audio: AudioInfo | None, cfg: dict, check_audio: bool = True) -> list[str]:
    """All problems of one clip. An empty list means the clip is kept.

    With check_audio=False only the metadata and the transcript are checked. Selection uses this
    to predict how much audio will survive cleaning; the real decision is made in step 5.
    """
    c = cfg["cleaning"]
    reasons: list[str] = []
    if not row.get("speaker_name"):
        reasons.append("missing_speaker")

    meta_dur = float(row["audio_length"])
    if not check_audio:
        dur = meta_dur
    elif audio is None:
        reasons.append("decode_error")
        dur = meta_dur
    else:
        dur = audio.duration_s
        if abs(audio.duration_s - meta_dur) > c["max_duration_mismatch_s"]:
            reasons.append("duration_mismatch")
        if audio.rms_dbfs < c["min_rms_dbfs"]:
            reasons.append("silence")

    if not (c["min_duration_s"] <= dur <= c["max_duration_s"]):
        reasons.append("duration_out_of_range")

    reasons += text_reasons(text, cfg)

    n_chars = len(textnorm.normalize_for_scoring(text).replace(" ", ""))
    if dur > 0 and n_chars:
        rate = n_chars / dur
        if not (c["min_chars_per_s"] <= rate <= c["max_chars_per_s"]):
            reasons.append("speaking_rate")

    feats = alignment_features(row, meta_dur)
    if feats is None:
        reasons.append("no_word_alignment")
    else:
        gap = feats["inner"] > c["max_inner_gap_s"]
        if feats["relative"]:
            gap = gap or feats["lead"] > c["max_edge_silence_s"] or feats["trail"] > c["max_edge_silence_s"]
        if gap:
            reasons.append("alignment_gap")

    # Keep order stable and unique.
    return list(dict.fromkeys(reasons))
