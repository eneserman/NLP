"""Text rules.

Two different jobs, kept apart on purpose:

* `clean_reference` fixes formatting noise in a transcript and keeps casing and punctuation.
  Its output is the `text` field of the cleaned dataset.
* `normalize_for_scoring` is the single evaluation policy (slide W9-11). Every system's
  reference and hypothesis go through it before WER/CER. It is also the training target
  for our character-level models.
"""
from __future__ import annotations

import re
import unicodedata

_QUOTES = {"„": '"', "“": '"', "”": '"', "«": '"', "»": '"', "‚": "'", "‘": "'", "’": "'", "`": "'"}
_DASHES = {"–": "-", "—": "-", "‒": "-", "−": "-"}
_INVISIBLE = {"​": "", "‌": "", "‍": "", "﻿": "", "­": ""}
_SPACES = {" ": " ", " ": " ", " ": " ", "\t": " ", "\n": " ", "\r": " "}
_MAP = str.maketrans({**_QUOTES, **_DASHES, **_INVISIBLE, **_SPACES})
_WS = re.compile(r"\s+")
_ANNOTATION = re.compile(r"[\[\]\(\)\{\}<>*#_/\\|@]")
_DIGIT = re.compile(r"\d")


def clean_reference(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_MAP)
    return _WS.sub(" ", text).strip()


def normalize_for_scoring(text: str) -> str:
    """NFKC -> lowercase -> every non-letter, non-digit character becomes a space -> collapse spaces.

    Croatian letters č ć đ š ž are kept as they are: they distinguish words
    (e.g. "kuca" and "kuća" are different words), so removing diacritics would hide real errors.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    out = [ch if (ch.isalpha() or ch.isdigit()) else " " for ch in text]
    return _WS.sub(" ", "".join(out)).strip()


def has_digits(text: str) -> bool:
    return bool(_DIGIT.search(text))


def has_annotation(text: str) -> bool:
    """Brackets and similar symbols mark editorial notes in parliamentary transcripts."""
    return bool(_ANNOTATION.search(text))


def has_non_latin_letters(text: str) -> bool:
    """True if a letter is not from the Latin script (e.g. Cyrillic or Greek)."""
    for ch in text:
        if ch.isalpha() and "LATIN" not in unicodedata.name(ch, ""):
            return True
    return False
