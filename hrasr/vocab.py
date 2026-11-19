"""Character vocabulary for the models we train ourselves.

Targets are the scoring-normalized text (`normalize_for_scoring`): lowercase letters (with
č ć đ š ž), digits never occur (cleaning removes them), and single spaces between words.
Croatian spelling is close to phonemic, so characters are a reasonable modelling unit.

Index 0 is the CTC / transducer blank and the padding symbol. For the attention decoder, the
last two indices are <sos> and <eos>.
"""
from __future__ import annotations

from collections import Counter

BLANK = "<blank>"
UNK = "<unk>"


class CharVocab:
    def __init__(self, chars: list[str]):
        if " " not in chars:
            raise ValueError("the vocabulary must contain the space")
        self.itos = [BLANK, UNK, *chars]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts, min_count: int = 1) -> "CharVocab":
        counts = Counter(ch for t in texts for ch in t)
        return cls(sorted(c for c, n in counts.items() if n >= min_count))

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def chars(self) -> list[str]:
        return self.itos[2:]

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids) -> str:
        """Ids -> text; blank and unknown symbols are dropped, spaces collapsed."""
        out = "".join(self.itos[i] for i in ids if 2 <= i < len(self.itos))
        return " ".join(out.split())

    # attention decoder symbols (after the characters)
    @property
    def sos(self) -> int:
        return len(self.itos)

    @property
    def eos(self) -> int:
        return len(self.itos) + 1

    def to_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> "CharVocab":
        return cls(list(d["chars"]))
