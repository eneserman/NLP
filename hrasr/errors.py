"""Error taxonomy (W9 slide 13: names, quantities, negation and other important content).

For each reference word we know its category, and the WER alignment tells us whether the
system produced it ("hit"). We report recall per category: hits / reference words.

* name: the word starts with a capital letter and is not the first word of a sentence
  (Croatian writes common nouns in lower case, so this finds names of people, parties, places,
  laws and institutions).
* negation: Croatian negation words; dropping one inverts the meaning of a sentence.
* all: every word, for comparison.

Quantities are not a category here: clips with digits are removed in cleaning (see README), so
spelled-out numbers are rare in the references.
"""
from __future__ import annotations

from collections import Counter

import jiwer

from hrasr.textnorm import normalize_for_scoring

NEGATION = frozenset("""ne nije nisu nisam nisi nismo niste nema nemam nemaš nemamo nemate nemaju
neće neću nećeš nećemo nećete ni niti nikad nikada nitko ništa nijedan nijedna nijedno""".split())


def tag_reference(orth: str) -> list[tuple[str, set[str]]]:
    """Normalized reference words with their categories, in the same order as normalize_for_scoring."""
    out: list[tuple[str, set[str]]] = []
    sentence_start = True
    for tok in orth.split():
        pieces = normalize_for_scoring(tok).split()
        first_alpha = next((ch for ch in tok if ch.isalpha()), "")
        is_name = bool(first_alpha) and first_alpha.isupper() and not sentence_start
        for p in pieces:
            tags = {"all"}
            if is_name:
                tags.add("name")
            if p in NEGATION:
                tags.add("negation")
            out.append((p, tags))
        if pieces:
            sentence_start = tok.rstrip("\"')»”").endswith((".", "!", "?", ":"))
    return out


def category_recall(refs_orth: list[str], hyps: list[str], top_missed: int = 10) -> dict:
    counts = {c: [0, 0] for c in ("all", "name", "negation")}  # [hits, total]
    missed_names: Counter = Counter()
    for ref, hyp in zip(refs_orth, hyps):
        tagged = tag_reference(ref)
        ref_norm = " ".join(w for w, _ in tagged)
        assert ref_norm == normalize_for_scoring(ref)
        hit = [False] * len(tagged)
        out = jiwer.process_words(ref_norm, normalize_for_scoring(hyp))
        for chunk in out.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    hit[i] = True
        for (word, tags), h in zip(tagged, hit):
            for t in tags:
                counts[t][0] += int(h)
                counts[t][1] += 1
            if "name" in tags and not h:
                missed_names[word] += 1
    return {
        **{c: {"recall": (h / n if n else None), "ref_words": n} for c, (h, n) in counts.items()},
        "most_missed_names": missed_names.most_common(top_missed),
    }
