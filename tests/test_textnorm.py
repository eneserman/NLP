from hrasr.textnorm import (clean_reference, has_annotation, has_digits, has_non_latin_letters,
                            normalize_for_scoring)


def test_scoring_keeps_croatian_letters():
    assert normalize_for_scoring("Čestitam, ŽELJKO! Đurđa ćuti, šuti.") == "čestitam željko đurđa ćuti šuti"


def test_scoring_removes_punctuation_and_extra_spaces():
    assert normalize_for_scoring('  „Hvala“ —   lijepa ,  hvala. ') == "hvala lijepa hvala"


def test_scoring_is_idempotent():
    s = normalize_for_scoring("Gospodine predsjedniče, hvala!")
    assert normalize_for_scoring(s) == s


def test_scoring_unicode_forms_match():
    composed = "č"          # one code point
    decomposed = "č"  # c + combining caron
    assert normalize_for_scoring(composed) == normalize_for_scoring(decomposed)


def test_clean_reference_fixes_format_but_keeps_case():
    assert clean_reference("Hvala lijepa​  –  „Sabor“") == 'Hvala lijepa - "Sabor"'


def test_detectors():
    assert has_digits("članak 5.")
    assert not has_digits("članak pet")
    assert has_annotation("(Upadica: Neka govori.)")
    assert not has_annotation("Hvala, predsjedniče.")
    assert has_non_latin_letters("Хвала")
    assert not has_non_latin_letters("Hvala Müller")
