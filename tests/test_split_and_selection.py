import pandas as pd
import pytest
import yaml

from hrasr.config import ROOT
from hrasr.selection import select
from hrasr.split import assign_splits, check_disjoint

CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
GOOD_TEXT = "poštovani zastupnici danas raspravljamo o prijedlogu zakona"
WORDS = [{"time_s": 200, "time_e": 900}, {"time_s": 1000, "time_e": 3800}]


def fake_corpus(n_speakers=80, clips=120):
    """Even speakers are "new" (first speak in 2021); odd ones spoke in 2016.
    Each session (date) has several speakers, like a real sitting."""
    rows = []
    for s in range(n_speakers):
        new = s % 2 == 0
        gender = "F" if s % 4 < 2 else "M"
        for c in range(clips):
            year = 2021 if new else 2016
            date = f"{year}-0{1 + c % 9}-{10 + c % 17}"
            rows.append({
                "id": f"ParlaMint-HR_{date}-0.u{s}_{c}", "speaker_name": f"Speaker {s}", "speaker_gender": gender,
                "date": date, "audio_length": 4.0,
                "file": f"data/train-{s % 7:05d}.parquet", "row_group": int(c // 20),
                "text": GOOD_TEXT if c % 10 else "članak 5 zakona o radu i drugi zakoni danas",  # 10% fail cleaning
                "words": WORDS,
            })
    rows.append({**rows[0], "id": "no_speaker", "speaker_name": None})
    return pd.DataFrame(rows)


def small_cfg():
    cfg = yaml.safe_load(yaml.safe_dump(CFG))
    cfg["selection"].update(train_hours=4, dev_hours=0.5, test_hours=0.5, oversample=1.0,
                            max_hours_per_train_speaker=0.1, max_hours_per_eval_speaker=0.05,
                            min_hours_per_eval_speaker=0.01)
    return cfg


def with_text_norm(sel):
    from hrasr.textnorm import normalize_for_scoring
    return sel.assign(text_norm=[normalize_for_scoring(t) for t in sel["text"]])


def test_selection_rules():
    cfg = small_cfg()
    v1 = {"Speaker 0", "speaker 2"}  # name matching ignores case
    sel, rep = select(fake_corpus(), cfg, excluded_eval_speakers=v1)
    assert rep["rows_without_speaker"] == 1
    assert rep["speakers_excluded_as_in_v1"] == 2
    ev = sel[sel["role"].isin(["test", "dev"])]
    tr = sel[sel["role"] == "train"]
    te, de = sel[sel["role"] == "test"], sel[sel["role"] == "dev"]
    assert not set(te["speaker_name"]) & set(de["speaker_name"])
    assert not set(te["session"]) & set(de["session"])
    assert not {"Speaker 0", "Speaker 2"} & set(ev["speaker_name"])
    assert (ev.groupby("speaker_name")["date"].min() >= cfg["selection"]["new_speaker_first_seen_on_or_after"]).all()
    assert not set(ev["speaker_name"]) & set(tr["speaker_name"])
    assert not set(ev["session"]) & set(tr["session"])
    # caps and targets are counted in audio expected to survive cleaning
    good_tr = tr[tr["predicted_clean"]]
    assert (good_tr.groupby("speaker_name")["audio_length"].sum() <= 0.1 * 3600 + 1e-6).all()
    assert good_tr["audio_length"].sum() / 3600 == pytest.approx(4, abs=0.01)
    # problem clips are kept in the raw selection so that cleaning can remove them with a reason
    assert (~sel["predicted_clean"]).sum() > 0


def test_selection_is_deterministic():
    cfg = small_cfg()
    a, _ = select(fake_corpus(), cfg)
    b, _ = select(fake_corpus(), cfg)
    assert list(a["id"]) == list(b["id"])


def test_split_is_speaker_and_session_disjoint():
    cfg = small_cfg()
    sel, _ = select(fake_corpus(), cfg)
    sel = with_text_norm(sel[sel["predicted_clean"]])
    df, stats, removed = assign_splits(sel, cfg)
    check_disjoint(df)
    assert set(stats) >= {"train", "dev", "test"}
    assert stats["test"]["hours"] >= 0.5 and stats["dev"]["hours"] >= 0.5
    for k in ("dev", "test"):
        assert stats[k]["sentences_also_in_train"] == stats[k]["utterances"]  # same sentence everywhere here
    assert len(removed) == 0  # selection already planned separate sessions


def test_session_overlap_is_removed():
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"], "speaker_name": ["T", "D", "D", "X"], "speaker_gender": ["F", "M", "M", "F"],
        "audio_length": [3600.0] * 4, "role": ["test", "dev", "dev", "train"],
        "session": ["s1", "s1", "s2", "s2"], "text_norm": ["x"] * 4,
    })
    out, stats, removed = assign_splits(df)
    check_disjoint(out)
    assert list(removed["id"]) == ["b", "d"]
    assert list(removed["reason"]) == ["session_shared_with_test", "session_shared_with_dev"]


def test_disjoint_check_catches_leak():
    with pytest.raises(AssertionError):
        check_disjoint(pd.DataFrame({"speaker_name": ["a", "a"], "session": ["x", "y"], "split": ["train", "test"]}))
    with pytest.raises(AssertionError):
        check_disjoint(pd.DataFrame({"speaker_name": ["a", "b"], "session": ["x", "x"], "split": ["train", "test"]}))
