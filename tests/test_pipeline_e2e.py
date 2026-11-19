"""Runs steps 1-8 and unpack on a small synthetic corpus with the same Parquet schema as the
Hub dataset (audio stored as struct<bytes, path>). Checks that raw data turns into cleaned,
split data and that known bad clips are removed for the right reason."""
import csv
import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from hrasr.config import ROOT
from hrasr.manifest import read_jsonl

SENT = "poštovani zastupnici danas raspravljamo o prijedlogu zakona o proračunu"


def tone(seconds: float, freq: int, volume: float = 0.3) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency={freq}:duration={seconds}:sample_rate=48000",
         "-af", f"volume={volume}", "-ac", "1", "-f", "flac", "pipe:1"],
        capture_output=True, check=True).stdout


def build_corpus(folder):
    rows = []
    for s in range(8):
        new = s < 4
        for c in range(6):
            dur = 5.0
            date = f"{2021 if new else 2015}-0{s + 1}-0{c // 2 + 1}"  # 3 sessions per speaker, 2 clips each
            rows.append({
                "id": f"ParlaMint-HR_{date}-0.u{s}_{c}", "text": SENT.capitalize() + ".",
                "text_normalised": SENT, "audio_length": dur,
                "date": date,
                "speaker_name": f"Zastupnik {s}", "speaker_gender": "F" if s % 2 else "M",
                "speaker_birth": "1970", "speaker_party": "X",
                "words": [{"char_s": 0, "char_e": 5, "time_s": 200, "time_e": 2000},
                          {"char_s": 6, "char_e": 10, "time_s": 2100, "time_e": 4800}],
                "audio": {"bytes": tone(dur, 200 + 37 * s + 7 * c), "path": f"x{s}_{c}.flac"},
            })
    # Planted problems, in the clips of training speakers 4 and 5 (rows 24-35).
    rows[25]["text"] = "Članak 5. zakona o proračunu za iduću godinu je prihvaćen."     # digits
    rows[26]["audio"] = {"bytes": rows[24]["audio"]["bytes"], "path": "dup.flac"}     # duplicate audio
    rows[27]["audio"] = {"bytes": tone(5.0, 300, volume=0.0001), "path": "quiet.flac"}  # silence
    rows[28]["text"] = "Hvala (Upadica: ne) hvala lijepa svima na pažnji danas ovdje."   # annotation
    rows[31]["audio_length"] = 9.0                                                     # metadata disagrees with audio
    table = pa.Table.from_pylist(rows)
    folder.mkdir(parents=True)
    pq.write_table(table.slice(0, 24), folder / "train-00000-of-00002.parquet", row_group_size=8)
    pq.write_table(table.slice(24), folder / "train-00001-of-00002.parquet", row_group_size=8)
    return rows


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0, reason="needs ffmpeg")
def test_pipeline(tmp_path):
    src = tmp_path / "hub"
    rows = build_corpus(src)
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    cfg["selection"].update(train_hours=0.05, dev_hours=0.01, test_hours=0.01, oversample=1.0,
                            min_hours_per_eval_speaker=0.001)
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    env = {**os.environ, "HRASR_DATA": str(tmp_path / "data"), "HRASR_WORK": str(tmp_path / "work"),
           "PYTHONPATH": str(ROOT)}

    def run(script, *extra):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--config", str(cfg_path), *extra],
                       env=env, check=True, capture_output=True)

    v1 = tmp_path / "v1.jsonl"
    v1.write_text(json.dumps({"path": "x.flac", "speaker_info": {"Speaker_name": "Zastupnik 3"}}) + "\n"
                  + json.dumps({"path": "y.flac", "speaker_info": None}) + "\n", encoding="utf-8")

    run("01_fetch_metadata.py", "--local-dir", str(src))
    run("02_fetch_v1_speakers.py", "--local-file", str(v1))
    run("03_select_subset.py")
    run("04_fetch_audio.py", "--local-dir", str(src))
    run("05_clean.py")
    run("06_split.py")
    run("07_make_silence_probe.py")
    run("08_pack_audio.py")

    data = tmp_path / "data"
    raw = list(read_jsonl(data / "raw" / "manifest.jsonl.gz"))
    assert raw and all(r["text"] for r in raw)

    with open(data / "clean" / "cleaning_log.csv", encoding="utf-8") as f:
        removed = {r["id"]: r["reason"] for r in csv.DictReader(f)}
    raw_ids = {r["id"] for r in raw}
    expect = {rows[25]["id"]: "digits", rows[26]["id"]: "duplicate_audio", rows[27]["id"]: "silence",
              rows[28]["id"]: "annotation", rows[31]["id"]: "duration_mismatch"}
    for cid, reason in expect.items():
        assert cid in raw_ids, cid
        assert removed[cid] == reason, (cid, removed.get(cid))

    splits = {n: list(read_jsonl(data / "clean" / f"{n}.jsonl")) for n in ("train", "dev", "test")}
    speakers = {n: {r["speaker_name"] for r in v} for n, v in splits.items()}
    assert splits["test"] and splits["dev"] and splits["train"]
    assert not speakers["train"] & speakers["test"] and not speakers["dev"] & speakers["test"]
    assert not speakers["train"] & speakers["dev"]
    assert "Zastupnik 3" not in speakers["test"] | speakers["dev"]  # listed in v1.0
    sessions = {n: {r["session"] for r in v} for n, v in splits.items()}
    assert not sessions["train"] & sessions["test"] and not sessions["train"] & sessions["dev"]
    assert not sessions["dev"] & sessions["test"]
    assert all(r["id"] not in removed for v in splits.values() for r in v)
    stats = json.loads((data / "clean" / "split_stats.json").read_text(encoding="utf-8"))
    assert stats["test"]["sentences_also_in_train"] >= 1  # every clip has the same sentence here

    probe = list(read_jsonl(data / "clean" / "silence_probe.jsonl"))
    assert len(probe) == 60 and all(r["text"] == "" for r in probe)

    shards = list((data / "audio").glob("shard_*.tar"))
    assert shards
    subprocess.run([sys.executable, str(ROOT / "scripts" / "unpack_audio.py")], env={**env, "HRASR_WORK": str(tmp_path / "work2")},
                   check=True, capture_output=True)
    assert all((tmp_path / "work2" / r["audio"]).exists() for r in splits["test"] + probe)

    # step 5 runs from the committed data alone: audio is unpacked from data/audio when work/ is empty
    subprocess.run([sys.executable, str(ROOT / "scripts" / "05_clean.py"), "--config", str(cfg_path)],
                   env={**env, "HRASR_WORK": str(tmp_path / "work3")}, check=True, capture_output=True)
