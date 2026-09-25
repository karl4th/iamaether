import os

from kokoro_ds.manifest import (
    append_jsonl_line_durable,
    atomic_write_json,
    manifest_line,
    read_json,
    read_jsonl,
    write_split_manifest,
)


def test_manifest_line_shape_and_rounding():
    line = manifest_line("audio/batch_0001_0001.wav", 27.72345)
    assert line == {"path": "audio/batch_0001_0001.wav", "duration": 27.723}


def test_atomic_write_json_roundtrip_and_no_leftover_tmp(tmp_path):
    target = tmp_path / "nested" / "report.json"
    atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert target.exists()
    assert read_json(target) == {"a": 1, "b": [1, 2, 3]}
    leftovers = [p for p in target.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_atomic_write_json_overwrites_cleanly(tmp_path):
    target = tmp_path / "report.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert read_json(target) == {"v": 2}


def test_append_jsonl_line_durable_appends_in_order(tmp_path):
    path = tmp_path / "completed.jsonl"
    append_jsonl_line_durable(path, {"id": "a"})
    append_jsonl_line_durable(path, {"id": "b"})
    lines = list(read_jsonl(path))
    assert lines == [{"id": "a"}, {"id": "b"}]


def test_read_jsonl_on_missing_file_yields_nothing(tmp_path):
    assert list(read_jsonl(tmp_path / "does_not_exist.jsonl")) == []


def test_write_split_manifest_is_a_full_rewrite_not_an_append(tmp_path):
    path = tmp_path / "train" / "train.jsonl"
    write_split_manifest(path, [manifest_line("audio/a.wav", 1.0), manifest_line("audio/b.wav", 2.0)])
    write_split_manifest(path, [manifest_line("audio/a.wav", 1.5)])
    lines = list(read_jsonl(path))
    assert lines == [{"path": "audio/a.wav", "duration": 1.5}]


def test_write_split_manifest_leaves_no_partial_file_on_disk(tmp_path):
    path = tmp_path / "validation" / "validation.jsonl"
    write_split_manifest(path, [manifest_line("audio/x.wav", 3.0)])
    entries = os.listdir(path.parent)
    assert entries == ["validation.jsonl"]
