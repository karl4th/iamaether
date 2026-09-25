from kokoro_ds.checksums import (
    generation_config_checksum,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    source_dialogue_checksum,
)


def test_sha256_text_is_stable():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")


def test_sha256_bytes_matches_text():
    assert sha256_bytes("hello".encode("utf-8")) == sha256_text("hello")


def test_sha256_json_is_key_order_independent():
    a = sha256_json({"a": 1, "b": 2})
    b = sha256_json({"b": 2, "a": 1})
    assert a == b


def test_sha256_json_detects_value_change():
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_sha256_file_matches_content(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"some binary content" * 1000)
    assert sha256_file(path) == sha256_bytes(path.read_bytes())


def test_source_dialogue_checksum_ignores_line_no_and_extra_fields():
    record_a = {"id": "x", "category": "c", "turns": [{"speaker": "user", "text": "hi"}]}
    record_b = {"id": "x", "category": "c", "turns": [{"speaker": "user", "text": "hi"}], "line_no": 999}
    assert source_dialogue_checksum(record_a) == source_dialogue_checksum(record_b)


def test_source_dialogue_checksum_changes_with_text():
    record_a = {"id": "x", "category": "c", "turns": [{"speaker": "user", "text": "hi"}]}
    record_b = {"id": "x", "category": "c", "turns": [{"speaker": "user", "text": "hi there"}]}
    assert source_dialogue_checksum(record_a) != source_dialogue_checksum(record_b)


def test_generation_config_checksum_changes_with_voice_or_split():
    cfg = {"a": 1}
    c1 = generation_config_checksum(cfg, "train", "af_alloy")
    c2 = generation_config_checksum(cfg, "train", "af_bella")
    c3 = generation_config_checksum(cfg, "validation", "af_alloy")
    assert len({c1, c2, c3}) == 3
