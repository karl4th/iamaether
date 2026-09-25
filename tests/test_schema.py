import json

import pytest

from kokoro_ds.schema import (
    DEFAULT_ALLOWED_CATEGORIES,
    ValidationError,
    load_split_file,
    validate_source_tree,
)

GOOD_RECORD = {
    "id": "batch_0001_0001",
    "category": "general_conversation",
    "turns": [
        {"speaker": "user", "text": "Hello there."},
        {"speaker": "assistant", "text": "Hi, how can I help?"},
    ],
}


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_split_file_accepts_valid_record(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [GOOD_RECORD])
    records, problems = load_split_file(path)
    assert problems == []
    assert len(records) == 1
    assert records[0].id == "batch_0001_0001"


def test_rejects_wrong_first_speaker(tmp_path):
    bad = dict(GOOD_RECORD, id="x1", turns=[{"speaker": "assistant", "text": "hi"}, {"speaker": "user", "text": "hey"}])
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    assert any("first speaker" in reason for _, reason in problems)


def test_rejects_wrong_last_speaker(tmp_path):
    bad = dict(
        GOOD_RECORD,
        id="x2",
        turns=[
            {"speaker": "user", "text": "hi"},
            {"speaker": "assistant", "text": "hey"},
            {"speaker": "user", "text": "ok"},
            {"speaker": "user", "text": "bye"},
        ],
    )
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    reasons = [r for _, r in problems]
    assert any("alternate" in r or "final speaker" in r for r in reasons)


def test_rejects_odd_turn_count(tmp_path):
    bad = dict(GOOD_RECORD, id="x3", turns=GOOD_RECORD["turns"] + [{"speaker": "user", "text": "one more"}])
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    assert any("even count" in r for _, r in problems)


def test_rejects_disallowed_category(tmp_path):
    bad = dict(GOOD_RECORD, id="x4", category="not_a_real_category")
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    assert any("not in the allowed category set" in r for _, r in problems)


def test_rejects_url_in_text(tmp_path):
    bad = dict(GOOD_RECORD, id="x5", turns=[{"speaker": "user", "text": "check http://example.com"}, GOOD_RECORD["turns"][1]])
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    assert any("URL" in r for _, r in problems)


def test_rejects_markdown(tmp_path):
    bad = dict(GOOD_RECORD, id="x6", turns=[{"speaker": "user", "text": "**bold** text"}, GOOD_RECORD["turns"][1]])
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    assert any("markdown" in r for _, r in problems)


def test_rejects_unexpected_keys(tmp_path):
    bad = {**GOOD_RECORD, "id": "x7", "extra_field": True}
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    assert any("unexpected keys" in r for _, r in problems)


def test_rejects_empty_assistant_alignment_transcript(tmp_path):
    bad = dict(GOOD_RECORD, id="x8", turns=[{"speaker": "user", "text": "..."}, {"speaker": "assistant", "text": "!!! ??? ..."}])
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [bad])
    _, problems = load_split_file(path)
    assert any("empty forced-alignment transcript" in r for _, r in problems)


def test_validate_source_tree_detects_duplicate_ids_across_splits(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [GOOD_RECORD])
    write_jsonl(tmp_path / "validation.jsonl", [GOOD_RECORD])
    write_jsonl(tmp_path / "test.jsonl", [dict(GOOD_RECORD, id="other")])
    with pytest.raises(ValidationError) as exc_info:
        validate_source_tree(tmp_path, expected_counts={"train": 1, "validation": 1, "test": 1})
    assert any("split overlap" in reason for _, reason in exc_info.value.problems)


def test_validate_source_tree_checks_expected_counts(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [GOOD_RECORD])
    write_jsonl(tmp_path / "validation.jsonl", [])
    write_jsonl(tmp_path / "test.jsonl", [])
    with pytest.raises(ValidationError) as exc_info:
        validate_source_tree(tmp_path, expected_counts={"train": 5, "validation": 0, "test": 0})
    assert any("expected 5 records" in reason for _, reason in exc_info.value.problems)


def test_validate_source_tree_skips_count_check_when_expected_counts_is_empty_dict(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [GOOD_RECORD])
    write_jsonl(tmp_path / "validation.jsonl", [])
    write_jsonl(tmp_path / "test.jsonl", [])
    result = validate_source_tree(tmp_path, expected_counts={})
    assert len(result["train"]) == 1


def test_validate_source_tree_happy_path(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [GOOD_RECORD])
    write_jsonl(tmp_path / "validation.jsonl", [dict(GOOD_RECORD, id="v1")])
    write_jsonl(tmp_path / "test.jsonl", [dict(GOOD_RECORD, id="t1")])
    result = validate_source_tree(tmp_path, expected_counts={"train": 1, "validation": 1, "test": 1})
    assert set(result.keys()) == {"train", "validation", "test"}
    assert DEFAULT_ALLOWED_CATEGORIES  # sanity: default category list is non-empty
