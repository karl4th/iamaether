from kokoro_ds.alignment_schema import is_alignment_valid, validate_alignment_list
from kokoro_ds.config import SPEAKER_MAIN_LABEL


def good(word, start, end):
    return [word, [start, end], SPEAKER_MAIN_LABEL]


def test_valid_alignment_passes():
    alignments = [good("Hello", 0.0, 0.3), good("world", 0.35, 0.6)]
    assert validate_alignment_list(alignments, duration_seconds=1.0) == []
    assert is_alignment_valid(alignments, duration_seconds=1.0)


def test_empty_alignment_list_is_invalid():
    assert validate_alignment_list([], duration_seconds=1.0) != []
    assert not is_alignment_valid([], duration_seconds=1.0)


def test_non_list_is_invalid():
    assert validate_alignment_list("not a list", duration_seconds=1.0) != []


def test_wrong_speaker_label_is_invalid():
    problems = validate_alignment_list([["hi", [0.0, 0.1], "SOMEONE_ELSE"]], duration_seconds=1.0)
    assert any("speaker" in p for p in problems)


def test_negative_start_is_invalid():
    problems = validate_alignment_list([good("hi", -0.1, 0.1)], duration_seconds=1.0)
    assert any("nonnegative" in p for p in problems)


def test_end_not_greater_than_start_is_invalid():
    problems = validate_alignment_list([good("hi", 0.5, 0.5)], duration_seconds=1.0)
    assert any("greater than start" in p for p in problems)


def test_non_finite_timestamps_invalid():
    problems = validate_alignment_list([good("hi", float("nan"), 0.5)], duration_seconds=1.0)
    assert any("finite" in p for p in problems)
    problems = validate_alignment_list([good("hi", 0.0, float("inf"))], duration_seconds=1.0)
    assert problems  # inf end also exceeds duration and is non-finite


def test_end_beyond_duration_is_invalid():
    problems = validate_alignment_list([good("hi", 0.0, 5.0)], duration_seconds=1.0)
    assert any("exceeds WAV duration" in p for p in problems)


def test_non_monotonic_sequence_is_invalid():
    alignments = [good("second", 0.5, 0.8), good("first", 0.0, 0.3)]
    problems = validate_alignment_list(alignments, duration_seconds=1.0)
    assert any("monotonically increasing" in p for p in problems)


def test_overlapping_words_are_invalid():
    alignments = [good("a", 0.0, 0.5), good("b", 0.3, 0.6)]
    problems = validate_alignment_list(alignments, duration_seconds=1.0)
    assert any("monotonically increasing" in p for p in problems)


def test_malformed_entry_shape_is_invalid():
    problems = validate_alignment_list([["only_two_fields", [0, 1]]], duration_seconds=1.0)
    assert any("3-element" in p for p in problems)
