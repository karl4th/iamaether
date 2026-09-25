from kokoro_ds.determinism import (
    assign_turn_pause_seconds,
    assign_user_turn_speed,
    assign_user_voice,
)
from kokoro_ds.voices_data import TRAIN_USER_VOICES, USER_SPEED_MAX, USER_SPEED_MIN


def test_assign_user_voice_is_deterministic_across_calls():
    v1 = assign_user_voice("batch_0001_0001", "train", global_seed=0)
    v2 = assign_user_voice("batch_0001_0001", "train", global_seed=0)
    assert v1 == v2
    assert v1 in TRAIN_USER_VOICES


def test_assign_user_voice_varies_by_dialogue_id():
    voices = {assign_user_voice(f"batch_0001_{i:04d}", "train", global_seed=0) for i in range(30)}
    assert len(voices) > 1


def test_assign_user_voice_changes_with_seed():
    assign_user_voice("batch_0001_0001", "train", global_seed=0)
    assign_user_voice("batch_0001_0001", "train", global_seed=1)
    # different seeds are allowed to coincide occasionally on a single id; check independence over many ids instead
    diffs = sum(
        assign_user_voice(f"id_{i}", "train", 0) != assign_user_voice(f"id_{i}", "train", 1) for i in range(50)
    )
    assert diffs > 0


def test_assign_user_turn_speed_deterministic_and_in_range():
    for turn_idx in range(4):
        s1 = assign_user_turn_speed("batch_0001_0001", turn_idx, 0, USER_SPEED_MIN, USER_SPEED_MAX)
        s2 = assign_user_turn_speed("batch_0001_0001", turn_idx, 0, USER_SPEED_MIN, USER_SPEED_MAX)
        assert s1 == s2
        assert USER_SPEED_MIN <= s1 <= USER_SPEED_MAX


def test_assign_user_turn_speed_varies_by_turn_index():
    speeds = {assign_user_turn_speed("batch_0001_0001", i, 0, USER_SPEED_MIN, USER_SPEED_MAX) for i in range(6)}
    assert len(speeds) > 1


def test_assign_turn_pause_seconds_zero_for_first_gap():
    assert assign_turn_pause_seconds("batch_0001_0001", 0, 0, 0.25, 0.75) == 0.0
    assert assign_turn_pause_seconds("batch_0001_0001", -1, 0, 0.25, 0.75) == 0.0


def test_assign_turn_pause_seconds_deterministic_and_in_range():
    for gap_idx in range(1, 5):
        p1 = assign_turn_pause_seconds("batch_0001_0001", gap_idx, 0, 0.25, 0.75)
        p2 = assign_turn_pause_seconds("batch_0001_0001", gap_idx, 0, 0.25, 0.75)
        assert p1 == p2
        assert 0.25 <= p1 <= 0.75


def test_full_dialogue_assignment_reproducible_end_to_end():
    """Regenerating the same record id must reproduce identical voice, speeds, and pauses."""

    def snapshot(seed: int) -> tuple:
        voice = assign_user_voice("batch_0001_0042", "validation", seed)
        speeds = tuple(assign_user_turn_speed("batch_0001_0042", i, seed, USER_SPEED_MIN, USER_SPEED_MAX) for i in range(4))
        pauses = tuple(assign_turn_pause_seconds("batch_0001_0042", i, seed, 0.25, 0.75) for i in range(4))
        return voice, speeds, pauses

    assert snapshot(7) == snapshot(7)
