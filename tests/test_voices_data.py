import pytest

from kokoro_ds.voices_data import (
    ASSISTANT_VOICE,
    TEST_USER_VOICES,
    TRAIN_USER_VOICES,
    VALIDATION_USER_VOICES,
    assert_voice_pools_valid,
    lang_code_for_voice,
)


def test_pools_are_pairwise_disjoint():
    assert not (set(TRAIN_USER_VOICES) & set(VALIDATION_USER_VOICES))
    assert not (set(TRAIN_USER_VOICES) & set(TEST_USER_VOICES))
    assert not (set(VALIDATION_USER_VOICES) & set(TEST_USER_VOICES))


def test_assistant_voice_not_in_any_user_pool():
    assert ASSISTANT_VOICE not in TRAIN_USER_VOICES
    assert ASSISTANT_VOICE not in VALIDATION_USER_VOICES
    assert ASSISTANT_VOICE not in TEST_USER_VOICES


def test_assert_voice_pools_valid_passes_by_default():
    assert_voice_pools_valid()  # must not raise


def test_assert_voice_pools_valid_detects_overlap(monkeypatch):
    import kokoro_ds.voices_data as vd

    monkeypatch.setattr(vd, "VALIDATION_USER_VOICES", (TRAIN_USER_VOICES[0],))
    with pytest.raises(ValueError, match="not disjoint"):
        vd.assert_voice_pools_valid()


def test_assert_voice_pools_valid_detects_missing_available_voice():
    with pytest.raises(ValueError, match="not available"):
        assert_voice_pools_valid(available_voices=set())


def test_lang_code_for_voice():
    assert lang_code_for_voice("af_heart") == "a"
    assert lang_code_for_voice("bm_george") == "b"
    with pytest.raises(ValueError):
        lang_code_for_voice("zz_unknown")
