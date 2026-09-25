import numpy as np
import pytest

from kokoro_ds.audio_ops import (
    TurnAudio,
    all_finite,
    apply_fade,
    assemble_stereo,
    concat_kokoro_chunks,
    float_to_pcm16,
    has_speech,
    is_clipping,
    peak_amplitude,
    rms_amplitude,
)

SR = 24_000


def const_wave(seconds: float, value: float = 0.5) -> np.ndarray:
    return np.full(int(seconds * SR), value, dtype=np.float32)


def test_assemble_stereo_channel_layout_left_is_assistant_right_is_user():
    turns = [
        TurnAudio("user", const_wave(0.1, 0.5), pause_before_seconds=0.0),
        TurnAudio("assistant", const_wave(0.1, 0.8), pause_before_seconds=0.0),
    ]
    stereo, placements = assemble_stereo(turns, SR)
    assert stereo.shape[0] == 2

    user_start, user_end = placements[0].start_sample, placements[0].end_sample
    assistant_start, assistant_end = placements[1].start_sample, placements[1].end_sample

    # Right channel carries the user's speech; left is silent during that span.
    assert np.allclose(stereo[1, user_start:user_end], 0.5)
    assert np.allclose(stereo[0, user_start:user_end], 0.0)

    # Left channel carries the assistant's speech; right is silent during that span.
    assert np.allclose(stereo[0, assistant_start:assistant_end], 0.8)
    assert np.allclose(stereo[1, assistant_start:assistant_end], 0.0)


def test_assemble_stereo_channels_stay_equal_length():
    turns = [
        TurnAudio("user", const_wave(0.13), 0.0),
        TurnAudio("assistant", const_wave(0.07), 0.31),
        TurnAudio("user", const_wave(0.05), 0.5),
    ]
    stereo, placements = assemble_stereo(turns, SR)
    assert stereo[0].shape[0] == stereo[1].shape[0]
    assert placements[-1].end_sample == stereo.shape[1]


def test_assemble_stereo_pause_is_silence_on_both_channels():
    pause_seconds = 0.4
    turns = [
        TurnAudio("user", const_wave(0.1), 0.0),
        TurnAudio("assistant", const_wave(0.1), pause_seconds),
    ]
    stereo, placements = assemble_stereo(turns, SR)
    gap_start = placements[0].end_sample
    gap_end = placements[1].start_sample
    assert gap_end - gap_start == pytest.approx(int(pause_seconds * SR), abs=1)
    assert np.all(stereo[:, gap_start:gap_end] == 0.0)


def test_assemble_stereo_rejects_unknown_speaker():
    with pytest.raises(ValueError):
        assemble_stereo([TurnAudio("narrator", const_wave(0.1), 0.0)], SR)


def test_apply_fade_zeroes_endpoints_and_preserves_middle():
    wave = const_wave(0.2, 1.0)
    faded = apply_fade(wave, SR, fade_seconds=0.02)
    assert faded[0] == pytest.approx(0.0, abs=1e-3)
    assert faded[-1] == pytest.approx(0.0, abs=1e-3)
    assert faded[len(faded) // 2] == pytest.approx(1.0, abs=1e-6)


def test_apply_fade_no_op_on_zero_length():
    assert apply_fade(np.zeros(0, dtype=np.float32), SR, 0.02).shape[0] == 0


def test_concat_kokoro_chunks_never_drops_a_chunk():
    chunks = [const_wave(0.05, 0.1), const_wave(0.05, 0.2), const_wave(0.05, 0.3)]
    joined = concat_kokoro_chunks(chunks, SR, inter_chunk_pause_seconds=0.01)
    total_speech_samples = sum(c.shape[0] for c in chunks)
    pause_samples = 2 * int(0.01 * SR)
    assert joined.shape[0] == total_speech_samples + pause_samples


def test_concat_kokoro_chunks_single_chunk_passthrough():
    chunk = const_wave(0.05, 0.42)
    joined = concat_kokoro_chunks([chunk], SR, inter_chunk_pause_seconds=0.5)
    assert joined.shape[0] == chunk.shape[0]


def test_concat_kokoro_chunks_rejects_empty_list():
    with pytest.raises(ValueError):
        concat_kokoro_chunks([], SR, 0.1)


def test_peak_and_rms_and_clipping():
    wave = const_wave(0.1, 0.5)
    assert peak_amplitude(wave) == pytest.approx(0.5)
    assert rms_amplitude(wave) == pytest.approx(0.5)
    assert not is_clipping(wave, threshold=0.999)
    assert is_clipping(const_wave(0.01, 1.0), threshold=0.999)


def test_has_speech_and_all_finite():
    assert has_speech(const_wave(0.1, 0.1))
    assert not has_speech(np.zeros(100, dtype=np.float32))
    assert all_finite(const_wave(0.1, 0.1))
    bad = const_wave(0.1, 0.1)
    bad[0] = np.nan
    assert not all_finite(bad)


def test_float_to_pcm16_clips_and_scales():
    wave = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm = float_to_pcm16(wave)
    assert pcm.dtype == np.int16
    assert pcm[0] == pcm[1] == -32767
    assert pcm[3] == pcm[4] == 32767
