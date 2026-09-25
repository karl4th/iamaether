import numpy as np
import pytest

from kokoro_ds.wav_io import read_wav, wav_duration_seconds, write_stereo_wav_atomic

SR = 24_000


def test_write_and_read_roundtrip(tmp_path):
    left = np.linspace(-0.5, 0.5, SR, dtype=np.float32)
    right = np.linspace(0.5, -0.5, SR, dtype=np.float32)
    stereo = np.stack([left, right], axis=0)
    path = tmp_path / "audio" / "dialogue.wav"

    write_stereo_wav_atomic(path, stereo, SR)

    info = read_wav(path)
    assert info.sample_rate == SR
    assert info.channels == 2
    assert info.n_frames == SR
    assert wav_duration_seconds(info) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(info.data[:, 0], left, atol=2e-4)
    assert np.allclose(info.data[:, 1], right, atol=2e-4)


def test_write_leaves_no_tmp_file_behind(tmp_path):
    stereo = np.zeros((2, SR), dtype=np.float32)
    path = tmp_path / "dialogue.wav"
    write_stereo_wav_atomic(path, stereo, SR)
    siblings = list(path.parent.iterdir())
    assert siblings == [path]


def test_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        write_stereo_wav_atomic(tmp_path / "bad.wav", np.zeros((3, 100), dtype=np.float32), SR)


def test_overwrite_replaces_content(tmp_path):
    path = tmp_path / "dialogue.wav"
    write_stereo_wav_atomic(path, np.zeros((2, SR), dtype=np.float32), SR)
    write_stereo_wav_atomic(path, np.ones((2, SR), dtype=np.float32) * 0.25, SR)
    info = read_wav(path)
    assert info.data[:, 0].mean() == pytest.approx(0.25, abs=2e-4)
