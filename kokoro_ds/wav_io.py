"""Atomic, PCM16 stereo WAV I/O."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    n_frames: int
    data: np.ndarray  # shape (n_frames, channels), float32 in [-1, 1]


def write_stereo_wav_atomic(path: Path, stereo_float: np.ndarray, sample_rate: int) -> None:
    """Write a ``(2, n_samples)`` float32 array as PCM16 WAV, atomically.

    Writes to a temp file in the same directory then ``os.replace``s it into
    place, so a Colab interruption never leaves a partially written file at
    the final path.
    """
    if stereo_float.ndim != 2 or stereo_float.shape[0] != 2:
        raise ValueError(f"expected shape (2, n_samples), got {stereo_float.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(stereo_float.T.astype(np.float32))
    tmp_path = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        sf.write(str(tmp_path), data, sample_rate, subtype="PCM_16", format="WAV")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_wav(path: Path) -> WavInfo:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return WavInfo(sample_rate=sample_rate, channels=data.shape[1], n_frames=data.shape[0], data=data)


def wav_duration_seconds(info: WavInfo) -> float:
    if info.sample_rate <= 0:
        return 0.0
    return info.n_frames / float(info.sample_rate)
