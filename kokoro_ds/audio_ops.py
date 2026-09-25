"""Pure numpy audio assembly primitives.

Everything here operates on plain float32 numpy arrays in ``[-1, 1]`` and has
no Kokoro/torch dependency, so it is fully unit-testable without models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TurnAudio:
    speaker: str  # "user" | "assistant"
    samples: np.ndarray  # mono float32
    pause_before_seconds: float


@dataclass(frozen=True)
class TurnPlacement:
    index: int
    speaker: str
    start_sample: int
    end_sample: int


def apply_fade(samples: np.ndarray, sample_rate: int, fade_seconds: float) -> np.ndarray:
    n = samples.shape[0]
    fade_n = min(int(round(fade_seconds * sample_rate)), n // 2)
    if fade_n <= 0 or n == 0:
        return samples
    out = samples.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    out[:fade_n] *= ramp
    out[-fade_n:] *= ramp[::-1]
    return out


def concat_kokoro_chunks(chunks: list[np.ndarray], sample_rate: int, inter_chunk_pause_seconds: float) -> np.ndarray:
    """Concatenate every chunk a Kokoro generator produced for one utterance, in order.

    Never drops a chunk. A short silence is inserted between chunks so words
    are not run together at chunk boundaries.
    """
    if not chunks:
        raise ValueError("concat_kokoro_chunks received zero chunks")
    non_empty = [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]
    if len(non_empty) == 1:
        return non_empty[0]
    pause_n = max(0, int(round(inter_chunk_pause_seconds * sample_rate)))
    pieces: list[np.ndarray] = []
    for i, chunk in enumerate(non_empty):
        pieces.append(chunk)
        if i < len(non_empty) - 1 and pause_n > 0:
            pieces.append(np.zeros(pause_n, dtype=np.float32))
    return np.concatenate(pieces)


def assemble_stereo(turns: list[TurnAudio], sample_rate: int) -> tuple[np.ndarray, list[TurnPlacement]]:
    """Assemble alternating mono turns into one stereo dialogue.

    Left channel (index 0) carries the assistant (Aether); right channel
    (index 1) carries the user. Whichever side is not speaking is silence of
    equal length, so both channels always have identical total length.
    Returns the stereo array shaped ``(2, n_samples)`` plus per-turn sample
    offsets (post pause-insertion) for downstream alignment timestamp
    offsetting.
    """
    left_segments: list[np.ndarray] = []
    right_segments: list[np.ndarray] = []
    placements: list[TurnPlacement] = []
    cursor = 0

    for idx, turn in enumerate(turns):
        pause_n = max(0, int(round(turn.pause_before_seconds * sample_rate)))
        if pause_n > 0:
            left_segments.append(np.zeros(pause_n, dtype=np.float32))
            right_segments.append(np.zeros(pause_n, dtype=np.float32))
            cursor += pause_n

        samples = np.asarray(turn.samples, dtype=np.float32).reshape(-1)
        n = samples.shape[0]
        silence = np.zeros(n, dtype=np.float32)

        if turn.speaker == "assistant":
            left_segments.append(samples)
            right_segments.append(silence)
        elif turn.speaker == "user":
            left_segments.append(silence)
            right_segments.append(samples)
        else:
            raise ValueError(f"Unknown speaker {turn.speaker!r} at turn {idx}")

        placements.append(TurnPlacement(index=idx, speaker=turn.speaker, start_sample=cursor, end_sample=cursor + n))
        cursor += n

    left = np.concatenate(left_segments) if left_segments else np.zeros(0, dtype=np.float32)
    right = np.concatenate(right_segments) if right_segments else np.zeros(0, dtype=np.float32)
    if left.shape[0] != right.shape[0]:
        raise AssertionError("internal error: stereo channels drifted out of length-sync")

    stereo = np.stack([left, right], axis=0)
    return stereo, placements


def peak_amplitude(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def rms_amplitude(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def is_clipping(samples: np.ndarray, threshold: float) -> bool:
    return peak_amplitude(samples) >= threshold


def has_speech(samples: np.ndarray, floor: float = 1e-4) -> bool:
    return peak_amplitude(samples) > floor


def all_finite(samples: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(samples)))


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)
