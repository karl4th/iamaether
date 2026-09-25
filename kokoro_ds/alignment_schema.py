"""Structural validation of the forced-alignment output format.

This module only checks *shape and invariants* of an already-produced
alignment list. It has no torch/torchaudio dependency, so both the generator
and the validator (and their unit tests) can check alignment correctness
without loading any model.

Required per-word structure::

    ["Aether", [3.42, 3.88], "SPEAKER_MAIN"]
"""
from __future__ import annotations

import math

from .config import SPEAKER_MAIN_LABEL

DURATION_TOLERANCE_SECONDS = 0.05


def validate_alignment_list(alignments: object, duration_seconds: float) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    problems: list[str] = []

    if not isinstance(alignments, list):
        return ["alignments must be a list"]
    if len(alignments) == 0:
        return ["alignments must be nonempty"]

    prev_end: float | None = None
    for i, item in enumerate(alignments):
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            problems.append(f"alignments[{i}] must be a 3-element [word, [start, end], speaker] entry")
            continue

        word, span, speaker = item
        if not isinstance(word, str) or not word.strip():
            problems.append(f"alignments[{i}].word must be a nonempty string")
        if speaker != SPEAKER_MAIN_LABEL:
            problems.append(f"alignments[{i}].speaker must be {SPEAKER_MAIN_LABEL!r}, got {speaker!r}")

        if not (isinstance(span, (list, tuple)) and len(span) == 2):
            problems.append(f"alignments[{i}].span must be a 2-element [start, end]")
            continue

        start, end = span
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            problems.append(f"alignments[{i}] timestamps must be numeric")
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            problems.append(f"alignments[{i}] timestamps must be finite")
            continue
        if start < 0:
            problems.append(f"alignments[{i}] start must be nonnegative, got {start}")
        if end <= start:
            problems.append(f"alignments[{i}] end ({end}) must be greater than start ({start})")
        if end > duration_seconds + DURATION_TOLERANCE_SECONDS:
            problems.append(
                f"alignments[{i}] end ({end}) exceeds WAV duration ({duration_seconds})"
            )
        if prev_end is not None and start < prev_end:
            problems.append(
                f"alignments[{i}] start ({start}) is not monotonically increasing after previous end ({prev_end})"
            )
        prev_end = end

    return problems


def is_alignment_valid(alignments: object, duration_seconds: float) -> bool:
    return len(validate_alignment_list(alignments, duration_seconds)) == 0
