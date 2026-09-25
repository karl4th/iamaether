"""Deterministic per-dialogue assignment of voice, speed, and pauses.

Every value derived here is a pure function of ``(global_seed, dialogue_id,
purpose)``. Regenerating the same dialogue id under the same global seed and
config version always reproduces identical voice, speeds, and pause
durations, independent of process order or prior runs.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .voices_data import VOICE_POOLS_BY_SPLIT


def _seed_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def dialogue_rng(dialogue_id: str, global_seed: int, purpose: str) -> np.random.Generator:
    return np.random.default_rng(_seed_int(str(global_seed), dialogue_id, purpose))


def assign_user_voice(dialogue_id: str, split: str, global_seed: int) -> str:
    """Deterministically pick a user voice from the split's disjoint pool."""
    pool = VOICE_POOLS_BY_SPLIT.get(split)
    if not pool:
        raise ValueError(f"Unknown split {split!r}; no user voice pool configured")
    rng = dialogue_rng(dialogue_id, global_seed, "user_voice")
    index = int(rng.integers(0, len(pool)))
    return pool[index]


def assign_user_turn_speed(
    dialogue_id: str, turn_index: int, global_seed: int, speed_min: float, speed_max: float
) -> float:
    rng = dialogue_rng(dialogue_id, global_seed, f"user_speed:{turn_index}")
    return float(rng.uniform(speed_min, speed_max))


def assign_turn_pause_seconds(
    dialogue_id: str, gap_index: int, global_seed: int, pause_min: float, pause_max: float
) -> float:
    """Silence duration inserted before the turn at ``gap_index`` (0 = before the first turn, always 0)."""
    if gap_index <= 0:
        return 0.0
    rng = dialogue_rng(dialogue_id, global_seed, f"pause:{gap_index}")
    return float(rng.uniform(pause_min, pause_max))
