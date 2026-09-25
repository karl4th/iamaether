"""Deterministic stratified selection for the smoke-test subset.

Goal: pick ``total`` (default 50) dialogues spread across train / validation
/ test, proportional to split size, while guaranteeing (whenever the source
data mathematically allows it) that the selected subset contains at least
one dialogue for every category and every configured user voice in every
split's pool. This is not "take the first N rows" — coverage is solved for
explicitly and any unmet coverage goal is reported rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

from .determinism import assign_user_voice
from .schema import SourceRecord
from .voices_data import VOICE_POOLS_BY_SPLIT

SPLIT_ORDER = ("train", "validation", "test")


@dataclass
class SmokeSelection:
    ids_by_split: dict[str, list[str]]
    covered_categories: set[str]
    missing_categories: set[str]
    covered_voices_by_split: dict[str, set[str]]
    missing_voices_by_split: dict[str, set[str]]

    def total(self) -> int:
        return sum(len(v) for v in self.ids_by_split.values())


def _largest_remainder_allocate(weights: dict[str, int], total: int) -> dict[str, int]:
    total_weight = sum(weights.values())
    if total_weight <= 0 or total <= 0:
        return {k: 0 for k in weights}
    raw = {k: (w / total_weight) * total for k, w in weights.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(weights.keys(), key=lambda k: (raw[k] - floors[k]), reverse=True)
    for i in range(remainder):
        floors[order[i % len(order)]] += 1
    return floors


def compute_split_targets(split_sizes: dict[str, int], voice_pool_sizes: dict[str, int], total: int) -> dict[str, int]:
    splits = [s for s in SPLIT_ORDER if s in split_sizes]
    mins = {s: min(voice_pool_sizes.get(s, 0), split_sizes.get(s, 0)) for s in splits}
    base_total = sum(mins.values())

    if base_total >= total:
        # Not enough smoke-test budget to guarantee voice coverage everywhere;
        # fall back to proportional allocation and let coverage reporting
        # surface the shortfall.
        return _largest_remainder_allocate({s: split_sizes[s] for s in splits}, total)

    remaining = total - base_total
    extra = _largest_remainder_allocate({s: split_sizes[s] for s in splits}, remaining)
    targets = {s: mins[s] + extra.get(s, 0) for s in splits}

    overflow = 0
    for s in splits:
        if targets[s] > split_sizes[s]:
            overflow += targets[s] - split_sizes[s]
            targets[s] = split_sizes[s]
    if overflow > 0:
        for s in sorted(splits, key=lambda s: split_sizes[s] - targets[s], reverse=True):
            room = split_sizes[s] - targets[s]
            take = min(room, overflow)
            targets[s] += take
            overflow -= take
            if overflow <= 0:
                break

    return targets


def select_stratified_smoke(
    records_by_split: dict[str, list[SourceRecord]],
    global_seed: int,
    total: int = 50,
) -> SmokeSelection:
    splits = [s for s in SPLIT_ORDER if s in records_by_split]
    split_sizes = {s: len(records_by_split[s]) for s in splits}
    voice_pool_sizes = {s: len(VOICE_POOLS_BY_SPLIT[s]) for s in splits}
    targets = compute_split_targets(split_sizes, voice_pool_sizes, total)

    sorted_records = {s: sorted(records_by_split[s], key=lambda r: r.id) for s in splits}
    voice_of: dict[str, dict[str, str]] = {
        s: {r.id: assign_user_voice(r.id, s, global_seed) for r in sorted_records[s]} for s in splits
    }

    voice_to_ids: dict[str, dict[str, list[str]]] = {s: {} for s in splits}
    for s in splits:
        for r in sorted_records[s]:
            voice_to_ids[s].setdefault(voice_of[s][r.id], []).append(r.id)

    category_to_candidates: dict[str, list[tuple[str, str]]] = {}
    for s in splits:
        for r in sorted_records[s]:
            category_to_candidates.setdefault(r.category, []).append((s, r.id))

    selected: dict[str, set[str]] = {s: set() for s in splits}
    selected_order: dict[str, list[str]] = {s: [] for s in splits}

    def try_select(split: str, rid: str) -> bool:
        if rid in selected[split]:
            return True
        if len(selected[split]) >= targets[split]:
            return False
        selected[split].add(rid)
        selected_order[split].append(rid)
        return True

    covered_voices_by_split: dict[str, set[str]] = {s: set() for s in splits}
    missing_voices_by_split: dict[str, set[str]] = {s: set() for s in splits}
    for s in splits:
        for voice in VOICE_POOLS_BY_SPLIT[s]:
            candidates = voice_to_ids[s].get(voice, [])
            picked = False
            for rid in candidates:
                if try_select(s, rid):
                    picked = True
                    break
            if picked:
                covered_voices_by_split[s].add(voice)
            else:
                missing_voices_by_split[s].add(voice)

    all_categories = set(category_to_candidates.keys())
    covered_categories: set[str] = set()
    missing_categories: set[str] = set()
    for category in sorted(all_categories):
        already = False
        for s in splits:
            for rid in selected_order[s]:
                rec = next((r for r in sorted_records[s] if r.id == rid), None)
                if rec is not None and rec.category == category:
                    already = True
                    break
            if already:
                break
        if already:
            covered_categories.add(category)
            continue

        picked = False
        for split, rid in category_to_candidates[category]:
            if try_select(split, rid):
                picked = True
                break
        if picked:
            covered_categories.add(category)
        else:
            # Best-effort: borrow one slot from the fill phase by nudging the
            # target up, as long as the split has more source records available.
            for split, rid in category_to_candidates[category]:
                if len(selected[split]) < split_sizes[split]:
                    targets[split] += 1
                    if try_select(split, rid):
                        picked = True
                        break
            if picked:
                covered_categories.add(category)
            else:
                missing_categories.add(category)

    for s in splits:
        if len(selected[s]) >= targets[s]:
            continue
        for r in sorted_records[s]:
            if len(selected[s]) >= targets[s]:
                break
            try_select(s, r.id)

    return SmokeSelection(
        ids_by_split={s: sorted(selected_order[s]) for s in splits},
        covered_categories=covered_categories,
        missing_categories=missing_categories,
        covered_voices_by_split=covered_voices_by_split,
        missing_voices_by_split=missing_voices_by_split,
    )
