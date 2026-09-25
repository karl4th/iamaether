"""Source dialogue schema and validation.

``batch_0001.jsonl`` (shipped at the repo root) is only a schema/content
example. The authoritative data lives in Google Drive as
``train.jsonl`` / ``validation.jsonl`` / ``test.jsonl`` with expected counts
of 2700 / 150 / 150. The categories observed in the example batch are used
as the default allow-list (see ``DEFAULT_ALLOWED_CATEGORIES``); pass a
different set via ``allowed_categories`` if the full dataset introduces
categories not present in the example batch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import text_norm

EXPECTED_SPLIT_COUNTS: dict[str, int] = {
    "train": 2700,
    "validation": 150,
    "test": 150,
}

ALLOWED_SPEAKERS = ("user", "assistant")
MIN_TURNS = 2
MAX_TURNS = 8

# Derived from batch_0001.jsonl. See module docstring: this is an assumption,
# not a guarantee about the full 3,000-record dataset.
DEFAULT_ALLOWED_CATEGORIES = frozenset(
    {
        "general_conversation",
        "identity_facts",
        "identity_variations",
        "legacy_identity_corrections",
        "russian_language_status",
        "conversation_control",
    }
)


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class SourceRecord:
    id: str
    category: str
    turns: tuple[Turn, ...]
    line_no: int


class ValidationError(Exception):
    """Raised with a list of actionable (record_id, reason) problems."""

    def __init__(self, problems: list[tuple[str, str]]):
        self.problems = problems
        summary = "\n".join(f"  - {rid}: {reason}" for rid, reason in problems[:200])
        more = "" if len(problems) <= 200 else f"\n  ... and {len(problems) - 200} more"
        super().__init__(f"{len(problems)} source validation problem(s):\n{summary}{more}")


def _validate_record_dict(raw: dict, line_no: int, allowed_categories: set[str]) -> tuple[SourceRecord | None, list[str]]:
    problems: list[str] = []

    if set(raw.keys()) != {"id", "category", "turns"}:
        extra = set(raw.keys()) - {"id", "category", "turns"}
        missing = {"id", "category", "turns"} - set(raw.keys())
        problems.append(f"unexpected keys (extra={sorted(extra)}, missing={sorted(missing)})")
        return None, problems

    rid = raw.get("id")
    category = raw.get("category")
    turns_raw = raw.get("turns")

    if not isinstance(rid, str) or not rid.strip():
        problems.append("id must be a nonempty string")
        return None, problems
    if not isinstance(category, str) or not category.strip():
        problems.append("category must be a nonempty string")
    elif category not in allowed_categories:
        problems.append(f"category {category!r} is not in the allowed category set")
    if not isinstance(turns_raw, list) or not turns_raw:
        problems.append("turns must be a nonempty list")
        return None, problems

    if len(turns_raw) < MIN_TURNS or len(turns_raw) > MAX_TURNS or len(turns_raw) % 2 != 0:
        problems.append(f"turns must have an even count between {MIN_TURNS} and {MAX_TURNS}, got {len(turns_raw)}")

    turns: list[Turn] = []
    for i, t in enumerate(turns_raw):
        if not isinstance(t, dict) or set(t.keys()) != {"speaker", "text"}:
            problems.append(f"turn[{i}] must have exactly keys 'speaker' and 'text'")
            continue
        speaker = t.get("speaker")
        text = t.get("text")
        if speaker not in ALLOWED_SPEAKERS:
            problems.append(f"turn[{i}] has invalid speaker {speaker!r}")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"turn[{i}] text must be nonempty")
            continue
        if text_norm.has_control_characters(text):
            problems.append(f"turn[{i}] contains control characters")
        if text_norm.has_unsupported_unicode(text):
            problems.append(f"turn[{i}] contains unsupported unicode characters")
        if text_norm.has_url(text):
            problems.append(f"turn[{i}] contains a URL")
        if text_norm.has_markdown(text):
            problems.append(f"turn[{i}] contains markdown syntax")
        if text_norm.is_excessively_long(text):
            problems.append(f"turn[{i}] exceeds max utterance length ({text_norm.MAX_UTTERANCE_CHARS} chars)")
        if isinstance(speaker, str) and speaker == "assistant" and not text_norm.tokenize_words(text):
            problems.append(f"turn[{i}] normalizes to an empty forced-alignment transcript")
        turns.append(Turn(speaker=speaker if isinstance(speaker, str) else "", text=text))

    if problems:
        return None, problems

    if turns[0].speaker != "user":
        problems.append("first speaker must be 'user'")
    if turns[-1].speaker != "assistant":
        problems.append("final speaker must be 'assistant'")
    for i in range(1, len(turns)):
        if turns[i].speaker == turns[i - 1].speaker:
            problems.append(f"turn[{i}] does not alternate speaker with turn[{i - 1}]")
            break

    if problems:
        return None, problems

    return SourceRecord(id=rid, category=category, turns=tuple(turns), line_no=line_no), []


def load_split_file(path: Path, allowed_categories: Iterable[str] | None = None) -> tuple[list[SourceRecord], list[tuple[str, str]]]:
    """Parse and validate one split JSONL file.

    Returns ``(records, problems)`` where ``problems`` is a list of
    ``(record_id_or_line_ref, reason)`` tuples. Parsing continues past
    individual bad lines so all problems in a file are surfaced together.
    """
    allowed = set(allowed_categories) if allowed_categories is not None else set(DEFAULT_ALLOWED_CATEGORIES)
    records: list[SourceRecord] = []
    problems: list[tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            ref = f"{path.name}:L{line_no}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append((ref, f"invalid JSON: {e}"))
                continue
            if not isinstance(raw, dict):
                problems.append((ref, "line is not a JSON object"))
                continue
            record, record_problems = _validate_record_dict(raw, line_no, allowed)
            rid = raw.get("id") if isinstance(raw.get("id"), str) else ref
            for reason in record_problems:
                problems.append((rid, reason))
            if record is not None:
                records.append(record)

    return records, problems


def validate_source_tree(
    source_root: Path,
    splits: Iterable[str] = ("train", "validation", "test"),
    allowed_categories: Iterable[str] | None = None,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, list[SourceRecord]]:
    """Full cross-split validation. Raises ``ValidationError`` on any problem.

    ``expected_counts=None`` (the default) checks against the real dataset's
    known split sizes; pass ``{}`` explicitly to skip count checking entirely
    (used for smoke-test / single-record runs against a smaller source tree).
    """
    if expected_counts is None:
        expected_counts = EXPECTED_SPLIT_COUNTS
    problems: list[tuple[str, str]] = []
    by_split: dict[str, list[SourceRecord]] = {}

    for split in splits:
        path = source_root / f"{split}.jsonl"
        if not path.exists():
            problems.append((split, f"missing authoritative split file: {path}"))
            continue
        records, file_problems = load_split_file(path, allowed_categories)
        problems.extend(file_problems)
        by_split[split] = records

        expected = expected_counts.get(split)
        if expected is not None and len(records) != expected:
            problems.append((split, f"expected {expected} records, found {len(records)}"))

    seen_ids: dict[str, str] = {}
    for split, records in by_split.items():
        local_ids = set()
        for r in records:
            if r.id in local_ids:
                problems.append((r.id, f"duplicate id within {split}"))
            local_ids.add(r.id)
            if r.id in seen_ids and seen_ids[r.id] != split:
                problems.append((r.id, f"id appears in both {seen_ids[r.id]!r} and {split!r} (split overlap)"))
            else:
                seen_ids[r.id] = split

    if problems:
        raise ValidationError(problems)

    return by_split
