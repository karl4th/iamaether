"""Completed/failed receipt tracking and resume-time integrity verification.

``file exists`` is never treated as ``done``: :func:`verify_existing_record`
re-opens the WAV, re-parses the companion JSON, re-validates the alignment
list, and cross-checks the receipt's stored source/config checksums against
freshly computed ones before a record is skipped on resume.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import wav_io
from .alignment_schema import validate_alignment_list
from .audio_ops import all_finite, has_speech, is_clipping
from .checksums import generation_config_checksum, source_dialogue_checksum
from .config import GenerationConfig
from .manifest import append_jsonl_line_durable, read_jsonl
from .schema import SourceRecord


@dataclass
class VerificationResult:
    ok: bool
    reasons: list[str]


class ReceiptStore:
    """In-memory index over ``receipts/completed.jsonl`` and ``receipts/failed.jsonl``."""

    def __init__(self, receipts_dir: Path):
        self.receipts_dir = receipts_dir
        self.completed_path = receipts_dir / "completed.jsonl"
        self.failed_path = receipts_dir / "failed.jsonl"
        self._completed: dict[str, dict] = {}
        self._reload()

    def _reload(self) -> None:
        self._completed = {}
        for entry in read_jsonl(self.completed_path):
            rid = entry.get("id")
            if rid:
                self._completed[rid] = entry

    def get_completed(self, record_id: str) -> dict | None:
        return self._completed.get(record_id)

    def is_completed(self, record_id: str) -> bool:
        return record_id in self._completed

    def mark_completed(self, receipt: dict) -> None:
        append_jsonl_line_durable(self.completed_path, receipt)
        self._completed[receipt["id"]] = receipt

    def mark_failed(self, entry: dict) -> None:
        append_jsonl_line_durable(self.failed_path, entry)

    def failed_ids(self) -> set[str]:
        return {e.get("id") for e in read_jsonl(self.failed_path) if e.get("id")}


def wav_path_for(split_dir: Path, dialogue_id: str) -> Path:
    return split_dir / "audio" / f"{dialogue_id}.wav"


def json_path_for(split_dir: Path, dialogue_id: str) -> Path:
    return split_dir / "audio" / f"{dialogue_id}.json"


def verify_existing_record(
    split_dir: Path,
    record: SourceRecord,
    split: str,
    user_voice: str,
    config: GenerationConfig,
    receipt_store: ReceiptStore,
) -> VerificationResult:
    reasons: list[str] = []
    wav_path = wav_path_for(split_dir, record.id)
    json_path = json_path_for(split_dir, record.id)

    if not wav_path.exists():
        return VerificationResult(False, ["wav missing"])
    if not json_path.exists():
        return VerificationResult(False, ["companion json missing"])

    try:
        info = wav_io.read_wav(wav_path)
    except Exception as e:  # noqa: BLE001 - any decode failure means "not usable"
        return VerificationResult(False, [f"wav failed to open: {e}"])

    if info.sample_rate != config.sample_rate:
        reasons.append(f"wav sample_rate {info.sample_rate} != {config.sample_rate}")
    if info.channels != config.channels:
        reasons.append(f"wav channels {info.channels} != {config.channels}")
    if info.n_frames == 0:
        reasons.append("wav has zero samples")
    else:
        if not all_finite(info.data):
            reasons.append("wav contains non-finite samples")
        left = info.data[:, 0]
        right = info.data[:, 1] if info.channels > 1 else info.data[:, 0]
        if not has_speech(left):
            reasons.append("left (assistant) channel has no detectable speech")
        if not has_speech(right):
            reasons.append("right (user) channel has no detectable speech")
        if is_clipping(left, config.clipping_peak_threshold) or is_clipping(right, config.clipping_peak_threshold):
            reasons.append("wav peak at/above clipping threshold")

    try:
        from .manifest import read_json

        companion = read_json(json_path)
    except Exception as e:  # noqa: BLE001
        return VerificationResult(False, [f"companion json failed to parse: {e}"])

    alignments = companion.get("alignments")
    duration_seconds = info.n_frames / float(info.sample_rate) if info.sample_rate else 0.0
    if not alignments:
        reasons.append("alignments missing or empty")
    else:
        align_problems = validate_alignment_list(alignments, duration_seconds)
        reasons.extend(align_problems)

    receipt = receipt_store.get_completed(record.id)
    if receipt is None:
        reasons.append("no completed receipt found")
    else:
        expected_source = source_dialogue_checksum(
            {"id": record.id, "category": record.category, "turns": [{"speaker": t.speaker, "text": t.text} for t in record.turns]}
        )
        expected_config = generation_config_checksum(config.audio_affecting_dict(), split, user_voice)
        if receipt.get("source_checksum") != expected_source:
            reasons.append("receipt source_checksum does not match current source record")
        if receipt.get("config_checksum") != expected_config:
            reasons.append("receipt config_checksum does not match current generation config")

    return VerificationResult(ok=len(reasons) == 0, reasons=reasons)
