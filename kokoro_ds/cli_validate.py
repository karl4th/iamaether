"""CLI orchestrator for validate_generated_dataset.py.

No torch/kokoro dependency at all: validation only opens WAV/JSON files
already on disk and re-derives checksums/voice assignments, all of which are
pure-python/numpy operations.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import wav_io
from .alignment_schema import validate_alignment_list
from .audio_ops import all_finite, has_speech, is_clipping, peak_amplitude, rms_amplitude
from .checksums import source_dialogue_checksum
from .config import DEFAULT_CONFIG, GenerationConfig
from .determinism import assign_user_voice
from .manifest import atomic_write_json, read_json, read_jsonl
from .reporting import percentiles
from .schema import DEFAULT_ALLOWED_CATEGORIES, SourceRecord, load_split_file
from .voices_data import ASSISTANT_VOICE, VOICE_POOLS_BY_SPLIT

DEFAULT_SOURCE_ROOT = "/content/drive/MyDrive/deepseek_batches"
DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1"
DURATION_TOLERANCE_SECONDS = 0.01
MIN_PLAUSIBLE_DURATION_SECONDS = 0.3


@dataclass
class RecordCheck:
    record_id: str
    split: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a generated Kokoro Moshi fine-tuning dataset.")
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--source-root", type=Path, default=Path(DEFAULT_SOURCE_ROOT))
    p.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=None)
    return p.parse_args(argv)


def _discover_splits(output_root: Path, requested: list[str] | None) -> list[str]:
    candidates = requested or ["train", "validation", "test"]
    return [s for s in candidates if (output_root / s).exists()]


def _load_source_index(source_root: Path, splits: list[str]) -> dict[str, tuple[str, SourceRecord]]:
    index: dict[str, tuple[str, SourceRecord]] = {}
    for split in splits:
        path = source_root / f"{split}.jsonl"
        if not path.exists():
            continue
        records, _problems = load_split_file(path, allowed_categories=DEFAULT_ALLOWED_CATEGORIES)
        for r in records:
            index[r.id] = (split, r)
    return index


def check_one_record(
    split: str,
    dialogue_id: str,
    split_dir: Path,
    config: GenerationConfig,
    source_index: dict[str, tuple[str, SourceRecord]],
    manifest_index: dict[str, dict],
) -> RecordCheck:
    problems: list[str] = []
    stats: dict = {}
    wav_path = split_dir / "audio" / f"{dialogue_id}.wav"
    json_path = split_dir / "audio" / f"{dialogue_id}.json"

    if not wav_path.exists():
        return RecordCheck(dialogue_id, split, False, ["wav missing"])
    if not json_path.exists():
        return RecordCheck(dialogue_id, split, False, ["companion json missing"])

    try:
        info = wav_io.read_wav(wav_path)
    except Exception as e:  # noqa: BLE001
        return RecordCheck(dialogue_id, split, False, [f"wav failed to open: {e}"])

    if info.sample_rate != config.sample_rate:
        problems.append(f"sample_rate {info.sample_rate} != {config.sample_rate}")
    if info.channels != config.channels:
        problems.append(f"channels {info.channels} != {config.channels}")

    duration = info.n_frames / float(info.sample_rate) if info.sample_rate else 0.0
    stats["duration_seconds"] = duration
    if duration < MIN_PLAUSIBLE_DURATION_SECONDS:
        problems.append(f"implausibly short duration {duration:.3f}s")

    if info.n_frames > 0:
        if not all_finite(info.data):
            problems.append("non-finite audio samples")
        left, right = info.data[:, 0], info.data[:, 1] if info.channels > 1 else info.data[:, 0]
        stats["rms_left"], stats["rms_right"] = rms_amplitude(left), rms_amplitude(right)
        stats["peak_left"], stats["peak_right"] = peak_amplitude(left), peak_amplitude(right)
        if not has_speech(left):
            problems.append("left (assistant) channel has no detectable speech")
        if not has_speech(right):
            problems.append("right (user) channel has no detectable speech")
        if is_clipping(left, config.clipping_peak_threshold) or is_clipping(right, config.clipping_peak_threshold):
            problems.append("clipping detected")
    else:
        problems.append("wav has zero frames")

    try:
        companion = read_json(json_path)
    except Exception as e:  # noqa: BLE001
        return RecordCheck(dialogue_id, split, False, problems + [f"companion json failed to parse: {e}"])

    alignments = companion.get("alignments")
    if not alignments:
        problems.append("alignments missing or empty")
    else:
        problems.extend(validate_alignment_list(alignments, duration))
        confidence_info = companion.get("alignment_confidence", {})
        if confidence_info.get("mean") is not None:
            stats["alignment_confidence_mean"] = confidence_info["mean"]
        stats["flagged_low_confidence"] = bool(confidence_info.get("flagged_low_confidence", False))

    assistant_voice = companion.get("assistant_voice")
    if assistant_voice != ASSISTANT_VOICE:
        problems.append(f"assistant_voice {assistant_voice!r} != required {ASSISTANT_VOICE!r}")

    user_voice = companion.get("user_voice")
    stats["user_voice"] = user_voice
    pool = VOICE_POOLS_BY_SPLIT.get(split, ())
    if user_voice not in pool:
        problems.append(f"user_voice {user_voice!r} is not in the {split!r} pool")
    else:
        expected_voice = assign_user_voice(dialogue_id, split, companion.get("seed", 0))
        if expected_voice != user_voice:
            problems.append(
                f"user_voice {user_voice!r} does not match deterministic assignment {expected_voice!r} for seed {companion.get('seed')}"
            )

    source_entry = source_index.get(dialogue_id)
    if source_entry is None:
        problems.append("dialogue id not found in source split files")
    else:
        source_split, source_record = source_entry
        if source_split != split:
            problems.append(f"source split {source_split!r} != output split {split!r} (split overlap/mismatch)")
        if source_record.category != companion.get("category"):
            problems.append("category mismatch between source and companion json")
        expected_source_checksum = source_dialogue_checksum(
            {
                "id": source_record.id,
                "category": source_record.category,
                "turns": [{"speaker": t.speaker, "text": t.text} for t in source_record.turns],
            }
        )
        actual_source_checksum = companion.get("checksums", {}).get("source_dialogue")
        if actual_source_checksum != expected_source_checksum:
            problems.append("source_dialogue checksum mismatch between source data and companion json")

    manifest_entry = manifest_index.get(dialogue_id)
    if manifest_entry is None:
        problems.append("no corresponding entry in split manifest")
    elif abs(manifest_entry.get("duration", -1) - duration) > DURATION_TOLERANCE_SECONDS:
        problems.append(
            f"manifest duration {manifest_entry.get('duration')} does not match wav duration {duration:.3f}"
        )

    stats["category"] = companion.get("category")
    return RecordCheck(dialogue_id, split, len(problems) == 0, problems, stats)


def validate_dataset(output_root: Path, source_root: Path, splits: list[str] | None, config: GenerationConfig = DEFAULT_CONFIG) -> tuple[dict, int]:
    start = time.monotonic()
    active_splits = _discover_splits(output_root, splits)
    source_index = _load_source_index(source_root, active_splits)

    seen_ids: dict[str, str] = {}
    all_checks: list[RecordCheck] = []
    partial_files: list[str] = []
    per_split_report: dict[str, dict] = {}

    for split in active_splits:
        split_dir = output_root / split
        audio_dir = split_dir / "audio"
        tmp_files = list(audio_dir.glob(".*.tmp-*")) if audio_dir.exists() else []
        partial_files.extend(str(p) for p in tmp_files)

        manifest_path = split_dir / f"{split}.jsonl"
        manifest_index = {Path(m["path"]).stem: m for m in read_jsonl(manifest_path)}

        wav_ids = sorted(p.stem for p in audio_dir.glob("*.wav")) if audio_dir.exists() else []
        checks = []
        for dialogue_id in wav_ids:
            if dialogue_id in seen_ids:
                checks.append(RecordCheck(dialogue_id, split, False, [f"duplicate id also present in split {seen_ids[dialogue_id]!r}"]))
                continue
            seen_ids[dialogue_id] = split
            checks.append(check_one_record(split, dialogue_id, split_dir, config, source_index, manifest_index))

        all_checks.extend(checks)
        valid = [c for c in checks if c.ok]
        durations = [c.stats["duration_seconds"] for c in valid if "duration_seconds" in c.stats]
        per_split_report[split] = {
            "total_found": len(checks),
            "valid": len(valid),
            "invalid": len(checks) - len(valid),
            "total_hours": sum(durations) / 3600.0,
            "duration_percentiles": percentiles(durations),
            "voice_distribution": _voice_distribution(valid),
            "category_hours": _category_hours(valid),
        }

    rms_left = [c.stats["rms_left"] for c in all_checks if "rms_left" in c.stats]
    rms_right = [c.stats["rms_right"] for c in all_checks if "rms_right" in c.stats]
    peak_left = [c.stats["peak_left"] for c in all_checks if "peak_left" in c.stats]
    peak_right = [c.stats["peak_right"] for c in all_checks if "peak_right" in c.stats]
    conf = [c.stats["alignment_confidence_mean"] for c in all_checks if "alignment_confidence_mean" in c.stats]
    flagged_low_confidence = [c.record_id for c in all_checks if c.stats.get("flagged_low_confidence")]

    all_invalid = [c for c in all_checks if not c.ok]
    expected_total = len(source_index) if source_index else None
    completion_pct = (len(all_checks) - len(all_invalid)) / expected_total * 100.0 if expected_total else None

    report = {
        "splits_checked": active_splits,
        "total_files": len(all_checks),
        "total_valid": len(all_checks) - len(all_invalid),
        "total_invalid": len(all_invalid),
        "partially_written_files": partial_files,
        "per_split": per_split_report,
        "rms_percentiles": {"left": percentiles(rms_left), "right": percentiles(rms_right)},
        "peak_percentiles": {"left": percentiles(peak_left), "right": percentiles(peak_right)},
        "alignment_confidence_percentiles": percentiles(conf),
        "flagged_low_confidence_ids": flagged_low_confidence,
        "estimated_completion_percent": completion_pct,
        "invalid_records": [{"id": c.record_id, "split": c.split, "problems": c.problems} for c in all_invalid],
        "elapsed_seconds": time.monotonic() - start,
    }

    exit_code = 1 if (all_invalid or partial_files) else 0
    return report, exit_code


def _voice_distribution(checks: list[RecordCheck]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for c in checks:
        voice = c.stats.get("user_voice")
        if voice:
            dist[voice] = dist.get(voice, 0) + 1
    return dist


def _category_hours(checks: list[RecordCheck]) -> dict[str, float]:
    hours: dict[str, float] = {}
    for c in checks:
        category = c.stats.get("category") or "unknown"
        hours[category] = hours.get(category, 0.0) + c.stats.get("duration_seconds", 0.0) / 3600.0
    return hours


def _print_summary(report: dict) -> None:
    print("=== QA Summary ===")
    print(f"Splits checked: {report['splits_checked']}")
    print(f"Total files: {report['total_files']} (valid={report['total_valid']}, invalid={report['total_invalid']})")
    if report["estimated_completion_percent"] is not None:
        print(f"Estimated completion: {report['estimated_completion_percent']:.2f}%")
    if report["partially_written_files"]:
        print(f"WARNING: {len(report['partially_written_files'])} partially written file(s) found")
    if report["flagged_low_confidence_ids"]:
        print(f"WARNING: {len(report['flagged_low_confidence_ids'])} record(s) flagged for low alignment confidence")
    for split, s in report["per_split"].items():
        print(f"  [{split}] valid={s['valid']} invalid={s['invalid']} total_hours={s['total_hours']:.2f}")
    if report["total_invalid"]:
        print("First invalid records:")
        for entry in report["invalid_records"][:20]:
            print(f"  - {entry['id']} ({entry['split']}): {entry['problems']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, exit_code = validate_dataset(args.output_root, args.source_root, args.splits)
    atomic_write_json(args.output_root / "reports" / "qa_summary.json", report)
    _print_summary(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
