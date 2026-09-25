"""CLI orchestrator for generate_kokoro_dataset.py.

Heavy ML imports (torch / kokoro / torchaudio) happen lazily inside
``_build_backends`` so ``--validate-only`` and ``--dry-run`` work in a plain
CPU environment, and so this module stays importable from tests without
those packages installed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import env_info
from .config import DEFAULT_CONFIG, GenerationConfig
from .determinism import assign_user_voice
from .manifest import atomic_write_json, manifest_line, write_split_manifest
from .pronunciation import PRONUNCIATION_MAP, print_pronunciation_map
from .receipts import ReceiptStore, verify_existing_record
from .reporting import ProgressTracker
from .schema import SourceRecord, ValidationError, validate_source_tree
from .smoke import select_stratified_smoke
from .voices_data import assert_voice_pools_valid

DEFAULT_SOURCE_ROOT = "/content/drive/MyDrive/deepseek_batches"
DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1"

ESTIMATED_CHARS_PER_SECOND = 15.0
"""Rough English TTS speaking rate at speed=1.0, used only for --dry-run estimates."""

BYTES_PER_SECOND_STEREO_PCM16 = 24_000 * 2 * 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a Kokoro-voiced Moshi fine-tuning dataset from JSONL dialogues.")
    p.add_argument("--source-root", type=Path, default=Path(DEFAULT_SOURCE_ROOT))
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation", "test"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-count", type=int, default=None, help="Generate a deterministic stratified subset of this size instead of the full split(s).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-intermediate", action="store_true", help="Also write per-turn debug WAVs under audio/_debug/.")
    p.add_argument("--record-id", type=str, default=None, help="Generate/regenerate exactly one dialogue id, for debugging.")
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def _note_ignored_raw_batches(source_root: Path) -> None:
    authoritative = all((source_root / f"{s}.jsonl").exists() for s in ("train", "validation", "test"))
    raw_batches = sorted(source_root.glob("batch_*.jsonl"))
    if authoritative and raw_batches:
        print(
            f"Note: {len(raw_batches)} raw batch_*.jsonl file(s) found in {source_root} but ignored, "
            "because the authoritative train/validation/test.jsonl split files are present."
        )


def _load_and_validate_source(args: argparse.Namespace) -> dict[str, list[SourceRecord]]:
    expected = {} if (args.record_id or args.smoke_count) else None
    try:
        by_split = validate_source_tree(args.source_root, splits=args.splits, expected_counts=expected)
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        raise
    total = sum(len(v) for v in by_split.values())
    print(f"Source validation passed: {total} dialogues across splits {list(by_split.keys())}.")
    return by_split


def _select_records(args: argparse.Namespace, by_split: dict[str, list[SourceRecord]]) -> dict[str, list[SourceRecord]]:
    if args.record_id:
        for split, records in by_split.items():
            match = next((r for r in records if r.id == args.record_id), None)
            if match:
                print(f"--record-id {args.record_id!r} found in split {split!r}.")
                return {split: [match]}
        raise SystemExit(f"--record-id {args.record_id!r} not found in any loaded split.")

    if args.smoke_count:
        selection = select_stratified_smoke(by_split, global_seed=args.seed, total=args.smoke_count)
        print(
            f"Stratified smoke selection: {selection.total()} dialogues "
            f"({', '.join(f'{s}={len(ids)}' for s, ids in selection.ids_by_split.items())})"
        )
        if selection.missing_categories:
            print(f"WARNING: could not cover categories: {sorted(selection.missing_categories)}")
        for split, missing in selection.missing_voices_by_split.items():
            if missing:
                print(f"WARNING: could not cover {split} user voices: {sorted(missing)}")
        result: dict[str, list[SourceRecord]] = {}
        for split, ids in selection.ids_by_split.items():
            id_set = set(ids)
            result[split] = [r for r in by_split[split] if r.id in id_set]
        return result

    return by_split


def _dry_run_estimate(records_by_split: dict[str, list[SourceRecord]]) -> None:
    total_chars = 0
    total_dialogues = 0
    for records in records_by_split.values():
        for r in records:
            total_dialogues += 1
            total_chars += sum(len(t.text) for t in r.turns)

    estimated_speech_seconds = total_chars / ESTIMATED_CHARS_PER_SECOND
    n_pauses_estimate = sum(max(0, len(r.turns) - 1) for records in records_by_split.values() for r in records)
    estimated_pause_seconds = n_pauses_estimate * 0.5
    estimated_total_seconds = estimated_speech_seconds + estimated_pause_seconds
    estimated_bytes = estimated_total_seconds * BYTES_PER_SECOND_STEREO_PCM16

    print("Dry-run estimate (rough, no synthesis performed):")
    print(f"  dialogues: {total_dialogues}")
    print(f"  estimated audio duration: {estimated_total_seconds / 3600.0:.2f} hours")
    print(f"  estimated disk usage: {estimated_bytes / (1024 ** 3):.2f} GiB (WAV, PCM16, 24kHz, stereo)")


def _build_backends(device: str):
    from .kokoro_backend import KokoroBackend
    from .alignment_engine import ForcedAligner

    backend = KokoroBackend(device=device)
    aligner = ForcedAligner(device=device)
    return backend, aligner


def _assert_runtime_ready() -> None:
    assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version_info[:2]}"
    import torch

    assert torch.cuda.is_available(), "CUDA is not available; run on a GPU (A100) Colab runtime."


def run_generation(
    args: argparse.Namespace,
    records_by_split: dict[str, list[SourceRecord]],
    config: GenerationConfig = DEFAULT_CONFIG,
) -> int:
    from .kokoro_backend import assert_all_voices_available
    from .pipeline import generate_one_record

    _assert_runtime_ready()
    print_pronunciation_map(PRONUNCIATION_MAP)

    backend, aligner = _build_backends(args.device)
    assert_all_voices_available(backend)
    environment = env_info.collect_environment(backend.model_identity(), aligner.model_identity())

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "reports" / "provenance.json", {"seed": args.seed, **environment})

    had_unresolved_failure = False
    grand_total = sum(len(v) for v in records_by_split.values())
    tracker = ProgressTracker(total=grand_total)
    split_summaries: dict[str, dict] = {}

    for split, records in records_by_split.items():
        split_dir = args.output_root / split
        receipt_store = ReceiptStore(args.output_root / "receipts")
        manifest_lines: dict[str, dict] = {}

        for i, record in enumerate(records):
            already_good = False
            if args.resume and receipt_store.is_completed(record.id):
                user_voice = assign_user_voice(record.id, split, args.seed)
                result = verify_existing_record(split_dir, record, split, user_voice, config, receipt_store)
                if result.ok:
                    already_good = True
                    receipt = receipt_store.get_completed(record.id)
                    tracker.record_success(receipt["duration_seconds"])
                    manifest_lines[record.id] = manifest_line(f"audio/{record.id}.wav", receipt["duration_seconds"])
                else:
                    print(f"Resume check failed for {record.id}, regenerating: {result.reasons}")

            if not already_good:
                outcome = generate_one_record(
                    record=record,
                    split=split,
                    split_dir=split_dir,
                    config=config,
                    global_seed=args.seed,
                    synth_fn=backend.synthesize,
                    align_fn=aligner.align,
                    receipt_store=receipt_store,
                    environment=environment,
                    pronunciation_map=PRONUNCIATION_MAP,
                    keep_intermediate=args.keep_intermediate,
                )
                if outcome.success:
                    tracker.record_success(outcome.duration_seconds)
                    manifest_lines[record.id] = manifest_line(outcome.relative_wav_path, outcome.duration_seconds)
                else:
                    tracker.record_failure()
                    had_unresolved_failure = True
                    receipt_store.mark_failed(
                        {"id": record.id, "split": split, "reasons": outcome.reasons, "timestamp": time.time()}
                    )
                    print(f"FAILED {record.id}: {outcome.reasons}")

            if (i + 1) % max(1, args.progress_every) == 0 or (i + 1) == len(records):
                gpu_mem = env_info.gpu_memory_mb()
                print(tracker.line(split, record.id, gpu_mem))
                ordered = [manifest_lines[r.id] for r in records if r.id in manifest_lines]
                write_split_manifest(split_dir / f"{split}.jsonl", ordered)

        ordered = [manifest_lines[r.id] for r in records if r.id in manifest_lines]
        write_split_manifest(split_dir / f"{split}.jsonl", ordered)
        split_summaries[split] = {
            "total": len(records),
            "completed": len(manifest_lines),
            "failed": len(records) - len(manifest_lines),
        }

    generation_summary = {
        "elapsed_seconds": tracker.elapsed_seconds(),
        "completed": tracker.completed,
        "failed": tracker.failed,
        "audio_hours_generated": tracker.audio_seconds_generated / 3600.0,
        "by_split": split_summaries,
        "seed": args.seed,
    }
    atomic_write_json(args.output_root / "reports" / "generation_summary.json", generation_summary)
    print(f"Generation complete: {generation_summary}")

    return 1 if had_unresolved_failure else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    assert_voice_pools_valid()
    _note_ignored_raw_batches(args.source_root)

    try:
        by_split = _load_and_validate_source(args)
    except ValidationError:
        return 1

    if args.validate_only:
        return 0

    selected = _select_records(args, by_split)

    if args.dry_run:
        _dry_run_estimate(selected)
        return 0

    return run_generation(args, selected)


if __name__ == "__main__":
    raise SystemExit(main())
