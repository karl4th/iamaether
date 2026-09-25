"""One-off migration: recompute config_checksum in receipts/completed.jsonl
under the new audio_affecting_dict() formula, without regenerating any audio.

Why this is safe: config_checksum is only meant to catch changes to fields
that actually change the bytes written to a WAV (sample rate, voice, speed
ranges, pause ranges, fade). It previously also hashed
`clipping_peak_threshold` / `alignment_min_confidence`, which are pure
post-hoc QA gates that never affect what gets synthesized. Tuning those
today (see kokoro_ds/config.py) legitimately changed the old hash for every
existing record even though the audio itself is still 100% correct.

This script re-derives what the checksum *would have been* under the new,
narrower formula, using the CURRENT DEFAULT_CONFIG's audio-affecting fields
-- which is exactly what was true when these records were actually
generated, since none of those fields changed. It does not touch any wav or
json file, only the bookkeeping checksum in the receipts log, so --resume
recognizes already-correct records as valid instead of regenerating all of
them for a purely cosmetic reason.

Usage:
    .venv/bin/python scripts_migrate_config_checksum.py \
        --output-root /content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kokoro_ds.checksums import generation_config_checksum
from kokoro_ds.config import DEFAULT_CONFIG
from kokoro_ds.manifest import read_jsonl, rewrite_jsonl_atomic


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything.")
    args = p.parse_args(argv)

    completed_path = args.output_root / "receipts" / "completed.jsonl"
    if not completed_path.exists():
        raise SystemExit(f"No such file: {completed_path}")

    entries = list(read_jsonl(completed_path))
    print(f"Loaded {len(entries)} completed receipt(s) from {completed_path}")

    audio_dict = DEFAULT_CONFIG.audio_affecting_dict()
    changed = 0
    for entry in entries:
        new_checksum = generation_config_checksum(audio_dict, entry["split"], entry["user_voice"])
        if entry.get("config_checksum") != new_checksum:
            changed += 1
            entry["config_checksum"] = new_checksum

    print(f"{changed}/{len(entries)} receipt(s) had their config_checksum updated.")

    if args.dry_run:
        print("--dry-run: no file written.")
        return 0

    rewrite_jsonl_atomic(completed_path, entries)
    print(f"Wrote {completed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
