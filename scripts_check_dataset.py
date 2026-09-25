"""Colab cell: authoritative correctness check of the generated dataset.

Runs the same validator as validate_generated_dataset.py, but calls it
in-process so we get the full report dict back for a readable verdict
instead of just an exit code. Does not require torch/kokoro/GPU.

Usage (Colab):
    exec(open("/content/iamaether/scripts_check_dataset.py").read())
"""
import sys

REPO_ROOT = "/content/iamaether"
OUTPUT_ROOT = "/content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1"
SOURCE_ROOT = "/content/drive/MyDrive/deepseek_batches"
EXPECTED_TOTAL = {"train": 2700, "validation": 150, "test": 150}

sys.path.insert(0, REPO_ROOT)

from pathlib import Path  # noqa: E402

from kokoro_ds.cli_validate import validate_dataset  # noqa: E402
from kokoro_ds.voices_data import ASSISTANT_VOICE, VOICE_POOLS_BY_SPLIT  # noqa: E402

report, exit_code = validate_dataset(Path(OUTPUT_ROOT), Path(SOURCE_ROOT), splits=None)

print("=" * 72)
print("DATASET CHECK:", "PASS ✅" if exit_code == 0 else "FAIL ❌")
print("=" * 72)

print(f"\nSplits checked: {report['splits_checked']}")
print(f"Total files: {report['total_files']}  (valid={report['total_valid']}, invalid={report['total_invalid']})")

print("\n-- Per split --")
all_counts_match = True
for split, s in report["per_split"].items():
    expected = EXPECTED_TOTAL.get(split)
    count_ok = expected is None or s["total_found"] == expected
    all_counts_match &= count_ok
    flag = "" if count_ok else f"  <-- expected {expected}!"
    print(f"  [{split}] found={s['total_found']} valid={s['valid']} invalid={s['invalid']} "
          f"hours={s['total_hours']:.2f}{flag}")

print("\n-- Voice pool sanity --")
pool_ok = True
for split, s in report["per_split"].items():
    used_voices = set(s["voice_distribution"].keys())
    expected_pool = set(VOICE_POOLS_BY_SPLIT.get(split, ()))
    leaked = used_voices - expected_pool
    missing = expected_pool - used_voices
    if leaked:
        pool_ok = False
        print(f"  [{split}] WRONG voices used (leaked from another split's pool!): {sorted(leaked)}")
    if missing:
        print(f"  [{split}] voices in pool never used (not necessarily a problem): {sorted(missing)}")
    if not leaked:
        print(f"  [{split}] all {len(used_voices)} used voices correctly confined to this split's pool")

print(f"\n-- Assistant voice fixed to {ASSISTANT_VOICE!r} everywhere: "
      f"{'YES' if not any('assistant_voice' in p for e in report['invalid_records'] for p in e['problems']) else 'NO - see invalid_records below'}")

if report["partially_written_files"]:
    print(f"\n⚠️  {len(report['partially_written_files'])} partially-written file(s) found:")
    for p in report["partially_written_files"][:10]:
        print(f"    {p}")

if report["flagged_low_confidence_ids"]:
    print(f"\n⚠️  {len(report['flagged_low_confidence_ids'])} record(s) flagged for low forced-alignment confidence "
          f"(structurally valid, but worth spot-checking by ear):")
    for rid in report["flagged_low_confidence_ids"][:10]:
        print(f"    {rid}")

print("\n-- Duration / RMS / peak / alignment-confidence percentiles --")
print(f"  duration (train): {report['per_split'].get('train', {}).get('duration_percentiles')}")
print(f"  rms   left/right: {report['rms_percentiles']['left']} / {report['rms_percentiles']['right']}")
print(f"  peak  left/right: {report['peak_percentiles']['left']} / {report['peak_percentiles']['right']}")
print(f"  alignment confidence: {report['alignment_confidence_percentiles']}")

if report["total_invalid"]:
    print(f"\n-- First {min(20, report['total_invalid'])} invalid record(s) --")
    for entry in report["invalid_records"][:20]:
        print(f"  {entry['id']} ({entry['split']}): {entry['problems']}")

print("\n" + "=" * 72)
if exit_code == 0 and all_counts_match and pool_ok:
    print("VERDICT: dataset is structurally correct and ready to publish.")
elif exit_code == 0:
    print("VERDICT: no invalid records, but check the warnings above (counts/voice pools) before publishing.")
else:
    print(f"VERDICT: NOT ready - {report['total_invalid']} invalid record(s). Fix and rerun generation before publishing.")
print("=" * 72)
