"""Colab diagnostic cell: inspect the 4 source validation failures.

Paste this whole file's contents into a Colab cell (adjust SOURCE_ROOT /
REPO_ROOT if different), or run it as a script inside the venv:

    .venv/bin/python scripts_inspect_source_issues.py
"""
import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = "/content/iamaether"  # where this repo is checked out in Colab
SOURCE_ROOT = "/content/drive/MyDrive/deepseek_batches"

sys.path.insert(0, REPO_ROOT)
from kokoro_ds.text_norm import _ALLOWED_EXTRA_UNICODE  # noqa: E402

UNICODE_IDS = {"batch_0015_0020", "batch_0012_0027"}
DUPLICATE_IDS = {"batch_0023_0202", "batch_0021_0201"}


def bad_chars(text: str):
    return [
        (ch, f"U+{ord(ch):04X}", unicodedata.name(ch, "UNKNOWN"))
        for ch in text
        if not (0x20 <= ord(ch) <= 0x7E) and ch not in _ALLOWED_EXTRA_UNICODE
    ]


def inspect_unicode(path: Path, target_ids: set[str]) -> None:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] not in target_ids:
                continue
            print(f"\n=== {rec['id']} (line {line_no}) ===")
            for i, t in enumerate(rec["turns"]):
                problems = bad_chars(t["text"])
                marker = f"  <-- BAD: {problems}" if problems else ""
                print(f"  turn[{i}] {t['speaker']:9s}: {t['text']!r}{marker}")


def inspect_duplicates(path: Path, target_ids: set[str]) -> None:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] in target_ids:
                print(f"\n=== line {line_no}: {rec['id']} ===")
                print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    train_path = Path(SOURCE_ROOT) / "train.jsonl"

    print("############ UNICODE ISSUES ############")
    inspect_unicode(train_path, UNICODE_IDS)

    print("\n\n############ DUPLICATE IDS ############")
    inspect_duplicates(train_path, DUPLICATE_IDS)
