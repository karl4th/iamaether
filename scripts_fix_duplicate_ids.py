"""One-off repair: rename duplicate ids in train.jsonl so every id is unique.

Usage:
    uv run scripts_fix_duplicate_ids.py /content/drive/MyDrive/deepseek_batches/train.jsonl

Writes a sibling file with .fixed.jsonl suffix. Does not touch the original.
"""
import json
import sys
from pathlib import Path


def fix(path: Path) -> Path:
    seen: dict[str, int] = {}
    out_path = path.with_suffix(".fixed.jsonl")
    with open(path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec["id"]
            seen[rid] = seen.get(rid, 0) + 1
            if seen[rid] > 1:
                new_id = f"{rid}__dup{seen[rid]}"
                print(f"renaming duplicate {rid!r} (occurrence {seen[rid]}) -> {new_id!r}")
                rec["id"] = new_id
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = fix(src)
    print(f"\nWrote {out}")
    print(f"Now run:  mv {out} {src}")
