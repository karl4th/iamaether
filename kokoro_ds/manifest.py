"""Atomic JSON/JSONL writers and manifest line helpers.

Split manifests and receipts (``completed.jsonl`` / ``failed.jsonl``) are
append-only logs: every append is a single ``write`` + ``flush`` + ``fsync``
so a Colab interruption can lose at most the in-flight record, never a
previously committed line. Whole-file JSON documents (companion metadata,
reports) are written to a temp file and ``os.replace``d into place so a
reader never observes a partially written file.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Iterator

import orjson


def atomic_write_json(path: Path, obj: Any, indent: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    option = orjson.OPT_INDENT_2 if indent else 0
    data = orjson.dumps(obj, option=option)
    tmp_path = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def append_jsonl_line_durable(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = orjson.dumps(obj) + b"\n"
    with open(path, "ab") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, "rb") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            yield orjson.loads(raw_line)


def rewrite_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Fully rewrite a jsonl file (used to compact receipts, e.g. drop a stale entry before regenerating)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"".join(orjson.dumps(r) + b"\n" for r in records)
    tmp_path = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def manifest_line(relative_path: str, duration_seconds: float) -> dict:
    return {"path": relative_path, "duration": round(float(duration_seconds), 3)}


def write_split_manifest(path: Path, ordered_lines: list[dict]) -> None:
    """Fully (re)write a split manifest (``train/train.jsonl`` etc.) from scratch.

    Rewriting wholesale (instead of incremental append) keeps the manifest
    free of duplicate or stale entries across resumed runs; call this after
    every batch of newly completed/reverified records.
    """
    rewrite_jsonl_atomic(path, ordered_lines)
