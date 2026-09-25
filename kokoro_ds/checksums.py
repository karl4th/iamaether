"""Stable SHA-256 checksums used for resume/integrity verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Checksum of the canonical (sorted-key, compact) JSON encoding of ``obj``."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(canonical)


def source_dialogue_checksum(record: dict) -> str:
    """Checksum over the fields that, if changed, must invalidate generated audio."""
    payload = {
        "id": record["id"],
        "category": record["category"],
        "turns": [{"speaker": t["speaker"], "text": t["text"]} for t in record["turns"]],
    }
    return sha256_json(payload)


def generation_config_checksum(config_dict: dict, split: str, user_voice: str) -> str:
    payload = {"config": config_dict, "split": split, "user_voice": user_voice}
    return sha256_json(payload)
