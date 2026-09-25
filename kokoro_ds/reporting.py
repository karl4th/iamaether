"""Small aggregation helpers shared by the generator's progress printer and the validator's report builder."""
from __future__ import annotations

import math
import time


def percentiles(values: list[float], ps: tuple[float, ...] = (0, 25, 50, 75, 90, 95, 99, 100)) -> dict[str, float]:
    if not values:
        return {f"p{int(p)}": float("nan") for p in ps}
    ordered = sorted(values)
    n = len(ordered)

    def _pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        rank = (p / 100.0) * (n - 1)
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return ordered[int(rank)]
        frac = rank - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {f"p{int(p)}": round(_pct(p), 4) for p in ps}


def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressTracker:
    """Tracks throughput for the periodic progress line the generator prints."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.audio_seconds_generated = 0.0
        self.start_time = time.monotonic()

    def record_success(self, duration_seconds: float) -> None:
        self.completed += 1
        self.audio_seconds_generated += duration_seconds

    def record_failure(self) -> None:
        self.failed += 1

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def files_per_minute(self) -> float:
        elapsed = self.elapsed_seconds()
        if elapsed <= 0:
            return 0.0
        return (self.completed / elapsed) * 60.0

    def eta_seconds(self) -> float:
        rate = self.files_per_minute()
        remaining = max(0, self.total - self.completed - self.failed)
        if rate <= 0:
            return float("inf")
        return (remaining / rate) * 60.0

    def line(self, split: str, current_record_id: str, gpu_memory_mb: float | None = None) -> str:
        parts = [
            f"[{split}] {self.completed + self.failed}/{self.total}",
            f"record={current_record_id}",
            f"elapsed={format_hms(self.elapsed_seconds())}",
            f"rate={self.files_per_minute():.1f}/min",
            f"audio={self.audio_seconds_generated / 3600.0:.2f}h",
            f"eta={format_hms(self.eta_seconds()) if math.isfinite(self.eta_seconds()) else 'n/a'}",
            f"failed={self.failed}",
        ]
        if gpu_memory_mb is not None:
            parts.append(f"gpu_mem={gpu_memory_mb:.0f}MB")
        return " | ".join(parts)
