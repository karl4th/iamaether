"""Multi-process parallel generation.

The real bottleneck for this workload is not GPU compute (an A100 sits at a
few percent utilization synthesizing a handful of seconds of speech at a
time) -- it's the fixed per-call Python/host<->device sync latency, paid
once per utterance, thousands of times. Running several worker processes
concurrently overlaps that latency across processes instead of paying it
strictly one call at a time, which is where the actual speedup comes from.

Uses the 'spawn' start method (never fork): CUDA contexts cannot be safely
forked, so each worker builds its own Kokoro + forced-alignment model
instances from scratch. That costs a few GB of VRAM per worker, which is
cheap headroom on an A100 that is otherwise nearly idle for this workload.

Correctness invariants preserved under parallelism:
- Work is sharded by dialogue id up front, in the parent, before any worker
  starts -- no two workers ever touch the same dialogue id.
- Workers only write per-record files (wav/json) and append to
  receipts/completed.jsonl or receipts/failed.jsonl. Concurrent append is
  safe: each line is written with one write() syscall well under PIPE_BUF,
  which POSIX guarantees is atomic for O_APPEND writers, so lines from
  different processes never interleave or corrupt each other.
- Workers never write the split manifest. It is rebuilt exactly once, by the
  parent, after every worker has exited -- a single writer reading the now-
  complete receipts, so it can never race another writer or observe a
  partial view of another worker's progress.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

from .config import GenerationConfig
from .schema import SourceRecord


@dataclass(frozen=True)
class WorkItem:
    split: str
    record: SourceRecord


def shard_work(items: list[WorkItem], n_workers: int) -> list[list[WorkItem]]:
    """Round-robin partition so each worker gets a roughly even mix of splits/categories."""
    n_workers = max(1, n_workers)
    shards: list[list[WorkItem]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        shards[i % n_workers].append(item)
    return shards


def _worker_main(
    worker_id: int,
    shard: list[WorkItem],
    output_root: Path,
    config: GenerationConfig,
    global_seed: int,
    pronunciation_map: dict[str, str],
    keep_intermediate: bool,
    device: str,
    progress_every: int,
    result_queue: mp.Queue,
) -> None:
    # Deferred imports: torch/CUDA must be initialized fresh in this child
    # process, never inherited across a fork.
    from . import env_info
    from .alignment_engine import ForcedAligner
    from .kokoro_backend import KokoroBackend
    from .pipeline import generate_one_record
    from .receipts import ReceiptStore
    from .reporting import ProgressTracker

    try:
        backend = KokoroBackend(device=device)
        aligner = ForcedAligner(device=device)
        environment = env_info.collect_environment(backend.model_identity(), aligner.model_identity())

        receipt_stores: dict[str, ReceiptStore] = {}
        tracker = ProgressTracker(total=len(shard))
        completed = 0
        failed = 0

        for i, item in enumerate(shard):
            split_dir = output_root / item.split
            store = receipt_stores.setdefault(item.split, ReceiptStore(output_root / "receipts"))

            outcome = generate_one_record(
                record=item.record,
                split=item.split,
                split_dir=split_dir,
                config=config,
                global_seed=global_seed,
                synth_fn=backend.synthesize,
                align_fn=aligner.align,
                receipt_store=store,
                environment=environment,
                pronunciation_map=pronunciation_map,
                keep_intermediate=keep_intermediate,
            )
            if outcome.success:
                completed += 1
                tracker.record_success(outcome.duration_seconds)
            else:
                failed += 1
                tracker.record_failure()
                store.mark_failed(
                    {"id": item.record.id, "split": item.split, "reasons": outcome.reasons, "timestamp": time.time()}
                )
                print(f"[worker {worker_id}] FAILED {item.record.id}: {outcome.reasons}")

            if (i + 1) % max(1, progress_every) == 0 or (i + 1) == len(shard):
                gpu_mem = env_info.gpu_memory_mb()
                print(f"[worker {worker_id}] " + tracker.line(item.split, item.record.id, gpu_mem))

        result_queue.put({"worker_id": worker_id, "completed": completed, "failed": failed, "error": None})
    except Exception as e:  # noqa: BLE001 - report, don't let one worker's crash hang the pool
        result_queue.put({"worker_id": worker_id, "completed": 0, "failed": len(shard), "error": str(e)})
        raise


def run_parallel(
    work_items: list[WorkItem],
    output_root: Path,
    config: GenerationConfig,
    global_seed: int,
    pronunciation_map: dict[str, str],
    keep_intermediate: bool,
    device: str,
    n_workers: int,
    progress_every: int,
) -> tuple[int, int]:
    """Run generation across ``n_workers`` spawned processes. Returns (completed, failed)."""
    if not work_items:
        return 0, 0

    ctx = mp.get_context("spawn")
    shards = [s for s in shard_work(work_items, n_workers) if s]
    result_queue: mp.Queue = ctx.Queue()
    processes = []

    for worker_id, shard in enumerate(shards):
        p = ctx.Process(
            target=_worker_main,
            args=(
                worker_id,
                shard,
                output_root,
                config,
                global_seed,
                pronunciation_map,
                keep_intermediate,
                device,
                progress_every,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    results = [result_queue.get() for _ in processes]
    for p in processes:
        p.join()

    total_completed = sum(r["completed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    errors = [r["error"] for r in results if r["error"]]
    if errors:
        print(f"WARNING: {len(errors)} worker(s) crashed: {errors}")

    return total_completed, total_failed
