from kokoro_ds.cli_generate import _build_manifest_lines, _resolve_resume_state
from kokoro_ds.config import DEFAULT_CONFIG
from kokoro_ds.pipeline import generate_one_record
from kokoro_ds.receipts import ReceiptStore, wav_path_for

from .conftest import make_record

ENV = {"python_version": "3.12.0"}


def _generate_all(records, split_dir, receipt_store, stub_synth_fn, stub_align_fn):
    for r in records:
        outcome = generate_one_record(
            record=r,
            split="train",
            split_dir=split_dir,
            config=DEFAULT_CONFIG,
            global_seed=0,
            synth_fn=stub_synth_fn,
            align_fn=stub_align_fn,
            receipt_store=receipt_store,
            environment=ENV,
        )
        assert outcome.success


def test_resolve_resume_state_without_resume_reprocesses_everything(tmp_path, stub_synth_fn, stub_align_fn):
    records = [make_record(f"id_{i}") for i in range(3)]
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate_all(records, split_dir, receipt_store, stub_synth_fn, stub_align_fn)

    to_process, already_good = _resolve_resume_state(records, "train", split_dir, DEFAULT_CONFIG, 0, receipt_store, resume=False)
    assert already_good == 0
    assert len(to_process) == 3


def test_resolve_resume_state_with_resume_skips_valid_records(tmp_path, stub_synth_fn, stub_align_fn):
    records = [make_record(f"id_{i}") for i in range(3)]
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate_all(records, split_dir, receipt_store, stub_synth_fn, stub_align_fn)

    to_process, already_good = _resolve_resume_state(records, "train", split_dir, DEFAULT_CONFIG, 0, receipt_store, resume=True)
    assert already_good == 3
    assert to_process == []


def test_resolve_resume_state_reprocesses_corrupted_record(tmp_path, stub_synth_fn, stub_align_fn):
    records = [make_record(f"id_{i}") for i in range(2)]
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate_all(records, split_dir, receipt_store, stub_synth_fn, stub_align_fn)

    wav_path_for(split_dir, records[0].id).write_bytes(b"corrupted")

    to_process, already_good = _resolve_resume_state(records, "train", split_dir, DEFAULT_CONFIG, 0, receipt_store, resume=True)
    assert already_good == 1
    assert [r.id for r in to_process] == [records[0].id]


def test_resolve_resume_state_new_records_are_never_already_good(tmp_path):
    records = [make_record("never_generated")]
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")

    to_process, already_good = _resolve_resume_state(records, "train", split_dir, DEFAULT_CONFIG, 0, receipt_store, resume=True)
    assert already_good == 0
    assert to_process == records


def test_build_manifest_lines_only_includes_completed_records(tmp_path, stub_synth_fn, stub_align_fn):
    records = [make_record(f"id_{i}") for i in range(3)]
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate_all(records[:2], split_dir, receipt_store, stub_synth_fn, stub_align_fn)  # leave records[2] ungenerated

    lines = _build_manifest_lines(records, receipt_store)
    assert len(lines) == 2
    assert {l["path"] for l in lines} == {f"audio/{records[0].id}.wav", f"audio/{records[1].id}.wav"}
    assert all(l["duration"] > 0 for l in lines)
