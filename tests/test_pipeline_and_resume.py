from dataclasses import replace

from kokoro_ds.config import DEFAULT_CONFIG
from kokoro_ds.determinism import assign_user_voice
from kokoro_ds.pipeline import generate_one_record
from kokoro_ds.receipts import ReceiptStore, json_path_for, verify_existing_record, wav_path_for

from .conftest import make_record

ENV = {"python_version": "3.12.0", "torch_version": "test"}


def _generate(record, split, split_dir, receipt_store, synth_fn, align_fn, seed=0, config=DEFAULT_CONFIG):
    return generate_one_record(
        record=record,
        split=split,
        split_dir=split_dir,
        config=config,
        global_seed=seed,
        synth_fn=synth_fn,
        align_fn=align_fn,
        receipt_store=receipt_store,
        environment=ENV,
    )


def test_generate_one_record_success(tmp_path, stub_synth_fn, stub_align_fn):
    record = make_record("batch_0001_0001")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")

    outcome = _generate(record, "train", split_dir, receipt_store, stub_synth_fn, stub_align_fn)

    assert outcome.success, outcome.reasons
    assert outcome.duration_seconds > 0
    assert wav_path_for(split_dir, record.id).exists()
    assert json_path_for(split_dir, record.id).exists()
    assert receipt_store.is_completed(record.id)


def test_generate_one_record_alignment_failure_leaves_no_files(tmp_path, stub_synth_fn):
    record = make_record("batch_0001_0002")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")

    def failing_align(samples, sample_rate, text):
        raise RuntimeError("alignment model exploded")

    outcome = _generate(record, "train", split_dir, receipt_store, stub_synth_fn, failing_align)

    assert not outcome.success
    assert any("alignment" in r for r in outcome.reasons)
    assert not wav_path_for(split_dir, record.id).exists()
    assert not receipt_store.is_completed(record.id)


def test_generate_one_record_synthesis_failure_leaves_no_files(tmp_path, stub_align_fn):
    record = make_record("batch_0001_0003")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")

    def failing_synth(text, voice, speed, lang_code):
        raise RuntimeError("kokoro crashed")

    outcome = _generate(record, "train", split_dir, receipt_store, failing_synth, stub_align_fn)

    assert not outcome.success
    assert not wav_path_for(split_dir, record.id).exists()
    assert not receipt_store.is_completed(record.id)


def test_resume_accepts_valid_existing_record(tmp_path, stub_synth_fn, stub_align_fn):
    record = make_record("batch_0001_0004")
    split_dir = tmp_path / "validation"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate(record, "validation", split_dir, receipt_store, stub_synth_fn, stub_align_fn)

    reloaded_store = ReceiptStore(tmp_path / "receipts")  # simulate a fresh process picking the run back up
    user_voice = assign_user_voice(record.id, "validation", 0)
    result = verify_existing_record(split_dir, record, "validation", user_voice, DEFAULT_CONFIG, reloaded_store)

    assert result.ok, result.reasons


def test_resume_rejects_corrupted_wav(tmp_path, stub_synth_fn, stub_align_fn):
    record = make_record("batch_0001_0005")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate(record, "train", split_dir, receipt_store, stub_synth_fn, stub_align_fn)

    wav_path_for(split_dir, record.id).write_bytes(b"not a real wav file")

    user_voice = assign_user_voice(record.id, "train", 0)
    result = verify_existing_record(split_dir, record, "train", user_voice, DEFAULT_CONFIG, receipt_store)
    assert not result.ok
    assert any("failed to open" in r for r in result.reasons)


def test_resume_rejects_when_source_text_changed(tmp_path, stub_synth_fn, stub_align_fn):
    record = make_record("batch_0001_0006")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate(record, "train", split_dir, receipt_store, stub_synth_fn, stub_align_fn)

    from kokoro_ds.schema import Turn

    changed_record = replace(record, turns=tuple(Turn(t.speaker, t.text + " EDITED") for t in record.turns))

    user_voice = assign_user_voice(record.id, "train", 0)
    result = verify_existing_record(split_dir, changed_record, "train", user_voice, DEFAULT_CONFIG, receipt_store)
    assert not result.ok
    assert any("source_checksum" in r for r in result.reasons)


def test_resume_rejects_when_config_changed(tmp_path, stub_synth_fn, stub_align_fn):
    record = make_record("batch_0001_0007")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate(record, "train", split_dir, receipt_store, stub_synth_fn, stub_align_fn, config=DEFAULT_CONFIG)

    changed_config = replace(DEFAULT_CONFIG, fade_seconds=DEFAULT_CONFIG.fade_seconds * 5)
    user_voice = assign_user_voice(record.id, "train", 0)
    result = verify_existing_record(split_dir, record, "train", user_voice, changed_config, receipt_store)
    assert not result.ok
    assert any("config_checksum" in r for r in result.reasons)


def test_resume_ignores_qa_gate_only_config_changes(tmp_path, stub_synth_fn, stub_align_fn):
    """Tuning a post-hoc QA gate (never baked into the audio) must never force a re-verify."""
    record = make_record("batch_0001_0007b")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    _generate(record, "train", split_dir, receipt_store, stub_synth_fn, stub_align_fn, config=DEFAULT_CONFIG)

    changed_config = replace(
        DEFAULT_CONFIG,
        clipping_peak_threshold=DEFAULT_CONFIG.clipping_peak_threshold * 2,
        alignment_min_confidence=DEFAULT_CONFIG.alignment_min_confidence + 0.3,
    )
    user_voice = assign_user_voice(record.id, "train", 0)
    result = verify_existing_record(split_dir, record, "train", user_voice, changed_config, receipt_store)
    assert result.ok, result.reasons


def test_resume_rejects_missing_files(tmp_path):
    record = make_record("batch_0001_0008")
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")
    user_voice = assign_user_voice(record.id, "train", 0)
    result = verify_existing_record(split_dir, record, "train", user_voice, DEFAULT_CONFIG, receipt_store)
    assert not result.ok
    assert result.reasons == ["wav missing"]


def test_regeneration_is_idempotent_across_a_full_batch(tmp_path, stub_synth_fn, stub_align_fn):
    """Simulates a resumed run: every previously completed record verifies clean without re-synthesizing."""
    records = [make_record(f"batch_0001_{i:04d}") for i in range(5)]
    split_dir = tmp_path / "train"
    receipt_store = ReceiptStore(tmp_path / "receipts")

    for r in records:
        outcome = _generate(r, "train", split_dir, receipt_store, stub_synth_fn, stub_align_fn)
        assert outcome.success

    resumed_store = ReceiptStore(tmp_path / "receipts")
    for r in records:
        user_voice = assign_user_voice(r.id, "train", 0)
        result = verify_existing_record(split_dir, r, "train", user_voice, DEFAULT_CONFIG, resumed_store)
        assert result.ok, (r.id, result.reasons)
