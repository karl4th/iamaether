import json

from kokoro_ds.cli_validate import validate_dataset
from kokoro_ds.config import DEFAULT_CONFIG
from kokoro_ds.manifest import write_split_manifest, manifest_line
from kokoro_ds.pipeline import generate_one_record
from kokoro_ds.receipts import ReceiptStore

from .conftest import make_record

ENV = {"python_version": "3.12.0"}


def _write_source_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "id": r.id,
                        "category": r.category,
                        "turns": [{"speaker": t.speaker, "text": t.text} for t in r.turns],
                    }
                )
                + "\n"
            )


def _build_dataset(tmp_path, stub_synth_fn, stub_align_fn, n=3):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    records = [make_record(f"batch_0001_{i:04d}") for i in range(n)]
    _write_source_jsonl(source_root / "train.jsonl", records)
    (source_root / "validation.jsonl").write_text("")
    (source_root / "test.jsonl").write_text("")

    split_dir = output_root / "train"
    receipt_store = ReceiptStore(output_root / "receipts")
    lines = []
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
        assert outcome.success, outcome.reasons
        lines.append(manifest_line(outcome.relative_wav_path, outcome.duration_seconds))
    write_split_manifest(split_dir / "train.jsonl", lines)

    return source_root, output_root, records


def test_validate_dataset_reports_all_valid(tmp_path, stub_synth_fn, stub_align_fn):
    source_root, output_root, records = _build_dataset(tmp_path, stub_synth_fn, stub_align_fn)

    report, exit_code = validate_dataset(output_root, source_root, splits=["train"])

    assert exit_code == 0
    assert report["total_files"] == len(records)
    assert report["total_invalid"] == 0
    assert report["per_split"]["train"]["valid"] == len(records)


def test_validate_dataset_flags_manifest_duration_mismatch(tmp_path, stub_synth_fn, stub_align_fn):
    source_root, output_root, records = _build_dataset(tmp_path, stub_synth_fn, stub_align_fn)

    bad_lines = [manifest_line(f"audio/{r.id}.wav", 9999.0) for r in records]
    write_split_manifest(output_root / "train" / "train.jsonl", bad_lines)

    report, exit_code = validate_dataset(output_root, source_root, splits=["train"])
    assert exit_code == 1
    assert report["total_invalid"] == len(records)
    assert all("manifest duration" in p for entry in report["invalid_records"] for p in entry["problems"])


def test_validate_dataset_flags_missing_companion_json(tmp_path, stub_synth_fn, stub_align_fn):
    source_root, output_root, records = _build_dataset(tmp_path, stub_synth_fn, stub_align_fn)

    json_path = output_root / "train" / "audio" / f"{records[0].id}.json"
    json_path.unlink()

    report, exit_code = validate_dataset(output_root, source_root, splits=["train"])
    assert exit_code == 1
    assert report["total_invalid"] == 1


def test_validate_dataset_flags_wrong_assistant_voice(tmp_path, stub_synth_fn, stub_align_fn):
    from kokoro_ds.manifest import atomic_write_json, read_json

    source_root, output_root, records = _build_dataset(tmp_path, stub_synth_fn, stub_align_fn)
    json_path = output_root / "train" / "audio" / f"{records[0].id}.json"
    companion = read_json(json_path)
    companion["assistant_voice"] = "am_adam"
    atomic_write_json(json_path, companion)

    report, exit_code = validate_dataset(output_root, source_root, splits=["train"])
    assert exit_code == 1
    assert any("assistant_voice" in p for entry in report["invalid_records"] for p in entry["problems"])
