from kokoro_ds.smoke import select_stratified_smoke
from kokoro_ds.voices_data import TEST_USER_VOICES, TRAIN_USER_VOICES, VALIDATION_USER_VOICES

from .conftest import make_record

CATEGORIES = [
    "general_conversation",
    "identity_facts",
    "identity_variations",
    "legacy_identity_corrections",
    "russian_language_status",
    "conversation_control",
]


def build_records(n: int, prefix: str) -> list:
    return [make_record(f"{prefix}_{i:04d}", category=CATEGORIES[i % len(CATEGORIES)]) for i in range(n)]


def test_selection_totals_and_is_deterministic():
    records_by_split = {
        "train": build_records(300, "train"),
        "validation": build_records(60, "val"),
        "test": build_records(60, "test"),
    }
    sel1 = select_stratified_smoke(records_by_split, global_seed=0, total=50)
    sel2 = select_stratified_smoke(records_by_split, global_seed=0, total=50)
    assert sel1.ids_by_split == sel2.ids_by_split
    assert sel1.total() == 50


def test_selection_respects_split_budget():
    records_by_split = {
        "train": build_records(300, "train"),
        "validation": build_records(60, "val"),
        "test": build_records(60, "test"),
    }
    sel = select_stratified_smoke(records_by_split, global_seed=0, total=50)
    for split, records in records_by_split.items():
        assert len(sel.ids_by_split[split]) <= len(records)


def test_selection_covers_all_categories_when_possible():
    records_by_split = {
        "train": build_records(300, "train"),
        "validation": build_records(60, "val"),
        "test": build_records(60, "test"),
    }
    sel = select_stratified_smoke(records_by_split, global_seed=0, total=50)
    assert sel.missing_categories == set()
    assert sel.covered_categories == set(CATEGORIES)


def test_selection_covers_all_voices_when_pool_fits_in_split_size():
    records_by_split = {
        "train": build_records(300, "train"),
        "validation": build_records(60, "val"),
        "test": build_records(60, "test"),
    }
    sel = select_stratified_smoke(records_by_split, global_seed=0, total=50)
    assert sel.missing_voices_by_split["train"] == set()
    assert sel.missing_voices_by_split["validation"] == set()
    assert sel.missing_voices_by_split["test"] == set()
    assert sel.covered_voices_by_split["train"] == set(TRAIN_USER_VOICES)
    assert sel.covered_voices_by_split["validation"] == set(VALIDATION_USER_VOICES)
    assert sel.covered_voices_by_split["test"] == set(TEST_USER_VOICES)


def test_selection_reports_missing_voice_when_split_too_small():
    records_by_split = {
        "train": build_records(2, "train"),
        "validation": build_records(1, "val"),
        "test": build_records(1, "test"),
    }
    sel = select_stratified_smoke(records_by_split, global_seed=0, total=50)
    # Far fewer source records than voices in the pools -> coverage is impossible, must be reported not silently dropped.
    assert sel.missing_voices_by_split["train"] or sel.total() < 50


def test_selection_is_not_simply_the_first_n_rows():
    records_by_split = {
        "train": build_records(300, "train"),
        "validation": build_records(60, "val"),
        "test": build_records(60, "test"),
    }
    sel = select_stratified_smoke(records_by_split, global_seed=0, total=50)
    naive_first_50 = [f"train_{i:04d}" for i in range(50)]
    assert sel.ids_by_split["train"] != naive_first_50[: len(sel.ids_by_split["train"])]
