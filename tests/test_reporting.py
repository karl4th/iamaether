from kokoro_ds.reporting import ProgressTracker, format_hms, percentiles


def test_percentiles_empty():
    p = percentiles([])
    assert p["p50"] != p["p50"]  # NaN


def test_percentiles_basic():
    p = percentiles([1, 2, 3, 4, 5], ps=(0, 50, 100))
    assert p["p0"] == 1
    assert p["p50"] == 3
    assert p["p100"] == 5


def test_percentiles_single_value():
    p = percentiles([7.0])
    assert all(v == 7.0 for v in p.values())


def test_format_hms():
    assert format_hms(0) == "00:00:00"
    assert format_hms(3661) == "01:01:01"


def test_progress_tracker_counts_and_rate():
    tracker = ProgressTracker(total=10)
    tracker.record_success(2.0)
    tracker.record_success(3.0)
    tracker.record_failure()
    assert tracker.completed == 2
    assert tracker.failed == 1
    assert tracker.audio_seconds_generated == 5.0
    line = tracker.line("train", "batch_0001_0001")
    assert "train" in line and "batch_0001_0001" in line
