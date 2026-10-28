from qa_filter.checkpoint import CheckpointStore
from qa_filter.progress import compute_progress


def test_compute_progress_with_no_shards_done_yet(tmp_path):
    checkpoint = CheckpointStore(str(tmp_path / "run_state"))
    checkpoint.set_total_shards(100)

    progress = compute_progress(checkpoint)

    assert progress["completed_shards"] == 0
    assert progress["total_shards"] == 100
    assert progress["percent_complete"] == 0.0
    assert progress["avg_seconds_per_shard"] is None
    assert progress["eta_seconds"] is None


def test_compute_progress_with_unknown_total_shards(tmp_path):
    checkpoint = CheckpointStore(str(tmp_path / "run_state"))
    checkpoint.mark_complete(0)

    progress = compute_progress(checkpoint)

    assert progress["total_shards"] is None
    assert progress["percent_complete"] is None
    assert progress["eta_seconds"] is None


def test_compute_progress_computes_percent_and_eta(tmp_path, monkeypatch):
    checkpoint = CheckpointStore(str(tmp_path / "run_state"))
    checkpoint.set_total_shards(10)

    times = iter([1000.0, 1010.0, 1030.0])  # 10s then 20s between completions
    monkeypatch.setattr("time.time", lambda: next(times))
    checkpoint.mark_complete(0)  # t=1000
    checkpoint.mark_complete(1)  # t=1010
    checkpoint.mark_complete(2)  # t=1030

    progress = compute_progress(checkpoint)

    assert progress["completed_shards"] == 3
    assert progress["percent_complete"] == 30.0
    # deltas [10, 20]; neither is an outlier (both within 3x the smallest, 10)
    # so both count: avg = (10 + 20) / 2 = 15s
    assert progress["avg_seconds_per_shard"] == 15.0
    # 7 shards remaining * 15s/shard = 105s
    assert progress["eta_seconds"] == 105.0


def test_compute_progress_excludes_pause_resume_gap_from_average(tmp_path, monkeypatch):
    # Real production finding: a pause/resume cycle left one delta far larger
    # than normal per-shard time in checkpoint.json's timestamps, and the naive
    # (last - first) / (n - 1) average over the whole history counted that dead
    # time as if it were processing time -- a 314-hour ETA after one pause.
    checkpoint = CheckpointStore(str(tmp_path / "run_state"))
    checkpoint.set_total_shards(100)

    # Two normal 10s shards, then a ~1-hour pause/resume gap, then another
    # normal 10s shard.
    times = iter([1000.0, 1010.0, 1020.0, 4620.0])
    monkeypatch.setattr("time.time", lambda: next(times))
    checkpoint.mark_complete(0)  # t=1000
    checkpoint.mark_complete(1)  # t=1010 (delta 10s)
    checkpoint.mark_complete(2)  # t=1020 (delta 10s)
    checkpoint.mark_complete(3)  # t=4620 (delta 3600s -- the pause, excluded)

    progress = compute_progress(checkpoint)

    assert progress["completed_shards"] == 4
    # only the two 10s deltas count; the 3600s pause gap is filtered out
    assert progress["avg_seconds_per_shard"] == 10.0
    # 96 shards remaining * 10s/shard = 960s, NOT inflated by the pause
    assert progress["eta_seconds"] == 960.0
