from qa_filter.checkpoint import CheckpointStore


def compute_progress(checkpoint: CheckpointStore) -> dict:
    completed_ids = checkpoint.completed_shard_ids()
    completed_shards = len(completed_ids)
    total_shards = checkpoint.get_total_shards()
    timestamps = checkpoint.shard_timestamps()

    percent_complete = None
    if total_shards:
        percent_complete = round(100.0 * completed_shards / total_shards, 4)

    avg_seconds_per_shard = None
    if completed_shards >= 2:
        ordered = sorted(timestamps[shard_id] for shard_id in completed_ids)
        deltas = [b - a for a, b in zip(ordered, ordered[1:])]
        # A pause/resume cycle shows up as one delta far larger than normal
        # per-shard processing time (observed in practice: real shards run in
        # minutes; a pause/resume gap inflated one delta to over an hour,
        # ballooning the naive full-history average into a 314-hour ETA).
        # Filtering to deltas within 3x the smallest observed delta cleanly
        # separates real processing time from dead time in every case seen so
        # far. If every delta gets filtered out (e.g. only two shards done,
        # straddling a pause), there's no reliable data yet -- stay None
        # rather than report a number built entirely from dead time.
        min_delta = min(deltas)
        normal_deltas = [d for d in deltas if d <= 3 * min_delta]
        if normal_deltas:
            avg_seconds_per_shard = sum(normal_deltas) / len(normal_deltas)

    eta_seconds = None
    if avg_seconds_per_shard is not None and total_shards is not None:
        remaining = max(total_shards - completed_shards, 0)
        eta_seconds = remaining * avg_seconds_per_shard

    return {
        "completed_shards": completed_shards,
        "total_shards": total_shards,
        "percent_complete": percent_complete,
        "avg_seconds_per_shard": avg_seconds_per_shard,
        "eta_seconds": eta_seconds,
    }
