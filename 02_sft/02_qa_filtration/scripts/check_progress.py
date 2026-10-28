"""Reports how far a run_pipeline.py run has gotten: shards done, percent
complete, and an ETA based on the observed average time per shard so far.
Safe to run at any time, including while run_pipeline.py is still running
(reads checkpoint.json, doesn't write to it).
"""
import sys

sys.path.insert(0, ".")

from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import load_settings
from qa_filter.progress import compute_progress


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


def main():
    settings = load_settings()
    checkpoint = CheckpointStore(settings.run_state_dir)
    progress = compute_progress(checkpoint)

    if progress["total_shards"] is None:
        print("no run in progress yet (total_shards not set -- run_pipeline.py hasn't started)")
        return

    print(f"shards: {progress['completed_shards']}/{progress['total_shards']}")
    if progress["percent_complete"] is not None:
        print(f"complete: {progress['percent_complete']}%")
    if progress["avg_seconds_per_shard"] is not None:
        print(f"avg time/shard: {_format_duration(progress['avg_seconds_per_shard'])}")
    if progress["eta_seconds"] is not None:
        print(f"ETA: {_format_duration(progress['eta_seconds'])}")
    else:
        print("ETA: not enough shards completed yet to estimate")


if __name__ == "__main__":
    main()
