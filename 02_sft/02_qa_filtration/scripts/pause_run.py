"""Request a clean pause of a running scripts/run_pipeline.py. The running
process checks for this sentinel only after a shard fully finishes (never
mid-shard), finishes uploading everything done so far to S3, then exits.
Resuming is just rerunning run_pipeline.py again -- checkpointing already
skips completed shards.
"""
import sys

sys.path.insert(0, ".")

from qa_filter.config import load_settings
from qa_filter.pause import create_pause_sentinel


def main():
    settings = load_settings()
    create_pause_sentinel(settings.run_state_dir)
    print("pause requested -- run_pipeline.py will stop after its current shard finishes")


if __name__ == "__main__":
    main()
