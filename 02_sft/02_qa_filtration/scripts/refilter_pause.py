"""Request a clean pause of a running scripts/refilter_measurement.py.
Mirrors scripts/pause_run.py but points at the re-filter's own state dir.
"""
import sys

sys.path.insert(0, ".")

from qa_filter.pause import create_pause_sentinel

# Kept in sync by hand with REFILTER_STATE_DIR in refilter_measurement.py --
# not imported cross-script since scripts/ isn't a package.
REFILTER_STATE_DIR = "./refilter_state"


def main():
    create_pause_sentinel(REFILTER_STATE_DIR)
    print("pause requested -- refilter_measurement.py will stop after its current shard finishes")


if __name__ == "__main__":
    main()
