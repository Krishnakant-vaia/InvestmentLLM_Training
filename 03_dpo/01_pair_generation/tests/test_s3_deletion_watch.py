"""The deletion watcher reports new delete markers and vanished versions, and nothing else."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import s3_deletion_watch as w  # noqa: E402


def test_nothing_changed_nothing_reported_and_new_files_are_not_deletions():
    before = ({("a", "v1")}, set())
    assert w.changes(before, before) == []
    assert w.changes(before, ({("a", "v1"), ("b", "v1")}, set())) == []


def test_a_delete_marker_and_a_vanished_version_are_both_reported():
    before = ({("a", "v1"), ("b", "v1")}, set())
    after = ({("a", "v1")}, {("c", "m1")})
    lines = w.changes(before, after)
    assert any("NEW DELETE MARKER" in l and "c" in l for l in lines)
    assert any("VERSION GONE" in l and "b" in l for l in lines)
