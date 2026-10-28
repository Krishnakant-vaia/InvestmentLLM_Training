import os
from qa_filter.pause import clear_pause_sentinel, create_pause_sentinel, pause_requested


def test_pause_requested_false_when_no_sentinel(tmp_path):
    assert pause_requested(str(tmp_path)) is False


def test_create_and_detect_pause_sentinel(tmp_path):
    create_pause_sentinel(str(tmp_path))
    assert pause_requested(str(tmp_path)) is True
    assert os.path.exists(os.path.join(str(tmp_path), "PAUSE"))


def test_clear_pause_sentinel_removes_it(tmp_path):
    create_pause_sentinel(str(tmp_path))

    clear_pause_sentinel(str(tmp_path))

    assert pause_requested(str(tmp_path)) is False


def test_clear_pause_sentinel_is_a_noop_when_absent(tmp_path):
    clear_pause_sentinel(str(tmp_path))  # must not raise
    assert pause_requested(str(tmp_path)) is False
