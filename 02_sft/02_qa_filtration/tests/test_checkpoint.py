import json
import time
from qa_filter.checkpoint import CheckpointStore


def test_checkpoint_starts_empty(tmp_path):
    store = CheckpointStore(str(tmp_path / "run_state"))
    assert store.is_complete(0) is False
    assert store.completed_shard_ids() == []
    assert store.get_total_shards() is None


def test_checkpoint_mark_complete_persists(tmp_path):
    run_state_dir = str(tmp_path / "run_state")
    store = CheckpointStore(run_state_dir)

    before = time.time()
    store.mark_complete(3)
    after = time.time()

    assert store.is_complete(3) is True
    assert store.is_complete(4) is False
    assert store.completed_shard_ids() == [3]
    assert before <= store.shard_timestamps()[3] <= after

    reloaded = CheckpointStore(run_state_dir)
    assert reloaded.is_complete(3) is True
    assert reloaded.shard_timestamps()[3] == store.shard_timestamps()[3]


def test_checkpoint_creates_directory_if_missing(tmp_path):
    run_state_dir = str(tmp_path / "does_not_exist_yet")
    store = CheckpointStore(run_state_dir)

    store.mark_complete(0)

    checkpoint_file = tmp_path / "does_not_exist_yet" / "checkpoint.json"
    assert checkpoint_file.exists()
    assert json.loads(checkpoint_file.read_text())["completed_shards"] == [0]


def test_checkpoint_total_shards_persists(tmp_path):
    run_state_dir = str(tmp_path / "run_state")
    store = CheckpointStore(run_state_dir)

    store.set_total_shards(175)

    assert store.get_total_shards() == 175
    reloaded = CheckpointStore(run_state_dir)
    assert reloaded.get_total_shards() == 175
