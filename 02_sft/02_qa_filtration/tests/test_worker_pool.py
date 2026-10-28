import threading
import time
from qa_filter.config import Settings
from qa_filter.worker_pool import run_batches_concurrently


def _settings(max_workers=3):
    return Settings(
        s3_bucket="b", source_key="k", output_prefix="p",
        cloudflare_account_id="a", cloudflare_api_token="t", model_id="m",
        reasoning_effort="low", max_output_tokens=2000, request_timeout_seconds=180,
        max_retries=4, retry_base_seconds=0,
        batch_token_ceiling=24000, max_batch_records=50, shard_size=5000, max_workers=max_workers,
        run_state_dir="./run_state", judge_prompt_version="v1",
    )


def test_run_batches_concurrently_aggregates_results():
    batches = [[{"_batch_item_id": i}] for i in range(4)]

    def fake_judge_with_retry(batch, settings):
        return [{"id": r["_batch_item_id"], "verdict": "keep"} for r in batch], []

    verdicts, failed = run_batches_concurrently(batches, _settings(), fake_judge_with_retry)

    assert len(verdicts) == 4
    assert failed == []
    assert {v["id"] for v in verdicts} == {0, 1, 2, 3}


def test_run_batches_concurrently_respects_max_workers():
    batches = [[{"_batch_item_id": i}] for i in range(6)]
    active = []
    max_seen = [0]
    lock = threading.Lock()

    def fake_judge_with_retry(batch, settings):
        with lock:
            active.append(1)
            max_seen[0] = max(max_seen[0], len(active))
        time.sleep(0.05)
        with lock:
            active.pop()
        return [{"id": batch[0]["_batch_item_id"], "verdict": "keep"}], []

    run_batches_concurrently(batches, _settings(max_workers=2), fake_judge_with_retry)

    assert max_seen[0] <= 2


def test_run_batches_concurrently_collects_permanent_failures():
    batches = [[{"_batch_item_id": 0}], [{"_batch_item_id": 1}]]

    def fake_judge_with_retry(batch, settings):
        if batch[0]["_batch_item_id"] == 1:
            return [], list(batch)
        return [{"id": 0, "verdict": "keep"}], []

    verdicts, failed = run_batches_concurrently(batches, _settings(), fake_judge_with_retry)

    assert len(verdicts) == 1
    assert len(failed) == 1
    assert failed[0]["_batch_item_id"] == 1
