from qa_filter.config import Settings
from qa_filter.judge_client import JudgeCallError
from qa_filter.judge_retry import judge_batch_with_retry, validate_verdicts


def _settings():
    return Settings(
        s3_bucket="b", source_key="k", output_prefix="p",
        cloudflare_account_id="a", cloudflare_api_token="t", model_id="m",
        reasoning_effort="low", max_output_tokens=2000, request_timeout_seconds=180,
        max_retries=4, retry_base_seconds=0,
        batch_token_ceiling=24000, max_batch_records=50, shard_size=5000, max_workers=10,
        run_state_dir="./run_state", judge_prompt_version="v1",
    )


def _record(item_id):
    return {"_batch_item_id": item_id, "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "<document>d</document>\n\nQ?"},
        {"role": "assistant", "content": "<think>r</think>\n\nA."},
    ]}


def test_validate_verdicts_true_when_ids_match_exactly():
    batch = [_record(1), _record(2)]
    verdicts = [{"id": 2, "verdict": "keep"}, {"id": 1, "verdict": "reject"}]
    assert validate_verdicts(batch, verdicts) is True


def test_validate_verdicts_false_when_id_missing():
    batch = [_record(1), _record(2)]
    verdicts = [{"id": 1, "verdict": "keep"}]
    assert validate_verdicts(batch, verdicts) is False


def test_validate_verdicts_false_when_id_duplicated():
    batch = [_record(1), _record(2)]
    verdicts = [{"id": 1, "verdict": "keep"}, {"id": 1, "verdict": "keep"}]
    assert validate_verdicts(batch, verdicts) is False


def test_judge_batch_with_retry_returns_verdicts_on_first_success():
    batch = [_record(1), _record(2)]

    def fake_judge(b, settings):
        return [{"id": r["_batch_item_id"], "verdict": "keep"} for r in b]

    verdicts, failed = judge_batch_with_retry(batch, _settings(), judge_fn=fake_judge)

    assert len(verdicts) == 2
    assert failed == []


def test_judge_batch_with_retry_recovers_via_same_size_retry_without_splitting():
    # Real production finding: an id-mismatch/null-content failure turned out to
    # be largely intermittent (GPT-OSS-120B is MoE-routed; routing can vary call
    # to call even at temperature=0), not a deterministic property of the batch.
    # A batch that fails once but succeeds on a same-size retry must NOT split --
    # splitting unnecessarily would multiply API calls for something that was
    # going to resolve itself.
    batch = [_record(1), _record(2)]
    call_log = []
    attempts = {"count": 0}

    def fake_judge(b, settings):
        call_log.append([r["_batch_item_id"] for r in b])
        attempts["count"] += 1
        if attempts["count"] == 1:
            return [{"id": 1, "verdict": "keep"}]  # first call: missing id 2 -> invalid
        return [{"id": r["_batch_item_id"], "verdict": "keep"} for r in b]  # second call: succeeds

    verdicts, failed = judge_batch_with_retry(batch, _settings(), judge_fn=fake_judge)

    assert len(verdicts) == 2
    assert failed == []
    assert call_log == [[1, 2], [1, 2]]  # never split -- both calls were the full batch


def test_judge_batch_with_retry_splits_batch_after_exhausting_same_size_retries():
    batch = [_record(1), _record(2)]
    call_log = []

    def fake_judge(b, settings):
        call_log.append([r["_batch_item_id"] for r in b])
        if len(b) == 2:
            return [{"id": 1, "verdict": "keep"}]  # missing id 2 -> invalid, every time
        return [{"id": r["_batch_item_id"], "verdict": "keep"} for r in b]

    verdicts, failed = judge_batch_with_retry(batch, _settings(), judge_fn=fake_judge, same_size_retries=2)

    assert len(verdicts) == 2
    assert failed == []
    # same_size_retries=2 -> the full batch is tried twice before splitting
    assert call_log == [[1, 2], [1, 2], [1], [2]]


def test_judge_batch_with_retry_gives_up_on_solo_record_after_repeated_failure():
    batch = [_record(1)]

    def always_fails(b, settings):
        raise JudgeCallError("simulated failure")

    verdicts, failed = judge_batch_with_retry(batch, _settings(), judge_fn=always_fails)

    assert verdicts == []
    assert len(failed) == 1
    assert failed[0]["_batch_item_id"] == 1
