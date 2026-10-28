from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import Settings
from qa_filter.pipeline import process_shard


def _settings(tmp_path):
    return Settings(
        s3_bucket="b", source_key="k", output_prefix="p",
        cloudflare_account_id="a", cloudflare_api_token="t", model_id="m",
        reasoning_effort="low", max_output_tokens=2000, request_timeout_seconds=180,
        max_retries=4, retry_base_seconds=0,
        batch_token_ceiling=24000, max_batch_records=50, shard_size=5000, max_workers=2,
        run_state_dir=str(tmp_path / "run_state"), judge_prompt_version="v1",
    )


def _record(item_id, user="Q?", answer="A real answer."):
    return {
        "id": item_id,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": f"<document>d{item_id}</document>\n\n{user}{item_id}"},
            {"role": "assistant", "content": f"<think>r</think>\n\n{answer}"},
        ],
    }


def test_process_shard_separates_kept_and_rejected(tmp_path):
    settings = _settings(tmp_path)
    checkpoint = CheckpointStore(settings.run_state_dir)
    records = [_record(0), _record(1)]

    def fake_judge(batch, settings):
        verdicts = []
        for r in batch:
            v = "keep" if r["_batch_item_id"] == 0 else "reject"
            verdicts.append({"id": r["_batch_item_id"], "verdict": v, "reason": "test"})
        return verdicts, []

    result = process_shard(records, shard_id=0, settings=settings, checkpoint=checkpoint, judge_fn=fake_judge)

    assert len(result["kept"]) == 1
    assert result["kept"][0]["id"] == 0
    assert "_batch_item_id" not in result["kept"][0]
    assert result["kept"][0]["_judge_prompt_version"] == "v1"
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["id"] == 1
    assert "_batch_item_id" not in result["rejected"][0]
    assert result["rejected"][0]["_judge_prompt_version"] == "v1"
    assert "judge_rejected" in result["rejected"][0]["_reject_reason"]
    assert checkpoint.is_complete(0) is True


def test_process_shard_skips_already_completed_shard(tmp_path):
    settings = _settings(tmp_path)
    checkpoint = CheckpointStore(settings.run_state_dir)
    checkpoint.mark_complete(0)
    call_count = [0]

    def fake_judge(batch, settings):
        call_count[0] += 1
        return [], []

    result = process_shard([_record(0)], shard_id=0, settings=settings, checkpoint=checkpoint, judge_fn=fake_judge)

    assert call_count[0] == 0
    assert result == {"kept": [], "rejected": [], "shard_id": 0, "skipped": True}


def test_process_shard_includes_prefilter_rejections(tmp_path):
    settings = _settings(tmp_path)
    checkpoint = CheckpointStore(settings.run_state_dir)
    broken = {"id": 5, "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "no document tag here"},
        {"role": "assistant", "content": "<think>r</think>\n\nA."},
    ]}

    def fake_judge(batch, settings):
        return [], []

    result = process_shard([broken], shard_id=1, settings=settings, checkpoint=checkpoint, judge_fn=fake_judge)

    assert len(result["kept"]) == 0
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["_reject_reason"] == "missing_document_tag"
    assert result["rejected"][0]["_judge_prompt_version"] == "prefilter"
