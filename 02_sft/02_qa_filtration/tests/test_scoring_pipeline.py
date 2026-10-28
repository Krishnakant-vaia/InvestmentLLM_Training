import json

from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import Settings
from qa_filter.judge_client import JudgeCallError
from qa_scoring.pipeline import score_shard
from qa_scoring.records import scorer_payload

CAT = "CATEGORY_GENERAL_FINANCIAL_REASONING_AND_RATIOS"


def _settings(tmp_path):
    return Settings(
        s3_bucket="b", source_key="k", output_prefix="p",
        cloudflare_account_id="a", cloudflare_api_token="t", model_id="m",
        reasoning_effort="low", max_output_tokens=2000, request_timeout_seconds=180,
        max_retries=4, retry_base_seconds=0,
        batch_token_ceiling=24000, max_batch_records=50, shard_size=5000, max_workers=2,
        run_state_dir=str(tmp_path / "state"), judge_prompt_version="v1",
    )


def _record(n):
    return {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": f"<document>doc{n}</document>\n\nQ{n}?"},
            {"role": "assistant", "content": f"<think>SECRET{n}</think>\n\nAnswer {n}."},
        ],
        "_chunk_id": f"chunk{n}",
        "_judge_prompt_version": "gepa_strict_v2",
        "_question_type": "Calculation/Numerical",
    }


def _all_pass(batch, settings):
    return [{"id": i, "score": 9, "categories": [CAT], "reason": "PASS_ALL_GATES"} for i in range(len(batch))]


def test_output_keeps_every_original_field_and_adds_quality_fields(tmp_path):
    settings = _settings(tmp_path)
    checkpoint = CheckpointStore(settings.run_state_dir)
    records = [_record(0), _record(1)]

    result = score_shard(records, shard_id=3, first_index=15000, settings=settings,
                         checkpoint=checkpoint, prompt_version="pv-1", score_fn=_all_pass)

    assert result["unscored"] == []
    by_index = {r["_quality_src_index"]: r for r in result["scored"]}
    assert set(by_index) == {15000, 15001}
    out = by_index[15000]
    # the deliverable record is the ORIGINAL one: think block and metadata intact
    assert "SECRET0" in out["messages"][2]["content"]
    assert out["_chunk_id"] == "chunk0"
    assert out["_judge_prompt_version"] == "gepa_strict_v2"
    assert out["_quality_score"] == 9
    assert out["_quality_categories"] == [CAT]
    assert out["_quality_reason"] == "PASS_ALL_GATES"
    assert out["_quality_prompt_version"] == "pv-1"
    assert "_quality_anomalies" not in out
    assert checkpoint.is_complete(3) is True


def test_the_scorer_only_ever_sees_messages_without_think_or_metadata(tmp_path):
    settings = _settings(tmp_path)
    seen = []

    def spy(batch, settings):
        seen.extend(json.dumps(scorer_payload(item)) for item in batch)
        return _all_pass(batch, settings)

    score_shard([_record(0), _record(1)], 0, 0, settings,
                CheckpointStore(settings.run_state_dir), "pv-1", spy)

    assert len(seen) == 2
    blob = "\n".join(seen)
    assert "SECRET" not in blob and "chunk" not in blob and "gepa_strict_v2" not in blob


def test_a_record_that_cannot_be_scored_goes_to_unscored_with_its_index(tmp_path):
    settings = _settings(tmp_path)

    def always_fails(batch, settings):
        raise JudgeCallError("boom")

    result = score_shard([_record(0)], 1, 5000, settings,
                         CheckpointStore(settings.run_state_dir), "pv-1", always_fails)

    assert result["scored"] == []
    assert len(result["unscored"]) == 1
    row = result["unscored"][0]
    assert row["_quality_reason"] == "score_call_failed"
    assert row["_quality_src_index"] == 5000
    assert "SECRET0" in row["messages"][2]["content"]


def test_anomalies_are_recorded_on_the_record_only_when_present(tmp_path):
    settings = _settings(tmp_path)

    def sloppy(batch, settings):
        return [{"id": 0, "score": 8, "categories": ["CATEGORY_MADE_UP"], "reason": "PASS_ALL_GATES"}]

    result = score_shard([_record(0)], 0, 0, settings,
                         CheckpointStore(settings.run_state_dir), "pv-1", sloppy)

    assert result["scored"][0]["_quality_anomalies"] == ["bad_categories"]
    assert result["scored"][0]["_quality_categories"] == []


def test_an_already_completed_shard_is_skipped_without_calling_the_scorer(tmp_path):
    settings = _settings(tmp_path)
    checkpoint = CheckpointStore(settings.run_state_dir)
    checkpoint.mark_complete(2)
    calls = []

    def spy(batch, settings):
        calls.append(1)
        return []

    assert score_shard([_record(0)], 2, 0, settings, checkpoint, "pv-1", spy) is None
    assert calls == []
