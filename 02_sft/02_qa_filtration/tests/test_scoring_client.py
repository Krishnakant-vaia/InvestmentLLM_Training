import json

import pytest
import responses

from qa_filter.config import Settings
from qa_filter.judge_client import JudgeCallError
from qa_scoring.records import to_scorer_item
from qa_scoring.score_client import build_scorer_prompt, call_scorer

TEMPLATE = "HEADER\n{{batch_records_jsonl}}\nFOOTER"


def _settings():
    return Settings(
        s3_bucket="b", source_key="k", output_prefix="p",
        cloudflare_account_id="acct123", cloudflare_api_token="tok123",
        model_id="@cf/openai/gpt-oss-120b", reasoning_effort="low",
        max_output_tokens=2000, request_timeout_seconds=180,
        max_retries=4, retry_base_seconds=0,
        batch_token_ceiling=24000, max_batch_records=50, shard_size=5000, max_workers=10,
        run_state_dir="./run_state", judge_prompt_version="v1",
    )


def _item(index, document="D", answer="A."):
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"<document>{document}</document>\n\nQ{index}?"},
            {"role": "assistant", "content": f"<think>SECRET REASONING</think>\n\n{answer}"},
        ],
        "_chunk_id": "abc123",
        "_judge_prompt_version": "gepa_strict_v2",
    }
    return to_scorer_item(record, index)


def _url(settings):
    return f"https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}/ai/run/{settings.model_id}"


def test_prompt_has_one_json_object_per_line_and_no_metadata_or_reasoning():
    prompt = build_scorer_prompt(TEMPLATE, [_item(0), _item(1)])

    lines = prompt.split("\n")
    assert lines[0] == "HEADER" and lines[-1] == "FOOTER"
    payloads = [json.loads(line) for line in lines[1:-1]]
    assert len(payloads) == 2
    assert all(set(p) == {"messages"} for p in payloads)
    # blind judging: earlier verdicts, ids and the reasoning trace never reach the model
    assert "_chunk_id" not in prompt
    assert "gepa_strict_v2" not in prompt
    assert "SECRET REASONING" not in prompt


def test_non_ascii_is_sent_raw_not_escaped():
    prompt = build_scorer_prompt(TEMPLATE, [_item(0, document="Revenue rose to ₹120 crore")])

    assert "₹120 crore" in prompt
    assert "\\u20b9" not in prompt


@responses.activate
def test_call_scorer_returns_the_scores_array():
    settings = _settings()
    scores = [{"id": 0, "score": 9, "categories": ["CATEGORY_GENERAL_FINANCIAL_REASONING_AND_RATIOS"], "reason": "PASS_ALL_GATES"}]
    responses.add(
        responses.POST, _url(settings),
        json={"result": {"choices": [{"message": {"content": json.dumps({"scores": scores})}}]}},
        status=200,
    )

    assert call_scorer([_item(0)], settings, template=TEMPLATE) == scores


@responses.activate
def test_call_scorer_raises_when_scores_key_is_missing():
    settings = _settings()
    responses.add(
        responses.POST, _url(settings),
        json={"result": {"choices": [{"message": {"content": json.dumps({"verdicts": []})}}]}},
        status=200,
    )

    with pytest.raises(JudgeCallError):
        call_scorer([_item(0)], settings, template=TEMPLATE)


@responses.activate
def test_call_scorer_treats_400_as_permanent_and_does_not_retry():
    settings = _settings()
    responses.add(responses.POST, _url(settings), json={"errors": ["Input length exceeds"]}, status=400)

    with pytest.raises(JudgeCallError):
        call_scorer([_item(0)], settings, template=TEMPLATE)

    assert len(responses.calls) == 1
