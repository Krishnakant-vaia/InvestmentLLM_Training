import json
import pytest
import responses
from qa_filter.config import Settings
from qa_filter.judge_client import build_judge_prompt, call_judge, JudgeCallError


def _settings(max_retries=4, retry_base_seconds=0):
    # retry_base_seconds=0 in tests: real backoff delays would make the suite slow
    return Settings(
        s3_bucket="b", source_key="k", output_prefix="p",
        cloudflare_account_id="acct123", cloudflare_api_token="tok123",
        model_id="@cf/openai/gpt-oss-120b", reasoning_effort="low",
        max_output_tokens=2000, request_timeout_seconds=180,
        max_retries=max_retries, retry_base_seconds=retry_base_seconds,
        batch_token_ceiling=24000, max_batch_records=50, shard_size=5000, max_workers=10,
        run_state_dir="./run_state", judge_prompt_version="v1",
    )


def _record(item_id, doc="doc text", question="What is X?", answer="X is 42."):
    return {
        "_batch_item_id": item_id,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"<document>{doc}</document>\n\n{question}"},
            # "zzz_hidden_think_marker_zzz" (not the word "reasoning") so the
            # exclusion check below can't collide with that word appearing
            # naturally in the static prompt template's own prose.
            {"role": "assistant", "content": f"<think>zzz_hidden_think_marker_zzz</think>\n\n{answer}"},
        ],
    }


def _judge_url(settings):
    return f"https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}/ai/run/{settings.model_id}"


def test_build_judge_prompt_includes_document_question_and_final_answer_only():
    batch = [_record(1)]

    prompt = build_judge_prompt(batch)

    assert "doc text" in prompt
    assert "What is X?" in prompt
    assert "X is 42." in prompt
    assert "zzz_hidden_think_marker_zzz" not in prompt  # <think> block must be excluded
    assert '"id": 1' in prompt


@responses.activate
def test_call_judge_parses_valid_response_from_choices_shape():
    settings = _settings()
    responses.add(
        responses.POST, _judge_url(settings),
        json={
            "result": {
                "choices": [{
                    "message": {
                        "content": json.dumps({"verdicts": [
                            {"id": 1, "verdict": "keep", "reason": "grounded and correct"}
                        ]})
                    }
                }]
            }
        },
        status=200,
    )
    batch = [_record(1)]

    verdicts = call_judge(batch, settings)

    assert verdicts == [{"id": 1, "verdict": "keep", "reason": "grounded and correct"}]


@responses.activate
def test_call_judge_parses_valid_response_from_response_fallback_shape():
    settings = _settings()
    responses.add(
        responses.POST, _judge_url(settings),
        json={"result": {"response": json.dumps({"verdicts": [
            {"id": 1, "verdict": "reject", "reason": "ungrounded number"}
        ]})}},
        status=200,
    )
    batch = [_record(1)]

    verdicts = call_judge(batch, settings)

    assert verdicts == [{"id": 1, "verdict": "reject", "reason": "ungrounded number"}]


@responses.activate
def test_call_judge_retries_on_429_then_succeeds():
    settings = _settings()
    url = _judge_url(settings)
    responses.add(responses.POST, url, json={"error": "rate limited"}, status=429)
    responses.add(
        responses.POST, url,
        json={"result": {"choices": [{"message": {"content": json.dumps(
            {"verdicts": [{"id": 1, "verdict": "keep", "reason": "ok"}]}
        )}}]}},
        status=200,
    )
    batch = [_record(1)]

    verdicts = call_judge(batch, settings)

    assert verdicts == [{"id": 1, "verdict": "keep", "reason": "ok"}]
    assert len(responses.calls) == 2


@responses.activate
def test_call_judge_raises_after_exhausting_retries_on_repeated_429():
    settings = _settings(max_retries=2, retry_base_seconds=0)
    url = _judge_url(settings)
    responses.add(responses.POST, url, json={"error": "rate limited"}, status=429)
    responses.add(responses.POST, url, json={"error": "rate limited"}, status=429)
    batch = [_record(1)]

    with pytest.raises(JudgeCallError):
        call_judge(batch, settings)

    assert len(responses.calls) == 2


@responses.activate
def test_call_judge_raises_immediately_on_400_no_retry():
    # A real response observed against the account: "Input length (133188)
    # exceeds model's maximum context length (131072)" for a batch whose
    # chars/4 token estimate undershot badly. This is permanent -- the same
    # oversized request fails identically every time -- so it must not be
    # retried with backoff; it should propagate straight to judge_retry's
    # split path, which actually fixes it (a smaller batch, smaller prompt).
    settings = _settings()
    url = _judge_url(settings)
    responses.add(
        responses.POST, url,
        json={"errors": [{"message": "Input length (133188) exceeds model's maximum context length (131072).", "code": 8007}], "success": False},
        status=400,
    )
    batch = [_record(1)]

    with pytest.raises(JudgeCallError):
        call_judge(batch, settings)

    assert len(responses.calls) == 1


@responses.activate
def test_call_judge_raises_immediately_on_401_no_retry():
    settings = _settings()
    url = _judge_url(settings)
    responses.add(responses.POST, url, json={"error": "unauthorized"}, status=401)
    batch = [_record(1)]

    with pytest.raises(JudgeCallError):
        call_judge(batch, settings)

    assert len(responses.calls) == 1


@responses.activate
def test_call_judge_raises_on_null_content_instead_of_crashing():
    # A real response shape observed against the account: "content" key present
    # but explicitly null (model ran out of max_tokens on reasoning before
    # emitting the answer) -- must not reach json.loads(None).
    settings = _settings()
    responses.add(
        responses.POST, _judge_url(settings),
        json={"result": {"choices": [{"finish_reason": "length", "message": {"content": None}}]}},
        status=200,
    )
    batch = [_record(1)]

    with pytest.raises(JudgeCallError):
        call_judge(batch, settings)


@responses.activate
def test_call_judge_raises_on_unparseable_content():
    settings = _settings()
    responses.add(
        responses.POST, _judge_url(settings),
        json={"result": {"choices": [{"message": {"content": "not json at all"}}]}},
        status=200,
    )
    batch = [_record(1)]

    with pytest.raises(JudgeCallError):
        call_judge(batch, settings)
