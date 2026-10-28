from qa_filter.config import Settings
from qa_filter.judge_client import JudgeCallError
from qa_scoring.records import to_scorer_item
from qa_scoring.scoring import (
    clean_categories,
    score_anomalies,
    score_batch_with_retry,
    validate_scores,
)

CAT = "CATEGORY_GENERAL_FINANCIAL_REASONING_AND_RATIOS"
CAT2 = "CATEGORY_COVENANT_COMPLIANCE_AND_CREDIT_AGREEMENTS"


def _settings():
    return Settings(
        s3_bucket="b", source_key="k", output_prefix="p",
        cloudflare_account_id="a", cloudflare_api_token="t", model_id="m",
        reasoning_effort="low", max_output_tokens=2000, request_timeout_seconds=180,
        max_retries=4, retry_base_seconds=0,
        batch_token_ceiling=24000, max_batch_records=50, shard_size=5000, max_workers=10,
        run_state_dir="./run_state", judge_prompt_version="v1",
    )


def _item(index):
    record = {"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": f"<document>d</document>\n\nQ{index}?"},
        {"role": "assistant", "content": "<think>r</think>\n\nA."},
    ]}
    return to_scorer_item(record, index)


def _score(position, score=8, categories=None, reason=None):
    return {
        "id": position,
        "score": score,
        "categories": categories if categories is not None else [CAT],
        "reason": reason if reason is not None else ("PASS_ALL_GATES" if score >= 7 else "F06_NUMERICAL_ERROR"),
    }


# ---- validate_scores: hard checks (a failure here triggers retry/split) ----

def test_validate_accepts_exact_positions_in_any_order():
    assert validate_scores(3, [_score(2), _score(0), _score(1)]) is True


def test_validate_rejects_a_missing_position():
    assert validate_scores(3, [_score(0), _score(1)]) is False


def test_validate_rejects_a_duplicate_position():
    assert validate_scores(2, [_score(0), _score(0)]) is False


def test_validate_rejects_a_position_outside_the_batch():
    assert validate_scores(2, [_score(0), _score(2)]) is False


def test_validate_rejects_scores_outside_1_to_10():
    assert validate_scores(1, [_score(0, score=0)]) is False
    assert validate_scores(1, [_score(0, score=11)]) is False


def test_validate_rejects_non_integer_scores():
    for bad in (8.0, "8", True, None):
        obj = _score(0)
        obj["score"] = bad
        assert validate_scores(1, [obj]) is False, bad


def test_validate_rejects_non_integer_ids():
    obj = _score(0)
    obj["id"] = "0"
    assert validate_scores(1, [obj]) is False


# ---- anomalies / category cleaning: soft checks (recorded, never retried) ----

def test_no_anomalies_for_clean_pass_and_clean_fail():
    assert score_anomalies(_score(0, score=9)) == []
    assert score_anomalies(_score(0, score=2, reason="F05_CONTRADICTED_CLAIM")) == []


def test_anomaly_for_bad_categories():
    assert "bad_categories" in score_anomalies(_score(0, categories=[]))
    assert "bad_categories" in score_anomalies(_score(0, categories=["CATEGORY_MADE_UP"]))
    assert "bad_categories" in score_anomalies(_score(0, categories=[CAT, CAT]))
    assert "bad_categories" in score_anomalies(_score(0, categories=[CAT, CAT2, CAT, CAT2]))


def test_anomaly_for_reason_that_contradicts_the_score():
    assert "reason_mismatch" in score_anomalies(_score(0, score=8, reason="F06_NUMERICAL_ERROR"))
    assert "reason_mismatch" in score_anomalies(_score(0, score=3, reason="PASS_ALL_GATES"))
    assert "reason_mismatch" in score_anomalies(_score(0, score=3, reason="not a code"))


def test_clean_categories_drops_unknown_codes_and_duplicates_keeping_order():
    assert clean_categories([CAT2, "CATEGORY_MADE_UP", CAT2, CAT]) == [CAT2, CAT]
    assert clean_categories("nonsense") == []


# ---- score_batch_with_retry ----

def test_results_are_aligned_to_records_on_success():
    batch = [_item(10), _item(11), _item(12)]

    def fake(b, settings):
        return [_score(i, score=5 + i) for i in range(len(b))]

    results, failed = score_batch_with_retry(batch, _settings(), fake)

    assert failed == []
    assert [(r["item"]["_idx"], r["score"]) for r in results] == [(10, 5), (11, 6), (12, 7)]


def test_recovers_by_same_size_retry_without_splitting():
    batch = [_item(0), _item(1)]
    sizes = []

    def fake(b, settings):
        sizes.append(len(b))
        if len(sizes) == 1:
            return [_score(0)]  # missing position 1 -> invalid
        return [_score(i) for i in range(len(b))]

    results, failed = score_batch_with_retry(batch, _settings(), fake)

    assert len(results) == 2 and failed == []
    assert sizes == [2, 2]


def test_split_rebases_positions_for_each_sub_batch():
    # The model reports 0-indexed position WITHIN the batch it was sent, so after a
    # split each half is scored from 0 again. Each record's score encodes its own
    # _idx so a mapping bug (e.g. reusing the original positions) is caught.
    batch = [_item(10), _item(11), _item(12), _item(13)]
    sizes = []

    def fake(b, settings):
        sizes.append(len(b))
        if len(b) == 4:
            return [_score(0), _score(1), _score(2)]  # always invalid at full size
        return [_score(i, score=b[i]["_idx"] - 9) for i in range(len(b))]

    results, failed = score_batch_with_retry(batch, _settings(), fake)

    assert failed == []
    assert sizes == [4, 4, 2, 2]
    assert {r["item"]["_idx"]: r["score"] for r in results} == {10: 1, 11: 2, 12: 3, 13: 4}


def test_solo_record_that_keeps_failing_is_returned_as_failed():
    batch = [_item(5)]

    def fake(b, settings):
        raise JudgeCallError("boom")

    results, failed = score_batch_with_retry(batch, _settings(), fake)

    assert results == []
    assert [item["_idx"] for item in failed] == [5]


def test_anomalies_and_cleaned_categories_ride_along_without_forcing_a_retry():
    batch = [_item(0)]
    calls = []

    def fake(b, settings):
        calls.append(1)
        return [_score(0, score=8, categories=[CAT, "CATEGORY_MADE_UP"], reason="PASS_ALL_GATES")]

    results, failed = score_batch_with_retry(batch, _settings(), fake)

    assert len(calls) == 1 and failed == []
    assert results[0]["categories"] == [CAT]
    assert results[0]["anomalies"] == ["bad_categories"]
