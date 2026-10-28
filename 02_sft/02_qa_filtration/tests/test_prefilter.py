import pytest
from qa_filter.prefilter import (
    extract_final_answer,
    is_refusal_or_empty,
    is_structurally_valid,
    PreFilter,
)


def _record(user_content, assistant_content, question_type="Calculation/Numerical"):
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "_question_type": question_type,
    }


def test_extract_final_answer_strips_think_block():
    content = "<think>reasoning here</think>\n\nThe answer is 42."
    assert extract_final_answer(content) == "The answer is 42."


def test_extract_final_answer_raises_without_think_close():
    with pytest.raises(ValueError):
        extract_final_answer("no think block here")


def test_is_refusal_or_empty_flags_refusals():
    assert is_refusal_or_empty("") is True
    assert is_refusal_or_empty("   ") is True
    assert is_refusal_or_empty("I don't know the answer to that.") is True
    assert is_refusal_or_empty("As an AI, I cannot verify this.") is True
    assert is_refusal_or_empty("The MAPE drops from 0.60% to 0.55%.") is False


def test_is_structurally_valid_requires_document_and_think_tags():
    good = _record("<document>text</document>\n\nWhat is X?", "<think>r</think>\n\nAnswer.")
    ok, reason = is_structurally_valid(good)
    assert ok is True

    missing_doc = _record("What is X?", "<think>r</think>\n\nAnswer.")
    ok, reason = is_structurally_valid(missing_doc)
    assert ok is False
    assert "document" in reason

    missing_think = _record("<document>text</document>\n\nWhat is X?", "Answer with no think block.")
    ok, reason = is_structurally_valid(missing_think)
    assert ok is False
    assert "think" in reason


def test_prefilter_rejects_structurally_broken_and_refusals():
    good = _record("<document>d1</document>\n\nQ1?", "<think>r</think>\n\nA real answer here.")
    broken = _record("no doc tag", "<think>r</think>\n\nAnswer.")
    refusal = _record("<document>d2</document>\n\nQ2?", "<think>r</think>\n\nI don't know.")

    kept, rejected = PreFilter().filter([good, broken, refusal])

    assert kept == [good]
    assert len(rejected) == 2
    reasons = {r["_reject_reason"] for r in rejected}
    assert any("document" in r for r in reasons)
    assert any("refusal" in r for r in reasons)


def test_prefilter_dedups_near_identical_questions():
    r1 = _record(
        "<document>d1</document>\n\nWhat is the absolute change in MAPE from 0.05 to 0.20?",
        "<think>r</think>\n\nThe MAPE drops by 0.05 points.",
    )
    r2 = _record(
        "<document>d1</document>\n\nWhat is the absolute change in MAPE from 0.05 to 0.20 exactly?",
        "<think>r</think>\n\nThe MAPE drops by 0.05 points overall.",
    )

    kept, rejected = PreFilter(dedup_jaccard_threshold=0.5).filter([r1, r2])

    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0]["_reject_reason"] == "near_duplicate_question"
