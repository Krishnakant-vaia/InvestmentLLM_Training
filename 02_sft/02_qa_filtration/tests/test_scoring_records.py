from qa_scoring.records import scorer_payload, strip_think, to_scorer_item


def _record():
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<document>D</document>\n\nWhat is X?"},
            {"role": "assistant", "content": "<think>SECRET REASONING</think>\n\nThe answer is 42."},
        ],
        "_chunk_id": "abc123",
        "_judge_prompt_version": "gepa_strict_v2",
        "_question_type": "Calculation/Numerical",
    }


def test_strip_think_removes_block_and_surrounding_whitespace():
    assert strip_think("<think>reasoning</think>\n\nFinal.") == "Final."


def test_strip_think_leaves_text_without_a_think_block_alone():
    assert strip_think("Just an answer.") == "Just an answer."


def test_item_drops_think_and_all_metadata():
    item = to_scorer_item(_record(), 7)

    assert item["messages"][2]["content"] == "The answer is 42."
    assert set(item) == {"messages", "_orig", "_idx"}
    assert item["_idx"] == 7


def test_payload_is_messages_only_with_role_and_content_only():
    payload = scorer_payload(to_scorer_item(_record(), 0))

    assert set(payload) == {"messages"}
    for message in payload["messages"]:
        assert set(message) == {"role", "content"}


def test_original_record_is_not_mutated_and_stays_reachable():
    record = _record()
    item = to_scorer_item(record, 0)

    assert item["_orig"] is record
    assert "SECRET REASONING" in record["messages"][2]["content"]
    assert record["_chunk_id"] == "abc123"
