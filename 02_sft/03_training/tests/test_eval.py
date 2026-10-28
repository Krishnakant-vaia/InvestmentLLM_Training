"""Step 4 pieces: tool-call parsing (both data formats), scoring, and the judge (no network)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_common import judge as J
from sft_common import metrics
from sft_common.tool_calls import calls_equal, has_call, parse_calls

GLAIVE_CALL = '<functioncall> {"name": "track_calories", "arguments": \'{"food_items": ["banana", "apple"], "calories_per_item": [105, 95]}\'}'
TOOL_CALL = '<tool_call>[{"name": "get_zip_info", "arguments": {"zipcode": "90210"}}, {"name": "get_zip_info", "arguments": {"zipcode": "10001"}}]</tool_call>'


# ---- tool-call parsing -------------------------------------------------------------

def test_parse_glaive_functioncall():
    assert parse_calls(GLAIVE_CALL) == [{"name": "track_calories",
                                         "arguments": {"food_items": ["banana", "apple"], "calories_per_item": [105, 95]}}]
    assert parse_calls('<functioncall> {"name": "get_news_headlines", "arguments": {}}') == [
        {"name": "get_news_headlines", "arguments": {}}]                         # plain {} arguments


def test_parse_tool_call_list():
    assert [c["arguments"]["zipcode"] for c in parse_calls(TOOL_CALL)] == ["90210", "10001"]


def test_parse_no_call_and_malformed():
    assert parse_calls("The median is 5.") == [] and not has_call("The median is 5.")
    assert parse_calls('<functioncall> {"name": "f", "arguments": \'{broken\'}') is None
    assert parse_calls('<functioncall> {oops') is None
    assert parse_calls("<tool_call>[{\"name\": \"f\"") is None                     # unclosed


def test_calls_equal_ignores_number_rendering_and_key_order():
    got = [{"name": "f", "arguments": {"n": "3", "o": {"b": 1, "a": 2}}}]
    want = [{"name": "f", "arguments": {"n": 3, "o": {"a": 2, "b": 1}}}]
    assert calls_equal(got, want)
    assert not calls_equal(got, [{"name": "g", "arguments": want[0]["arguments"]}])


# ---- scoring -----------------------------------------------------------------------

TOOL_ROW = {"source": "tool_calling", "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to the following functions. {...}"},
    {"role": "user", "content": "Track my calories"},
    {"role": "assistant", "content": "Sure, what did you eat?"},
    {"role": "user", "content": "A banana and an apple, 105 and 95."},
    {"role": "assistant", "content": GLAIVE_CALL},
    {"role": "user", "content": '{"total_calories": 200}'},
    {"role": "assistant", "content": "You had 200 calories."},
]}


def test_target_is_first_call_turn_not_clarifying_question():
    prompt, ref = metrics.pick_target(TOOL_ROW)
    assert ref["content"] == GLAIVE_CALL and prompt[-1]["content"].startswith("A banana")


def test_score_correct_and_wrong_calls():
    good = metrics.score(TOOL_ROW, GLAIVE_CALL)
    assert good["call_made"] and good["call_parses"] and good["name_match"] and good["args_match"]
    wrong = metrics.score(TOOL_ROW, '<functioncall> {"name": "track_calories", "arguments": \'{"food_items": []}\'}')
    assert wrong["name_match"] and not wrong["args_match"]
    none = metrics.score(TOOL_ROW, "Please tell me more.")
    assert not none["call_made"] and not none["name_match"]


def test_score_domain_row_with_reasoning_and_truncation():
    row = {"source": "domain", "messages": [{"role": "user", "content": "Q"},
                                            {"role": "assistant", "content": "The answer is 30%", "reasoning": "r"}]}
    s = metrics.score(row, "I compute 9000*0.3.\n</think>\n\nThe answer is 30%", "stop")   # thinking-prompt output
    assert s["f1"] == 1.0 and s["complete"] and s["clean"] and not s["false_call"]
    cut = metrics.score(row, "I compute", "length")                                        # hit the token limit
    assert not cut["complete"]
    assert not metrics.score(row, "The answer<|im_end|> is", "stop")["clean"]


def test_aggregate_skips_missing_judge_scores():
    agg = metrics.aggregate([{"f1": 1.0, "judge_correctness": 4}, {"f1": 0.0, "judge_correctness": None}])
    assert agg["f1"] == 0.5 and agg["judge_correctness"] == 4.0 and agg["n"] == 2


# ---- judge (fake HTTP; never calls Cloudflare) ---------------------------------------

class FakeResp:
    def __init__(self, status, content="", text=""):
        self.status_code, self._content, self.text = status, content, text

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def make_judge(responses):
    calls = []

    def post(url, headers, json, timeout):  # noqa: A002
        calls.append({"url": url, "headers": headers, "json": json})
        return responses.pop(0)
    return J.CloudflareJudge("acct", "tok", "@cf/openai/gpt-oss-120b", wait=0, post=post), calls


def test_parse_verdict_finds_last_valid_json():
    text = 'thinking... {"x": 1} final: {"correctness": 4, "reasoning": 5, "reason": "ok"}'
    assert J.parse_verdict(text) == {"correctness": 4, "reasoning": 5, "reason": "ok"}
    assert J.parse_verdict('{"correctness": 9, "reasoning": 1}') is None           # out of range
    assert J.parse_verdict("no json") is None


def test_judge_request_shape_and_success():
    judge, calls = make_judge([FakeResp(200, '{"correctness": 5, "reasoning": 4, "reason": "matches"}')])
    v = judge.score("What is X?", "X is 30%", "30%")
    assert v["correctness"] == 5 and "error" not in v
    c = calls[0]
    assert c["url"].endswith("/accounts/acct/ai/v1/chat/completions")
    assert c["headers"]["Authorization"] == "Bearer tok" and c["json"]["model"] == "@cf/openai/gpt-oss-120b"
    assert "X is 30%" in c["json"]["messages"][0]["content"]


def test_judge_retries_rate_limit_then_succeeds():
    judge, calls = make_judge([FakeResp(429), FakeResp(200, '{"correctness": 3, "reasoning": 3}')])
    assert judge.score("q", "r", "a")["correctness"] == 3 and len(calls) == 2


def test_judge_errors_are_reported_not_raised():
    judge, _ = make_judge([FakeResp(401, text="bad token")])
    assert judge.score("q", "r", "a")["error"].startswith("HTTP 401")
    judge, _ = make_judge([FakeResp(200, "no verdict")] * 4)
    assert "unparseable" in judge.score("q", "r", "a")["error"]


def test_judge_many_keeps_order():
    judge, _ = make_judge([FakeResp(200, json.dumps({"correctness": k, "reasoning": k})) for k in (1, 2, 3)])
    items = [{"question": "q", "reference": "r", "answer": "a"}] * 3
    out = J.judge_many(judge, items, workers=1)
    assert [v["correctness"] for v in out] == [1, 2, 3]
