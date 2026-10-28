"""Pure scoring functions for step 4 (no model or GPU needed).

Each eval row is cut into a prompt and one reference assistant turn (pick_target). The model's
answer to that prompt is scored against the reference (score); scores are averaged per source.
"""
import re
from collections import Counter

from .chat_format import split_think
from .tool_calls import calls_equal, has_call, parse_calls

_WORD = re.compile(r"\w+")


def pick_target(row: dict) -> tuple[list[dict], dict]:
    """(prompt_messages, reference_assistant_turn) for an eval row.

    If the conversation makes a tool call, the target is the FIRST assistant turn with a call:
    that is the decision worth testing (earlier turns are often just clarifying questions).
    Otherwise it is the last assistant turn (the final answer).
    """
    msgs = row["messages"]
    assistant = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    calls = [i for i in assistant if has_call(msgs[i]["content"])]
    i = calls[0] if calls else assistant[-1]
    return msgs[:i], msgs[i]


def token_f1(pred: str, ref: str) -> float:
    """Word-overlap F1 between answer and reference. Rough: a correct paraphrase scores low."""
    p, r = Counter(_WORD.findall(pred.lower())), Counter(_WORD.findall(ref.lower()))
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def score(row: dict, output: str, finish_reason: str = "stop") -> dict:
    """Score one generation against its reference. Every value is a bool or a float.

    Always:   non_empty (answered at all), complete (stopped on its own, not cut at the token
              limit), clean (no chat-template tokens leaked, at most one think block).
    Call rows (reference makes a tool call):
              call_made, call_parses, name_match (same functions, same order), args_match.
    Other rows:
              false_call (called a tool when the reference did not), f1 (vs the reference).
    """
    _, ref = pick_target(row)
    _, answer = split_think(output)
    res = {
        "non_empty": bool(answer),
        "complete": finish_reason == "stop",
        "clean": "<|im_" not in output and output.count("<think>") <= 1 and output.count("</think>") <= 1,
    }
    want = parse_calls(ref["content"]) or []          # reference calls (the data is well-formed)
    if want:
        got = parse_calls(answer)                     # None = the model's call is malformed
        res["call_made"] = has_call(answer)
        res["call_parses"] = res["call_made"] and got is not None
        res["name_match"] = bool(got) and [c["name"] for c in got] == [c["name"] for c in want]
        res["args_match"] = bool(got) and calls_equal(got, want)
    else:
        res["false_call"] = has_call(answer)
        res["f1"] = token_f1(answer, ref["content"])
    return res


def aggregate(scores: list[dict]) -> dict:
    """Mean of every metric over the rows that have it (bools count as 0/1); also reports n."""
    keys = {k for s in scores for k in s}
    out = {}
    for k in sorted(keys):
        vals = [float(s[k]) for s in scores if k in s and s[k] is not None]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    out["n"] = len(scores)
    return out
