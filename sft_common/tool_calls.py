"""Tool calls in the two text formats used by the training data (kept exactly as provided).

    <functioncall> {"name": "get_weather", "arguments": '{"city": "Paris"}'}      (Glaive style;
        arguments are a quoted JSON string, or a plain {} object for no-argument calls)
    <tool_call>[{"name": "f", "arguments": {...}}, {"name": "g", ...}]</tool_call>  (JSON list)

The model is trained on these, so it answers in these; eval parses both from its output.
"""
import ast
import json
import re

FUNCTIONCALL = "<functioncall>"
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"

_HEAD = re.compile(r'\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*', re.DOTALL)  # up to the args value
_QUOTED_END = re.compile(r"'\s*\}")                                                      # closes '...args...'}
_DECODER = json.JSONDecoder()


def has_call(text: str) -> bool:
    """True if the text tries to make a tool call in either format."""
    return FUNCTIONCALL in text or TOOL_CALL_OPEN in text


def _load_args(raw: str):
    """Decode an arguments string: JSON first, then Python-literal syntax (Glaive sometimes uses it)."""
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw)
        except (ValueError, SyntaxError):
            continue
    return None


def _parse_functioncall(segment: str) -> dict | None:
    """One `<functioncall>` body -> {"name", "arguments"}; None if malformed."""
    m = _HEAD.match(segment)
    if not m:
        return None
    rest = segment[m.end():]
    if rest.startswith("'"):                                  # quoted JSON string: '{...}'
        ends = list(_QUOTED_END.finditer(rest))
        if not ends:
            return None
        args = _load_args(rest[1:ends[-1].start()])
    else:                                                     # plain JSON object: {...}
        try:
            args, _ = _DECODER.raw_decode(rest)
        except ValueError:
            return None
    return {"name": m.group(1), "arguments": args} if isinstance(args, dict) else None


def _parse_tool_call_block(body: str) -> list[dict] | None:
    """One `<tool_call>...</tool_call>` body (a JSON list or a single object) -> calls; None if malformed."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    items = data if isinstance(data, list) else [data]
    calls = []
    for c in items:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            return None
        args = c.get("arguments", {})
        if not isinstance(args, dict):
            return None
        calls.append({"name": c["name"], "arguments": args})
    return calls


def parse_calls(text: str) -> list[dict] | None:
    """All tool calls in `text`, in order. [] if there are none; None if any call is malformed."""
    calls: list[dict] = []
    # <tool_call> blocks: every opening tag needs a closing one.
    if text.count(TOOL_CALL_OPEN) != text.count(TOOL_CALL_CLOSE):
        return None
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        block = _parse_tool_call_block(body.strip())
        if block is None:
            return None
        calls += block
    # <functioncall> segments: each runs until the next marker (or the end of the text).
    for segment in text.split(FUNCTIONCALL)[1:]:
        call = _parse_functioncall(segment.split(TOOL_CALL_OPEN)[0])
        if call is None:
            return None
        calls.append(call)
    return calls


def _norm(v) -> str:
    """Canonical string for an argument value, so 3 == "3", True == "true", key order ignored."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    s = str(v).strip()
    if s[:1] in "[{":
        try:
            return json.dumps(json.loads(s), sort_keys=True, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return {"True": "true", "False": "false"}.get(s, s)


def args_equal(a: dict, b: dict) -> bool:
    return {k: _norm(v) for k, v in a.items()} == {k: _norm(v) for k, v in b.items()}


def calls_equal(got: list[dict], want: list[dict]) -> bool:
    """Same calls in the same order, with the same arguments."""
    return len(got) == len(want) and all(
        g["name"] == w["name"] and args_equal(g["arguments"], w["arguments"]) for g, w in zip(got, want))
