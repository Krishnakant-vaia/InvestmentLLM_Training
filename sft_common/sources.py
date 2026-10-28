"""Adapter: raw pool row -> list of messages (see chat_format for the schema).

The pool is used as provided: every row is {"messages": [system?, user, assistant, ...]}
and its text is kept verbatim (tool calls stay as the plain text they were written in).
Returns None for a row it cannot use, so callers just skip Nones.
"""
import re

_INLINE_THINK = re.compile(r"\s*<think>\n?(.*?)\n?</think>\s*(.*)", re.DOTALL)


def _assistant(content: str) -> dict:
    """Assistant turn; an inline <think>..</think> prefix is moved into `reasoning`.

    The chat template adds the think block itself, so leaving tags in `content`
    would duplicate them.
    """
    reasoning = ""
    m = _INLINE_THINK.match(content)
    if m:
        reasoning, content = m.group(1), m.group(2)
    return {"role": "assistant", "content": content.strip(), "reasoning": reasoning.strip()}


def pool_row(row: dict) -> list[dict] | None:
    msgs = []
    for m in row.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        msgs.append(_assistant(content) if role == "assistant" else {"role": role, "content": content})
    # Trailing user turns have no answer to learn from; trim them rather than lose the row.
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    if not any(m["role"] == "user" for m in msgs):
        return None
    return msgs
