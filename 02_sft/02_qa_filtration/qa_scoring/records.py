import re

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def to_scorer_item(record: dict, index: int) -> dict:
    """What the scorer sees: role/content only, no reasoning trace, no metadata.
    The untouched original rides along in _orig so the output keeps every field."""
    messages = []
    for m in record["messages"]:
        content = strip_think(m["content"]) if m["role"] == "assistant" else m["content"]
        messages.append({"role": m["role"], "content": content})
    return {"messages": messages, "_orig": record, "_idx": index}


def scorer_payload(item: dict) -> dict:
    return {"messages": [{"role": m["role"], "content": m["content"]} for m in item["messages"]]}
