"""The one place that knows the chat format.

Rendering is delegated to the model's own chat template (via its tokenizer), so training text
always matches what vLLM / transformers produce at inference. Nothing here hardcodes tokens
except the two loss-mask markers below.

Message schema used everywhere in this repo (rows are used as provided, so tool calls are
plain text inside `content`; see tool_calls.py):
    {"role": "system"|"user"|"assistant",
     "content": str,
     "reasoning": str}   # optional, assistant only: rendered inside <think>...</think>

Qwen3.5 template facts (verified on the pre-tuned model):
  * assistant turns after the last user message carry `<think>\\n{reasoning}\\n</think>\\n\\n`;
    earlier assistant turns have no think block. Empty reasoning renders an empty think block.
"""
import re

# Loss-mask markers for train_on_responses_only (ChatML turn starts).
USER_PART = "<|im_start|>user\n"
ASSISTANT_PART = "<|im_start|>assistant\n"

_THINK_RE = re.compile(r"^\s*<think>\n?(.*?)\n?</think>\s*", re.DOTALL)


class ChatFormat:
    """Renders conversations with a tokenizer's chat template."""

    def __init__(self, tokenizer):
        self.tok = getattr(tokenizer, "tokenizer", tokenizer)  # unwrap processors

    @staticmethod
    def _to_template(messages: list[dict]) -> list[dict]:
        """Our schema -> the template's: `reasoning` becomes `reasoning_content`."""
        out = []
        for m in messages:
            t = {"role": m["role"], "content": m.get("content") or ""}
            if m["role"] == "assistant":
                t["reasoning_content"] = (m.get("reasoning") or "").strip()
            out.append(t)
        return out

    def render(self, messages: list[dict]) -> str:
        """Full training text for a conversation that ends on an assistant turn."""
        return self.tok.apply_chat_template(self._to_template(messages), tokenize=False)

    def prompt(self, messages: list[dict], thinking: bool = False) -> str:
        """Text up to the point where the model starts answering (eval / inference).

        thinking=False ends with an empty think block, so the model answers directly.
        thinking=True ends with an open `<think>`, so the model reasons first; its output then
        contains the reasoning, `</think>`, and the answer (see split_think).
        """
        return self.tok.apply_chat_template(self._to_template(messages), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=thinking)


def split_think(text: str) -> tuple[str, str]:
    """Split model output into (reasoning, answer). Reasoning is '' if there is none.

    Also handles output generated after a thinking prompt, which starts inside the think block
    (no opening `<think>`, only the closing `</think>`).
    """
    if "</think>" in text and "<think>" not in text:
        text = "<think>\n" + text
    m = _THINK_RE.match(text)
    return (m.group(1).strip(), text[m.end():].strip()) if m else ("", text.strip())
