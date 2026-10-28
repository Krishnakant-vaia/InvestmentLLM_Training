import json

from qa_filter.config import Settings
from qa_filter.judge_client import JudgeCallError, call_model_json
from qa_scoring.prompt import PLACEHOLDER, load_prompt
from qa_scoring.records import scorer_payload


def build_scorer_prompt(template: str, batch: list[dict]) -> str:
    # ensure_ascii=False: raw non-ASCII (e.g. rupee sign) is fewer tokens than \uXXXX escapes.
    lines = [json.dumps(scorer_payload(item), ensure_ascii=False) for item in batch]
    return template.replace(PLACEHOLDER, "\n".join(lines))


def call_scorer(batch: list[dict], settings: Settings, template: str | None = None) -> list[dict]:
    if template is None:
        template, _ = load_prompt()
    parsed = call_model_json(build_scorer_prompt(template, batch), settings, len(batch))
    scores = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores, list):
        raise JudgeCallError("scorer response missing a 'scores' array")
    return scores
