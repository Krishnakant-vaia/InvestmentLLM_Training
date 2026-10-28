"""LLM judge for domain answers: gpt-oss-120b on Cloudflare Workers AI.

Word-overlap F1 punishes correct answers that are worded differently from the reference, so
a strong model reads the question, the reference and the candidate answer and scores:
    correctness 1-5  facts, numbers and conclusion agree with the reference
    reasoning   1-5  the explanation / calculation is sound and clear

Caveat: the domain references were written by this same model, so it may slightly favour
answers that sound like its own. Use the scores to compare models (pre-tuned vs fine-tuned),
not as absolute truth.

Credentials (.env): CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN.
Endpoint: Cloudflare's OpenAI-compatible chat completions API.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"

PROMPT = """You are grading an answer to a finance and investment question against a reference answer.

## Question (may include a source document)
{question}

## Reference answer
{reference}

## Candidate answer
{answer}

## Scoring
- correctness (1-5): do the candidate's facts, numbers and conclusion agree with the reference?
  5 = fully correct, 3 = partly correct or missing key points, 1 = wrong or no answer.
- reasoning (1-5): is the candidate's explanation or calculation sound and clear?
  5 = clear and valid, 3 = thin or partly flawed, 1 = absent or wrong.
Judge substance, not wording: a correct answer phrased differently from the reference is still correct.

Reply with only this JSON: {{"correctness": <1-5>, "reasoning": <1-5>, "reason": "<one sentence>"}}"""

_SCORE_KEYS = ("correctness", "reasoning")


def build_messages(question: str, reference: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": PROMPT.format(question=question, reference=reference,
                                                        answer=answer or "(no answer)")}]


def parse_verdict(text: str) -> dict | None:
    """Last JSON object in the reply that has both scores as integers 1-5; None if there is none.

    No JSON mode is documented for this model, so the reply is searched rather than trusted.
    """
    for raw in reversed(re.findall(r"\{[^{}]*\}", text or "")):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        try:
            scores = {k: int(obj[k]) for k in _SCORE_KEYS}
        except (KeyError, TypeError, ValueError):
            continue
        if all(1 <= v <= 5 for v in scores.values()):
            return {**scores, "reason": str(obj.get("reason", ""))[:500]}
    return None


def _requests_post(url, headers, json, timeout):  # noqa: A002 - mirrors requests.post's keywords
    import requests
    return requests.post(url, headers=headers, json=json, timeout=timeout)


class CloudflareJudge:
    """Scores one answer per call. Never raises: failures come back as {"error": ...}."""

    def __init__(self, account_id: str, api_token: str, model: str, max_tokens: int = 2048,
                 retries: int = 4, wait: float = 5.0, post=None):
        self.url = API_URL.format(account_id=account_id)
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        self.model, self.max_tokens = model, max_tokens   # room for gpt-oss's own reasoning + the JSON
        self.retries, self.wait = retries, wait
        self.post = post or _requests_post                # injectable, so tests never hit the network

    @classmethod
    def from_env(cls, model: str, max_tokens: int = 2048) -> "CloudflareJudge":
        """Build from .env; SystemExit with a clear message if credentials are missing."""
        account, token = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account or not token:
            raise SystemExit("Judge needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env "
                             "(or run evaluate.py with --no-judge).")
        return cls(account, token, model, max_tokens)

    def score(self, question: str, reference: str, answer: str) -> dict:
        body = {"model": self.model, "messages": build_messages(question, reference, answer),
                "max_tokens": self.max_tokens, "temperature": 0}
        error = "no attempt"
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.post(self.url, headers=self.headers, json=body, timeout=180)
            except Exception as e:  # noqa: BLE001 - network errors are retried
                error = f"request failed: {e}"
            else:
                if resp.status_code == 429 or resp.status_code >= 500:   # rate limit / server error: retry
                    error = f"HTTP {resp.status_code}"
                elif resp.status_code != 200:                             # bad request / auth: give up
                    return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
                else:
                    try:
                        content = resp.json()["choices"][0]["message"].get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError):
                        content = ""
                    verdict = parse_verdict(content)
                    if verdict:
                        return verdict
                    error = f"unparseable reply: {content[:200]!r}"      # retried: sampling may differ
            if attempt < self.retries:
                time.sleep(self.wait * attempt)
        return {"error": error}


def judge_many(judge: CloudflareJudge, items: list[dict], workers: int = 8) -> list[dict]:
    """Score items ({"question", "reference", "answer"}) in parallel; results keep input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda it: judge.score(it["question"], it["reference"], it["answer"]), items))
