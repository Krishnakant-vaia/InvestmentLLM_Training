import json
import threading
import time

import requests

from qa_filter.config import JUDGE_PROMPT_TEMPLATE, Settings
from qa_filter.prefilter import extract_final_answer


class JudgeCallError(Exception):
    pass


# One pooled Session shared across every worker thread. A bare requests.post()
# per call opens a brand-new TCP+TLS connection each time (urllib3's default
# pool caps at 10 connections per host regardless) -- at real concurrency
# (confirmed against the account: 300 concurrent bare-post calls collapsed to a
# ~0.4% success rate, almost all connection-level errors, not HTTP responses at
# all) that connection churn overwhelms the local machine/edge. A single Session
# with its HTTPAdapter pool sized to the actual worker count reuses connections
# instead of one-shotting a fresh handshake per call. requests.Session is safe
# for concurrent use from multiple threads (the underlying urllib3 pool manager
# is thread-safe); lazily built once and cached, sized to the largest
# max_workers seen so far, since Settings isn't available at import time.
_session_lock = threading.Lock()
_session: requests.Session | None = None
_session_pool_size = 0


def _get_session(max_workers: int) -> requests.Session:
    global _session, _session_pool_size
    with _session_lock:
        if _session is None or max_workers > _session_pool_size:
            pool_size = max(max_workers, 10)
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _session = session
            _session_pool_size = pool_size
        return _session


def build_judge_prompt(batch: list[dict]) -> str:
    items = []
    for record in batch:
        user_content = record["messages"][1]["content"]
        document = user_content.split("<document>", 1)[1].split("</document>", 1)[0].strip()
        question = user_content.split("</document>", 1)[1].strip()
        final_answer = extract_final_answer(record["messages"][2]["content"])
        items.append({
            "id": record["_batch_item_id"],
            "document": document,
            "question": question,
            "final_answer": final_answer,
        })
    batch_records_json = json.dumps(items, indent=2)
    return JUDGE_PROMPT_TEMPLATE.replace("{{batch_records_json}}", batch_records_json)


def _post_once(prompt: str, settings: Settings) -> requests.Response:
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{settings.cloudflare_account_id}/ai/run/{settings.model_id}"
    )
    session = _get_session(settings.max_workers)
    return session.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.cloudflare_api_token}",
            "Content-Type": "application/json",
        },
        json={
            "messages": [
                {"role": "system", "content": "Respond with only the JSON object specified. No other text, no Markdown fences."},
                {"role": "user", "content": prompt},
            ],
            "reasoning": {"effort": settings.reasoning_effort},
            "temperature": 0,
            "max_tokens": settings.max_output_tokens,
        },
        timeout=settings.request_timeout_seconds,
    )


def _extract_content(response: requests.Response) -> str:
    data = response.json()
    result = data.get("result")
    if not isinstance(result, dict):
        raise JudgeCallError(f"unrecognized response shape: {json.dumps(data)[:300]}")

    choices = result.get("choices")
    if choices:
        content = choices[0]["message"].get("content")
        finish_reason = choices[0].get("finish_reason")
        # content can be explicitly null (not merely absent -- .get(..., "") only
        # defaults on a MISSING key) when the model exhausts max_tokens on its own
        # reasoning before emitting the final answer. At temperature=0 a same-size
        # retry would very likely hit the identical truncation again (same input,
        # same greedy decoding), so this deliberately is NOT retried inside
        # call_judge's backoff loop -- it propagates straight out to
        # judge_retry.judge_batch_with_retry's split-on-failure path, since fewer
        # records per batch (shorter expected output) is what actually reduces
        # truncation risk, unlike waiting and asking the same question again.
        if not content:
            print(f"  [judge] null content (finish_reason={finish_reason!r}) -- will split batch and retry", flush=True)
            raise JudgeCallError(f"empty/null content (finish_reason={finish_reason!r})")
        return content
    if "response" in result:
        response_text = result["response"]
        if not response_text:
            raise JudgeCallError("empty/null response field")
        return response_text
    raise JudgeCallError(f"unrecognized response shape: {json.dumps(result)[:300]}")


def call_judge(batch: list[dict], settings: Settings) -> list[dict]:
    parsed = call_model_json(build_judge_prompt(batch), settings, len(batch))
    verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
    if not isinstance(verdicts, list):
        raise JudgeCallError("judge response missing a 'verdicts' array")
    return verdicts


def call_model_json(prompt: str, settings: Settings, n_records: int):
    """Sends one prompt and returns the model's reply parsed as JSON.

    Shared by every caller that asks the model for a JSON reply (the keep/reject
    judge and the quality scorer), so the hard-won failure handling lives in one
    place: 401/403 fatal, 400 permanent (goes straight to the caller's split path),
    429/5xx/request errors retried with backoff, null content raised immediately.
    n_records is only used to make retry log lines say which batch is struggling.
    """
    last_error = None
    for attempt in range(settings.max_retries):
        try:
            response = _post_once(prompt, settings)
        except requests.RequestException as e:
            last_error = JudgeCallError(f"request failed: {e}")
        else:
            if response.status_code in (401, 403):
                raise JudgeCallError(f"HTTP {response.status_code} (auth/permission): {response.text[:300]}")
            if response.status_code == 400:
                # Permanent, not transient -- e.g. "Input length exceeds model's
                # maximum context length" when the chars/4 token estimate (accurate
                # for plain prose, not for every document) undershoots badly for a
                # particular batch. Retrying the identical oversized request with
                # backoff would fail the same way every time; raising immediately
                # sends it straight to judge_retry's split path instead, which
                # actually fixes the problem (a smaller batch has a smaller prompt).
                raise JudgeCallError(f"HTTP 400 (permanent, not retried): {response.text[:300]}")
            if response.status_code == 429:
                last_error = JudgeCallError(f"HTTP 429 rate limited: {response.text[:200]}")
            elif response.status_code != 200:
                last_error = JudgeCallError(f"HTTP {response.status_code}: {response.text[:500]}")
            else:
                content = _extract_content(response)  # raises JudgeCallError straight out on null/malformed content -- not caught here, see its docstring note
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise JudgeCallError(f"could not parse judge response as JSON: {e}") from e

        # Visible in the run log (unlike a swallowed exception) so a slow shard's
        # cause -- rate limiting vs. something else -- is diagnosable without
        # guessing, which real production runs have needed. Only reached for
        # request failures / 429 / non-200s, since a null-content JudgeCallError
        # from _extract_content propagates straight out above, uncaught here.
        if attempt < settings.max_retries - 1:
            wait = settings.retry_base_seconds * (2 ** attempt)
            print(f"  [judge retry] batch of {n_records}: attempt {attempt + 1}/{settings.max_retries} failed ({last_error}), retrying in {wait}s", flush=True)
            time.sleep(wait)

    raise last_error
