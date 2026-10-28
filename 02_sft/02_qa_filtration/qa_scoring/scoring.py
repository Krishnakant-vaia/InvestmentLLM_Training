from qa_filter.config import Settings
from qa_filter.judge_client import JudgeCallError
from qa_scoring.prompt import CATEGORY_CODES, FAILURE_CODES
from qa_scoring.score_client import call_scorer

PASS_REASON = "PASS_ALL_GATES"
PASS_THRESHOLD = 7


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_scores(n: int, scores: list[dict]) -> bool:
    """Hard check: exactly positions 0..n-1 once each, integer score in 1..10."""
    if len(scores) != n:
        return False
    ids = []
    for s in scores:
        if not isinstance(s, dict) or not _is_int(s.get("id")) or not _is_int(s.get("score")):
            return False
        if not 1 <= s["score"] <= 10:
            return False
        ids.append(s["id"])
    return sorted(ids) == list(range(n))


def clean_categories(categories) -> list[str]:
    if not isinstance(categories, list):
        return []
    seen: list[str] = []
    for c in categories:
        if c in CATEGORY_CODES and c not in seen:
            seen.append(c)
    return seen


def score_anomalies(s: dict) -> list[str]:
    """Soft checks: recorded on the record, never trigger a retry."""
    anomalies = []
    cats = s.get("categories")
    if not (isinstance(cats, list) and 1 <= len(cats) <= 3 and len(set(cats)) == len(cats)
            and all(c in CATEGORY_CODES for c in cats)):
        anomalies.append("bad_categories")
    reason = s.get("reason")
    ok = reason == PASS_REASON if s["score"] >= PASS_THRESHOLD else reason in FAILURE_CODES
    if not ok:
        anomalies.append("reason_mismatch")
    return anomalies


def score_batch_with_retry(batch, settings: Settings, score_fn=call_scorer, same_size_retries: int = 2):
    # Same recovery ladder as the filter (intermittent MoE nondeterminism): retry the
    # same size, then halve. Positions are re-based on every call because the model
    # numbers records 0..n-1 within the batch it was actually sent.
    reason = None
    for attempt in range(same_size_retries):
        try:
            scores = score_fn(batch, settings)
            if validate_scores(len(batch), scores):
                by_pos = {s["id"]: s for s in scores}
                results = []
                for pos, item in enumerate(batch):
                    s = by_pos[pos]
                    results.append({
                        "item": item,
                        "score": s["score"],
                        "categories": clean_categories(s.get("categories")),
                        "reason": s.get("reason"),
                        "anomalies": score_anomalies(s),
                    })
                return results, []
            reason = f"invalid scores (sent {len(batch)}, got {len(scores)})"
        except JudgeCallError as e:
            reason = str(e)
        if attempt < same_size_retries - 1:
            print(f"  [score] batch of {len(batch)} failed ({reason}), retrying same size", flush=True)

    if len(batch) == 1:
        print(f"  [score] giving up on a solo record: {reason}", flush=True)
        return [], list(batch)

    mid = len(batch) // 2
    print(f"  [score] splitting batch of {len(batch)} -> {mid}+{len(batch) - mid}: {reason}", flush=True)
    lr, lf = score_batch_with_retry(batch[:mid], settings, score_fn, same_size_retries)
    rr, rf = score_batch_with_retry(batch[mid:], settings, score_fn, same_size_retries)
    return lr + rr, lf + rf
