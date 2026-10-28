from qa_filter.config import Settings
from qa_filter.judge_client import JudgeCallError, call_judge


def validate_verdicts(batch: list[dict], verdicts: list[dict]) -> bool:
    sent_ids = [r["_batch_item_id"] for r in batch]
    returned_ids = [v.get("id") for v in verdicts]
    return sorted(sent_ids) == sorted(returned_ids) and len(sent_ids) == len(set(sent_ids))


def judge_batch_with_retry(
    batch: list[dict],
    settings: Settings,
    judge_fn=call_judge,
    same_size_retries: int = 2,
) -> tuple[list[dict], list[dict]]:
    # A bad verdict count/null content turned out (confirmed against the real
    # account) to be largely intermittent, not deterministic, despite
    # temperature=0 -- GPT-OSS-120B is MoE-routed, and routing/inference can
    # vary run to run even at temp 0. A fresh same-size call frequently just
    # succeeds outright, so retrying the same batch a couple of times before
    # falling back to a split avoids multiplying API calls for something that
    # would have resolved itself; splitting is the fallback once the batch has
    # genuinely proven itself unreliable, not the first response to any hiccup.
    split_reason = None
    for attempt in range(same_size_retries):
        try:
            verdicts = judge_fn(batch, settings)
            if validate_verdicts(batch, verdicts):
                return verdicts, []
            split_reason = f"id mismatch (sent {len(batch)}, got {len(verdicts)} verdicts)"
        except JudgeCallError as e:
            split_reason = str(e)

        if attempt < same_size_retries - 1:
            print(f"  [judge] batch of {len(batch)} failed ({split_reason}), retrying same size (attempt {attempt + 2}/{same_size_retries})", flush=True)

    if len(batch) == 1:
        print(f"  [judge] giving up on a solo record after failure: {split_reason}", flush=True)
        return [], list(batch)

    mid = len(batch) // 2
    print(f"  [judge] splitting batch of {len(batch)} -> {mid}+{len(batch) - mid} after {same_size_retries} same-size attempts, last failure: {split_reason}", flush=True)
    left_verdicts, left_failed = judge_batch_with_retry(batch[:mid], settings, judge_fn, same_size_retries)
    right_verdicts, right_failed = judge_batch_with_retry(batch[mid:], settings, judge_fn, same_size_retries)

    return left_verdicts + right_verdicts, left_failed + right_failed
