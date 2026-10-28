"""One-off test: run the new stricter judge prompt (qa_judge_prompt_run2.txt)
against the first 1000 records that already passed round 1 (run_state/kept.jsonl),
to measure the pass/reject rate under the new rubric before committing to a full
re-filter pass. Read-only -- writes nothing to run_state/ or S3. Uses a modest
worker count since the real pipeline may already be running concurrently against
the same Cloudflare account.
"""
import dataclasses
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import qa_filter.judge_client as judge_client_module
from qa_filter.batching import build_batches
from qa_filter.config import load_settings
from qa_filter.judge_retry import judge_batch_with_retry
from qa_filter.worker_pool import run_batches_concurrently

COUNT = 1000
TEST_WORKERS = 10
PROMPT_PATH = "prompts/qa_judge_prompt_run2.txt"


def main():
    settings = load_settings()
    settings = dataclasses.replace(settings, max_workers=TEST_WORKERS)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        judge_client_module.JUDGE_PROMPT_TEMPLATE = f.read()
    print(f"loaded prompt from {PROMPT_PATH}, testing with MAX_WORKERS={TEST_WORKERS}")

    records = []
    with open("run_state/kept.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) == COUNT:
                break
    print(f"loaded {len(records)} already-passed records from run_state/kept.jsonl")

    for i, r in enumerate(records):
        r["_batch_item_id"] = i
    by_id = {r["_batch_item_id"]: r for r in records}

    batches = build_batches(records, settings.batch_token_ceiling, settings.max_batch_records)
    print(f"packed into {len(batches)} batches, judging...")

    verdicts, permanently_failed = run_batches_concurrently(batches, settings, judge_batch_with_retry)

    keep_count = sum(1 for v in verdicts if v.get("verdict") == "keep")
    reject_count = len(verdicts) - keep_count
    total = len(verdicts) + len(permanently_failed)

    print()
    print("=" * 60)
    print(f"total judged: {total} (of {len(records)} submitted)")
    print(f"keep:  {keep_count} ({100 * keep_count / total:.1f}%)")
    print(f"reject: {reject_count} ({100 * reject_count / total:.1f}%)")
    print(f"call_failed: {len(permanently_failed)}")
    print("=" * 60)

    reason_counts = Counter()
    for v in verdicts:
        if v.get("verdict") == "reject":
            reason_counts[v.get("reason", "")] += 1

    print("\nreject reasons:")
    for reason, count in reason_counts.most_common(30):
        print(f"  {count:4d}  {reason}")

    print("\nsample keeps:")
    keep_samples = [v for v in verdicts if v.get("verdict") == "keep"][:5]
    for v in keep_samples:
        q = by_id[v["id"]]["messages"][1]["content"].split("</document>", 1)[-1].strip()[:90]
        print(f"  reason={v.get('reason', '')!r} question={q!r}")


if __name__ == "__main__":
    main()
