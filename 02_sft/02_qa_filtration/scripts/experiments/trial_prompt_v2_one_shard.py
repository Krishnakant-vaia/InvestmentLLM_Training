"""Isolated test: pulls one shard's worth of RAW (never-before-judged) records
directly from S3, runs them through the real pre-filter + the new prompt
(qa_judge_prompt_run2.txt), and reports the pass/reject rate. Writes nothing to
run_state/ or S3 -- the real pipeline's checkpoint/output are untouched.

Usage: python scripts/test_prompt_v2_one_shard.py [--shard-id 96]
"""
import argparse
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
from qa_filter.prefilter import PreFilter
from qa_filter.s3_io import stream_jsonl_lines
from qa_filter.worker_pool import run_batches_concurrently

TEST_WORKERS = 10
PROMPT_PATH = "prompts/qa_judge_prompt_run2.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=96)
    args = parser.parse_args()

    settings = load_settings()
    settings = dataclasses.replace(settings, max_workers=TEST_WORKERS)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        judge_client_module.JUDGE_PROMPT_TEMPLATE = f.read()
    print(f"loaded prompt from {PROMPT_PATH}, testing with MAX_WORKERS={TEST_WORKERS}")

    start = args.shard_id * settings.shard_size
    end = start + settings.shard_size
    print(f"pulling raw records {start}-{end - 1} (shard {args.shard_id}) from S3, never judged before...")

    records = []
    for i, line in enumerate(stream_jsonl_lines(settings.s3_bucket, settings.source_key)):
        if i < start:
            continue
        if i >= end:
            break
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"pulled {len(records)} raw records")

    kept, rejected = PreFilter().filter(records)
    print(f"Stage 0 pre-filter: kept={len(kept)} rejected={len(rejected)}")

    for i, r in enumerate(kept):
        r["_batch_item_id"] = i
    by_id = {r["_batch_item_id"]: r for r in kept}

    batches = build_batches(kept, settings.batch_token_ceiling, settings.max_batch_records)
    print(f"packed into {len(batches)} batches, judging with new prompt...")

    verdicts, permanently_failed = run_batches_concurrently(batches, settings, judge_batch_with_retry)

    keep_count = sum(1 for v in verdicts if v.get("verdict") == "keep")
    judge_reject_count = len(verdicts) - keep_count
    total_raw = len(records)
    total_judged = len(verdicts) + len(permanently_failed)

    print()
    print("=" * 60)
    print(f"raw records pulled: {total_raw}")
    print(f"  prefilter rejected: {len(rejected)} ({100 * len(rejected) / total_raw:.1f}%)")
    print(f"  sent to judge: {len(kept)}")
    print(f"judge results (of {total_judged} judged):")
    print(f"  keep:   {keep_count} ({100 * keep_count / total_judged:.1f}%)")
    print(f"  reject: {judge_reject_count} ({100 * judge_reject_count / total_judged:.1f}%)")
    print(f"  call_failed: {len(permanently_failed)}")
    print(f"OVERALL pass rate (keep / total raw): {100 * keep_count / total_raw:.1f}%")
    print(f"OVERALL reject rate (everything else / total raw): {100 * (total_raw - keep_count) / total_raw:.1f}%")
    print("=" * 60)

    reason_counts = Counter()
    for v in verdicts:
        if v.get("verdict") == "reject":
            reason_counts[v.get("reason", "")] += 1
    for r in rejected:
        reason_counts[f"prefilter: {r['_reject_reason']}"] += 1

    print("\ntop reject reasons:")
    for reason, count in reason_counts.most_common(20):
        print(f"  {count:4d}  {reason}")


if __name__ == "__main__":
    main()
