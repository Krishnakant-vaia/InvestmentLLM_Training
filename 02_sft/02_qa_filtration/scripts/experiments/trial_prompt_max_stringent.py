"""Isolated test: pulls 1000 RAW (never-before-judged) records directly from
S3, starting right after the ranges already used by the real pipeline (shards
0-95, records 0-479999) and the earlier shard-96 test (480000-484999) --
records 485000-485999 are guaranteed untouched by anything so far. Runs them
through the real pre-filter + the new max-stringent prompt, reports the
pass/reject rate. Writes nothing to run_state/ or S3.

Usage: python scripts/test_prompt_max_stringent.py
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
from qa_filter.prefilter import PreFilter
from qa_filter.s3_io import stream_jsonl_lines
from qa_filter.worker_pool import run_batches_concurrently

TEST_WORKERS = 10
PROMPT_PATH = sys.argv[1] if len(sys.argv) > 1 else "prompts/qa_judge_prompt_max_stringent.txt"  # that prompt was never checked in; pass its path
START_OFFSET = 485000
COUNT = 1000


def main():
    settings = load_settings()
    settings = dataclasses.replace(settings, max_workers=TEST_WORKERS)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        judge_client_module.JUDGE_PROMPT_TEMPLATE = f.read()
    print(f"loaded prompt from {PROMPT_PATH}, testing with MAX_WORKERS={TEST_WORKERS}", flush=True)

    end = START_OFFSET + COUNT
    print(f"pulling raw records {START_OFFSET}-{end - 1} from S3, never judged before...", flush=True)

    records = []
    for i, line in enumerate(stream_jsonl_lines(settings.s3_bucket, settings.source_key)):
        if i < START_OFFSET:
            continue
        if i >= end:
            break
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"pulled {len(records)} raw records", flush=True)

    kept, rejected = PreFilter().filter(records)
    print(f"Stage 0 pre-filter: kept={len(kept)} rejected={len(rejected)}", flush=True)

    for i, r in enumerate(kept):
        r["_batch_item_id"] = i
    by_id = {r["_batch_item_id"]: r for r in kept}

    batches = build_batches(kept, settings.batch_token_ceiling, settings.max_batch_records)
    print(f"packed into {len(batches)} batches, judging with max-stringent prompt...", flush=True)

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

    print("\nsample keeps:")
    keep_samples = [v for v in verdicts if v.get("verdict") == "keep"][:5]
    for v in keep_samples:
        q = by_id[v["id"]]["messages"][1]["content"].split("</document>", 1)[-1].strip()[:90]
        print(f"  reason={v.get('reason', '')!r} question={q!r}")


if __name__ == "__main__":
    main()
