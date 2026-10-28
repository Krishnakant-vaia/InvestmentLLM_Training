"""Judge a few hundred real records and print every verdict for manual spot-check,
before committing all ~873K records to an unproven prompt/model/batching
combination. Writes nothing to S3 -- this is read-only against the source file.

Usage: python scripts/run_smoke_test.py [--count 300]
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
# Real document/question text contains Unicode punctuation (non-breaking hyphens,
# curly quotes) that Windows' default cp1252 console encoding can't print -- force
# UTF-8 so a smoke-test print doesn't crash the whole run over a display detail.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qa_filter.batching import build_batches
from qa_filter.config import load_settings
from qa_filter.judge_retry import judge_batch_with_retry
from qa_filter.prefilter import PreFilter
from qa_filter.s3_io import stream_jsonl_lines
from qa_filter.worker_pool import run_batches_concurrently


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()

    settings = load_settings()

    records = []
    for line in stream_jsonl_lines(settings.s3_bucket, settings.source_key):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(records) == args.count:
            break

    print(f"pulled {len(records)} real records from s3://{settings.s3_bucket}/{settings.source_key}")

    kept, rejected = PreFilter().filter(records)
    print(f"Stage 0 pre-filter: kept={len(kept)} rejected={len(rejected)}")
    for record in rejected:
        print(f"  PREFILTER REJECT reason={record['_reject_reason']!r} question={record['messages'][1]['content'][-80:]!r}")

    for i, record in enumerate(kept):
        record["_batch_item_id"] = i
    by_id = {r["_batch_item_id"]: r for r in kept}

    batches = build_batches(kept, settings.batch_token_ceiling, settings.max_batch_records)
    print(f"Stage 1: {len(kept)} records packed into {len(batches)} batches, judging with MAX_WORKERS={settings.max_workers} ...")

    verdicts, permanently_failed = run_batches_concurrently(batches, settings, judge_batch_with_retry)

    keep_count = sum(1 for v in verdicts if v.get("verdict") == "keep")
    print(f"\njudge verdicts: keep={keep_count} reject={len(verdicts) - keep_count} call_failed={len(permanently_failed)}\n")

    for verdict in verdicts:
        record = by_id[verdict["id"]]
        question = record["messages"][1]["content"].split("</document>", 1)[-1].strip()[:100]
        print(f"  {verdict['verdict'].upper():6} reason={verdict.get('reason', '')!r} question={question!r}")

    for record in permanently_failed:
        question = record["messages"][1]["content"].split("</document>", 1)[-1].strip()[:100]
        print(f"  CALL_FAILED question={question!r}")


if __name__ == "__main__":
    main()
