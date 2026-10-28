"""Measurement-only re-filter: takes the already-kept records from shards 0-96
(judged under either the original prompt, or max_stringent for shard 96 alone --
both identified by _judge_prompt_version being absent or "max_stringent") and
re-judges them against the CURRENT prompt (gepa_strict_v2, as configured in
qa_filter.config), to measure how many would still pass under the stricter bar.

Purely a measurement: writes to a separate local state dir and a separate S3
prefix (refilter_shards_0_96/) -- never touches run_state/ or the real
sft_data_kept.jsonl/sft_data_rejected.jsonl. Resumable via the same
checkpoint/pause pattern as the main pipeline (this reads a 5.85GB source and
re-judges ~359K records, expected to take hours).

Usage:
  python scripts/refilter_measurement.py        # run/resume
  python scripts/refilter_pause.py               # request a graceful pause
  python scripts/refilter_check_progress.py       # check progress + ETA
"""
import json
import math
import os
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qa_filter.batching import build_batches
from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import load_settings
from qa_filter.judge_retry import judge_batch_with_retry
from qa_filter.pause import clear_pause_sentinel, pause_requested
from qa_filter.s3_io import stream_jsonl_lines, upload_json, upload_jsonl
from qa_filter.worker_pool import run_batches_concurrently

REFILTER_STATE_DIR = "./refilter_state"
OUTPUT_PREFIX_SUFFIX = "refilter_shards_0_96/"

# A record targets this re-filter pass if it was judged under the original
# prompt (no _judge_prompt_version field at all -- that field was added after
# shard 95) or under max_stringent (shard 96 alone). Records already judged
# under gepa_strict_v2 (shards 97+) are excluded -- re-judging them against
# themselves would be meaningless.
_TARGET_VERSIONS = {None, "max_stringent"}


def _is_target_record(record: dict) -> bool:
    return record.get("_judge_prompt_version") in _TARGET_VERSIONS


def _shard_target_records(lines, shard_size):
    shard = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _is_target_record(record):
            continue
        shard.append(record)
        if len(shard) == shard_size:
            yield shard
            shard = []
    if shard:
        yield shard


def _strip_batch_item_id(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "_batch_item_id"}


def _process_shard(records, shard_id, settings, checkpoint):
    if checkpoint.is_complete(shard_id):
        return None

    for i, record in enumerate(records):
        record["_batch_item_id"] = i
    by_id = {r["_batch_item_id"]: r for r in records}

    batches = build_batches(records, settings.batch_token_ceiling, settings.max_batch_records)
    verdicts, permanently_failed = run_batches_concurrently(batches, settings, judge_batch_with_retry)

    kept: list[dict] = []
    rejected: list[dict] = []

    def _tag(record: dict) -> dict:
        cleaned = _strip_batch_item_id(record)
        cleaned["_original_judge_prompt_version"] = record.get("_judge_prompt_version")
        cleaned["_judge_prompt_version"] = settings.judge_prompt_version
        return cleaned

    for verdict in verdicts:
        record = by_id[verdict["id"]]
        tagged = _tag(record)
        if verdict.get("verdict") == "keep":
            kept.append(tagged)
        else:
            tagged["_reject_reason"] = f"judge_rejected: {verdict.get('reason', '')}"
            rejected.append(tagged)

    for record in permanently_failed:
        tagged = _tag(record)
        tagged["_reject_reason"] = "judge_call_failed"
        rejected.append(tagged)

    checkpoint.mark_complete(shard_id)
    return {"kept": kept, "rejected": rejected}


def main():
    settings = load_settings()
    os.makedirs(REFILTER_STATE_DIR, exist_ok=True)
    checkpoint = CheckpointStore(REFILTER_STATE_DIR)

    source_key = settings.output_prefix + "sft_data_kept.jsonl"

    if checkpoint.get_total_shards() is None:
        print("profiling sft_data_kept.jsonl for target records (original-prompt + max_stringent)...", flush=True)
        count = 0
        for line in stream_jsonl_lines(settings.s3_bucket, source_key):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_target_record(record):
                count += 1
        total_shards = math.ceil(count / settings.shard_size) or 1
        checkpoint.set_total_shards(total_shards)
        print(f"found {count} target records -> {total_shards} shards", flush=True)

    kept_path = os.path.join(REFILTER_STATE_DIR, "kept.jsonl")
    rejected_path = os.path.join(REFILTER_STATE_DIR, "rejected.jsonl")

    lines = stream_jsonl_lines(settings.s3_bucket, source_key)
    with open(kept_path, "a", encoding="utf-8") as kept_f, \
         open(rejected_path, "a", encoding="utf-8") as rejected_f:
        for shard_id, records in enumerate(_shard_target_records(lines, settings.shard_size)):
            result = _process_shard(records, shard_id, settings, checkpoint)
            if result is None:
                print(f"shard {shard_id}: already complete, skipping", flush=True)
                continue

            for record in result["kept"]:
                kept_f.write(json.dumps(record) + "\n")
            for record in result["rejected"]:
                rejected_f.write(json.dumps(record) + "\n")

            print(f"shard {shard_id}: kept={len(result['kept'])} rejected={len(result['rejected'])}", flush=True)

            if pause_requested(REFILTER_STATE_DIR):
                clear_pause_sentinel(REFILTER_STATE_DIR)
                print(f"pause requested -- stopping after shard {shard_id}", flush=True)
                break

    with open(kept_path, encoding="utf-8") as f:
        total_kept = sum(1 for _ in f)
    with open(rejected_path, encoding="utf-8") as f:
        total_rejected = sum(1 for _ in f)
    total = total_kept + total_rejected

    report = {
        "total_measured": total,
        "kept": total_kept,
        "rejected": total_rejected,
        "pass_rate": (total_kept / total) if total else 0.0,
    }

    output_prefix = settings.output_prefix + OUTPUT_PREFIX_SUFFIX
    upload_jsonl(settings.s3_bucket, output_prefix + "refilter_kept.jsonl", kept_path)
    upload_jsonl(settings.s3_bucket, output_prefix + "refilter_rejected.jsonl", rejected_path)
    upload_json(settings.s3_bucket, output_prefix + "refilter_report.json", report)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
