"""Full filtration run: streams the source from S3 in shards, processes each
shard through the pipeline (resumable via checkpoint), writes local per-shard
kept/rejected files, then uploads the combined result plus a report to S3.
"""
import json
import math
import os
import sys

sys.path.insert(0, ".")

from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import load_settings
from qa_filter.pause import clear_pause_sentinel, pause_requested
from qa_filter.pipeline import process_shard
from qa_filter.profiling import profile_dataset
from qa_filter.s3_io import stream_jsonl_lines, upload_json, upload_jsonl


def _shard_records(lines, shard_size):
    """Yields (shard_records, malformed_count_in_shard) tuples so the caller can
    report how many source lines were silently unparseable, matching the audit
    parity profiling.py already gives for Stage -1."""
    shard = []
    malformed_count = 0
    for line in lines:
        try:
            shard.append(json.loads(line))
        except json.JSONDecodeError:
            malformed_count += 1
            continue
        if len(shard) == shard_size:
            yield shard, malformed_count
            shard = []
            malformed_count = 0
    if shard or malformed_count:
        yield shard, malformed_count


def main():
    settings = load_settings()
    checkpoint = CheckpointStore(settings.run_state_dir)
    os.makedirs(settings.run_state_dir, exist_ok=True)

    # Total shard count is needed for check_progress.py's completeness percentage
    # and ETA. Computed once via a cheap no-LLM streaming pass (profiling.py) and
    # cached in checkpoint.json -- skipped on resume so a restart doesn't re-scan
    # the whole file just to re-derive a number it already knows.
    if checkpoint.get_total_shards() is None:
        profile = profile_dataset(stream_jsonl_lines(settings.s3_bucket, settings.source_key))
        total_shards = math.ceil(profile["total_records"] / settings.shard_size) or 1
        checkpoint.set_total_shards(total_shards)
        print(f"profiled source: {profile['total_records']} records -> {total_shards} shards")

    all_kept_path = os.path.join(settings.run_state_dir, "kept.jsonl")
    all_rejected_path = os.path.join(settings.run_state_dir, "rejected.jsonl")

    reject_reason_counts: dict = {}
    total_kept = 0
    total_processed = 0
    total_malformed_source_lines = 0

    lines = stream_jsonl_lines(settings.s3_bucket, settings.source_key)
    with open(all_kept_path, "a", encoding="utf-8") as kept_f, \
         open(all_rejected_path, "a", encoding="utf-8") as rejected_f:
        for shard_id, (records, malformed_count) in enumerate(_shard_records(lines, settings.shard_size)):
            total_malformed_source_lines += malformed_count
            if not records:
                continue

            result = process_shard(records, shard_id, settings, checkpoint)
            if result.get("skipped"):
                print(f"shard {shard_id}: already complete, skipping")
                continue

            for record in result["kept"]:
                kept_f.write(json.dumps(record) + "\n")
            for record in result["rejected"]:
                rejected_f.write(json.dumps(record) + "\n")
                reason = record.get("_reject_reason", "unknown")
                reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

            total_kept += len(result["kept"])
            total_processed += len(result["kept"]) + len(result["rejected"])
            print(f"shard {shard_id}: kept={len(result['kept'])} rejected={len(result['rejected'])}")

            # Checked only after a shard fully finishes (mark_complete already ran
            # inside process_shard above) -- never mid-shard, so a pause never
            # discards in-flight judge work. Uploading kept/rejected/report below
            # still runs on a paused exit, so S3 reflects everything done so far;
            # rerunning this same script later resumes at the next incomplete shard.
            if pause_requested(settings.run_state_dir):
                clear_pause_sentinel(settings.run_state_dir)
                print(f"pause requested -- stopping after shard {shard_id}")
                break

    report = {
        "total_processed": total_processed,
        "total_kept": total_kept,
        "pass_rate": (total_kept / total_processed) if total_processed else 0.0,
        "reject_reason_counts": reject_reason_counts,
        "total_malformed_source_lines": total_malformed_source_lines,
    }

    upload_jsonl(settings.s3_bucket, settings.output_prefix + "sft_data_kept.jsonl", all_kept_path)
    upload_jsonl(settings.s3_bucket, settings.output_prefix + "sft_data_rejected.jsonl", all_rejected_path)
    upload_json(settings.s3_bucket, settings.output_prefix + "filter_report.json", report)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
