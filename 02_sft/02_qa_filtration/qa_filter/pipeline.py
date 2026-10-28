from qa_filter.batching import build_batches
from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import Settings
from qa_filter.judge_retry import judge_batch_with_retry
from qa_filter.prefilter import PreFilter
from qa_filter.worker_pool import run_batches_concurrently


def process_shard(
    records: list[dict],
    shard_id: int,
    settings: Settings,
    checkpoint: CheckpointStore,
    judge_fn=judge_batch_with_retry,
) -> dict:
    if checkpoint.is_complete(shard_id):
        return {"kept": [], "rejected": [], "shard_id": shard_id, "skipped": True}

    prefiltered_kept, prefiltered_rejected = PreFilter().filter(records)

    # Never reached the judge, so tagging them with a judge prompt version would
    # be misleading -- "prefilter" marks these as rejected before any judge ran,
    # distinct from a judge-rejected record's real version string.
    prefiltered_rejected = [{**r, "_judge_prompt_version": "prefilter"} for r in prefiltered_rejected]

    for i, record in enumerate(prefiltered_kept):
        record["_batch_item_id"] = i
    by_batch_item_id = {r["_batch_item_id"]: r for r in prefiltered_kept}

    batches = build_batches(prefiltered_kept, settings.batch_token_ceiling, settings.max_batch_records)
    verdicts, permanently_failed = run_batches_concurrently(batches, settings, judge_fn)

    kept: list[dict] = []
    rejected: list[dict] = list(prefiltered_rejected)

    def _strip_batch_item_id(record: dict) -> dict:
        # _batch_item_id is an internal batching artifact added a few lines up
        # (by enumerate(prefiltered_kept)) purely so verdicts can be matched back
        # to records; it must never reach kept.jsonl/rejected.jsonl on S3.
        return {k: v for k, v in record.items() if k != "_batch_item_id"}

    for verdict in verdicts:
        record = by_batch_item_id[verdict["id"]]
        record_with_version = {**_strip_batch_item_id(record), "_judge_prompt_version": settings.judge_prompt_version}
        if verdict.get("verdict") == "keep":
            kept.append(record_with_version)
        else:
            rejected.append({
                **record_with_version,
                "_reject_reason": f"judge_rejected: {verdict.get('reason', '')}",
            })

    for record in permanently_failed:
        rejected.append({
            **_strip_batch_item_id(record),
            "_judge_prompt_version": settings.judge_prompt_version,
            "_reject_reason": "judge_call_failed",
        })

    checkpoint.mark_complete(shard_id)
    return {"kept": kept, "rejected": rejected, "shard_id": shard_id}
