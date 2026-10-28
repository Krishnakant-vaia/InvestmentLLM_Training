from qa_filter.batching import build_batches
from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import Settings
from qa_filter.worker_pool import run_batches_concurrently
from qa_scoring.records import to_scorer_item
from qa_scoring.scoring import score_batch_with_retry


def score_shard(records, shard_id, first_index, settings: Settings, checkpoint: CheckpointStore,
                prompt_version: str, score_fn=None) -> dict | None:
    if checkpoint.is_complete(shard_id):
        return None

    items = [to_scorer_item(r, first_index + i) for i, r in enumerate(records)]
    batches = build_batches(items, settings.batch_token_ceiling, settings.max_batch_records)

    if score_fn is None:
        retry_fn = score_batch_with_retry
    else:
        def retry_fn(batch, s):
            return score_batch_with_retry(batch, s, score_fn)

    results, failed = run_batches_concurrently(batches, settings, retry_fn)

    scored = []
    for r in results:
        out = {
            **r["item"]["_orig"],
            "_quality_score": r["score"],
            "_quality_categories": r["categories"],
            "_quality_reason": r["reason"],
            "_quality_prompt_version": prompt_version,
            "_quality_src_index": r["item"]["_idx"],
        }
        if r["anomalies"]:
            out["_quality_anomalies"] = r["anomalies"]
        scored.append(out)

    unscored = [
        {**item["_orig"], "_quality_reason": "score_call_failed",
         "_quality_prompt_version": prompt_version, "_quality_src_index": item["_idx"]}
        for item in failed
    ]

    checkpoint.mark_complete(shard_id)
    return {"scored": scored, "unscored": unscored, "shard_id": shard_id}
