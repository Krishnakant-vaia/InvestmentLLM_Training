"""Quality-scoring pass over the kept corpus.

Streams records (local file or S3), scores each 1-10 with the quality prompt,
writes one output file per shard (atomic, so a crash never leaves a half shard),
and concatenates them at the end. Resumable via the checkpoint; refuses to resume
under a different prompt hash. Pause with scripts/pause_run.py-style sentinel.

  python3 -u scripts/run_quality_scoring.py --input-file merged_output/sft_data_kept.jsonl \
      --state-dir scoring_pilot/run1 --sample-every 130 --no-upload

With --sample-every N only every Nth record is scored and _quality_src_index is the
position in the SAMPLE, not the source file. Full runs (N=1) index the source.
"""
import argparse
import dataclasses
import glob
import json
import math
import os
import sys

sys.path.insert(0, ".")

from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import load_settings
from qa_filter.pause import clear_pause_sentinel, pause_requested
from qa_filter.s3_io import stream_jsonl_lines, upload_json, upload_jsonl
from qa_scoring.pipeline import score_shard
from qa_scoring.prompt import load_prompt

DEFAULT_SOURCE_KEY = "Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_kept.jsonl"
# Main deliverables (scored/unscored/report) go at the TOP LEVEL, next to
# sft_data_kept.jsonl -- only working/shard files live under pc_krishnakant/.
TOP_LEVEL_PREFIX = "Corpus/Invest_dataset/09032026_SFTQ&Apairs/"
WORKING_PREFIX = "Corpus/Invest_dataset/09032026_SFTQ&Apairs/pc_krishnakant/qa_quality_analysis/"
DEFAULT_TOTAL_RECORDS = 647066


def _local_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line


def _sampled_shards(lines, sample_every, shard_size):
    shard, seen = [], 0
    for line in lines:
        if not line.strip():
            continue
        take = seen % sample_every == 0
        seen += 1
        if not take:
            continue
        try:
            shard.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(shard) == shard_size:
            yield shard
            shard = []
    if shard:
        yield shard


def _write_atomic(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _guard_prompt_version(state_dir, version):
    guard = os.path.join(state_dir, "prompt_version.txt")
    if os.path.exists(guard):
        existing = open(guard, encoding="utf-8").read().strip()
        if existing != version:
            sys.exit(f"refusing to resume: state was scored with {existing}, prompt is now {version}")
    else:
        with open(guard, "w", encoding="utf-8") as f:
            f.write(version)


def _concat(pattern, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for path in sorted(glob.glob(pattern), key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", help="local jsonl; default is streaming the kept file from S3")
    ap.add_argument("--state-dir", default="./scoring_state")
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-token-ceiling", type=int, default=None)
    ap.add_argument("--max-batch-records", type=int, default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--total-records", type=int, default=DEFAULT_TOTAL_RECORDS)
    args = ap.parse_args()

    settings = load_settings()
    overrides = {"run_state_dir": args.state_dir}
    if args.workers:
        overrides["max_workers"] = args.workers
    if args.batch_token_ceiling:
        overrides["batch_token_ceiling"] = args.batch_token_ceiling
    if args.max_batch_records:
        overrides["max_batch_records"] = args.max_batch_records
    settings = dataclasses.replace(settings, **overrides)

    template, version = load_prompt()
    os.makedirs(os.path.join(args.state_dir, "shards"), exist_ok=True)
    _guard_prompt_version(args.state_dir, version)
    print(f"prompt version {version}; workers {settings.max_workers}; state {args.state_dir}", flush=True)

    checkpoint = CheckpointStore(args.state_dir)
    if checkpoint.get_total_shards() is None:
        checkpoint.set_total_shards(max(1, math.ceil(args.total_records / args.sample_every / settings.shard_size)))

    def scorer(batch, s):
        from qa_scoring.score_client import call_scorer
        return call_scorer(batch, s, template=template)

    lines = _local_lines(args.input_file) if args.input_file else stream_jsonl_lines(settings.s3_bucket, DEFAULT_SOURCE_KEY)

    done = 0
    for shard_id, records in enumerate(_sampled_shards(lines, args.sample_every, settings.shard_size)):
        if args.max_shards is not None and done >= args.max_shards:
            break
        result = score_shard(records, shard_id, shard_id * settings.shard_size, settings,
                             checkpoint, version, score_fn=scorer)
        if result is None:
            print(f"shard {shard_id}: already complete, skipping", flush=True)
            continue
        _write_atomic(os.path.join(args.state_dir, "shards", f"scored_{shard_id}.jsonl"), result["scored"])
        _write_atomic(os.path.join(args.state_dir, "shards", f"unscored_{shard_id}.jsonl"), result["unscored"])
        done += 1
        print(f"shard {shard_id}: scored={len(result['scored'])} unscored={len(result['unscored'])}", flush=True)
        if pause_requested(args.state_dir):
            clear_pause_sentinel(args.state_dir)
            print(f"pause requested -- stopping after shard {shard_id}", flush=True)
            break

    scored_path = os.path.join(args.state_dir, "scored.jsonl")
    unscored_path = os.path.join(args.state_dir, "unscored.jsonl")
    n_scored = _concat(os.path.join(args.state_dir, "shards", "scored_*.jsonl"), scored_path)
    n_unscored = _concat(os.path.join(args.state_dir, "shards", "unscored_*.jsonl"), unscored_path)
    report = {"prompt_version": version, "scored": n_scored, "unscored": n_unscored,
              "sample_every": args.sample_every, "completed_shards": len(checkpoint.completed_shard_ids())}
    with open(os.path.join(args.state_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)

    if not args.no_upload:
        # Deliverables at top level; a copy of the report also kept under the
        # working prefix so pc_krishnakant/ has a self-contained run record.
        upload_jsonl(settings.s3_bucket, TOP_LEVEL_PREFIX + "sft_data_kept_scored.jsonl", scored_path)
        upload_jsonl(settings.s3_bucket, TOP_LEVEL_PREFIX + "sft_data_kept_unscored.jsonl", unscored_path)
        upload_json(settings.s3_bucket, TOP_LEVEL_PREFIX + "quality_score_report.json", report)
        upload_json(settings.s3_bucket, WORKING_PREFIX + "score_report.json", report)


if __name__ == "__main__":
    main()
