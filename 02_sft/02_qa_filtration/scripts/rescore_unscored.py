"""Re-scores records left in unscored.jsonl after the main run (the model kept
truncating its output on them), with a larger output-token cap and more retries,
one record per call. Successes are appended to scored.jsonl and removed from
unscored.jsonl; the combined top-level files are then re-uploaded.

  python3 -u scripts/rescore_unscored.py --state-dir scoring_state
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, ".")

from qa_filter.config import load_settings
from qa_filter.s3_io import upload_json, upload_jsonl
from qa_scoring.pipeline import score_shard  # noqa: F401  (keeps import path check honest)
from qa_scoring.prompt import load_prompt
from qa_scoring.records import to_scorer_item
from qa_scoring.score_client import call_scorer
from qa_scoring.scoring import score_batch_with_retry

TOP_LEVEL_PREFIX = "Corpus/Invest_dataset/09032026_SFTQ&Apairs/"
WORKING_PREFIX = TOP_LEVEL_PREFIX + "pc_krishnakant/qa_quality_analysis/"


def _original(record):
    return {k: v for k, v in record.items() if not k.startswith("_quality_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="./scoring_state")
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    settings = dataclasses.replace(load_settings(), max_output_tokens=args.max_output_tokens)
    template, version = load_prompt()
    guard = open(os.path.join(args.state_dir, "prompt_version.txt"), encoding="utf-8").read().strip()
    if guard != version:
        sys.exit(f"prompt changed since the main run ({guard} vs {version}); refusing to mix versions")

    unscored_path = os.path.join(args.state_dir, "unscored.jsonl")
    scored_path = os.path.join(args.state_dir, "scored.jsonl")
    pending = [json.loads(line) for line in open(unscored_path, encoding="utf-8") if line.strip()]
    print(f"{len(pending)} unscored records to retry", flush=True)

    def scorer(batch, s):
        return call_scorer(batch, s, template=template)

    rescored, still_failed = [], []
    for rec in pending:
        src_index = rec["_quality_src_index"]
        item = to_scorer_item(_original(rec), src_index)
        results, failed = score_batch_with_retry([item], settings, scorer, same_size_retries=args.retries)
        if not results:
            print(f"src_index {src_index}: still failing", flush=True)
            still_failed.append(rec)
            continue
        r = results[0]
        out = {
            **_original(rec),
            "_quality_score": r["score"],
            "_quality_categories": r["categories"],
            "_quality_reason": r["reason"],
            "_quality_prompt_version": version,
            "_quality_src_index": src_index,
        }
        if r["anomalies"]:
            out["_quality_anomalies"] = r["anomalies"]
        rescored.append(out)
        print(f"src_index {src_index}: score {r['score']} {r['reason']}", flush=True)

    if not rescored:
        print("nothing rescored; files unchanged")
        return

    with open(scored_path, "a", encoding="utf-8", newline="\n") as f:
        for r in rescored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(unscored_path, "w", encoding="utf-8", newline="\n") as f:
        for r in still_failed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_scored = sum(1 for _ in open(scored_path, encoding="utf-8"))
    report = {"prompt_version": version, "scored": n_scored, "unscored": len(still_failed),
              "rescored_after_main_run": len(rescored), "sample_every": 1, "completed_shards": 130}
    with open(os.path.join(args.state_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)

    if not args.no_upload:
        upload_jsonl(settings.s3_bucket, TOP_LEVEL_PREFIX + "sft_data_kept_scored.jsonl", scored_path)
        upload_jsonl(settings.s3_bucket, TOP_LEVEL_PREFIX + "sft_data_kept_unscored.jsonl", unscored_path)
        upload_json(settings.s3_bucket, TOP_LEVEL_PREFIX + "quality_score_report.json", report)
        upload_json(settings.s3_bucket, WORKING_PREFIX + "score_report.json", report)


if __name__ == "__main__":
    main()
