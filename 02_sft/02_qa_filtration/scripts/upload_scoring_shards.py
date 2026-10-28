"""Continuously uploads finished scoring shards to S3 while the scorer runs.

Each shard file is written atomically by run_quality_scoring.py (tmp + rename), so
any scored_N.jsonl that exists is complete. Uploaded shards are recorded in
uploaded.json so a restart never re-uploads. Exits once every shard is uploaded.

  python3 -u scripts/upload_scoring_shards.py --state-dir scoring_state
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, ".")

from qa_filter.config import load_settings
from qa_filter.s3_io import upload_jsonl

PREFIX = "Corpus/Invest_dataset/09032026_SFTQ&Apairs/pc_krishnakant/qa_quality_analysis/shards/"


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="./scoring_state")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()

    settings = load_settings()
    shard_dir = os.path.join(args.state_dir, "shards")
    log_path = os.path.join(args.state_dir, "uploaded.json")
    uploaded = set(_read_json(log_path, []))

    while True:
        total = _read_json(os.path.join(args.state_dir, "checkpoint.json"), {}).get("total_shards")
        names = sorted(n for n in os.listdir(shard_dir) if re.fullmatch(r"(un)?scored_\d+\.jsonl", n))
        for name in names:
            if name in uploaded:
                continue
            local = os.path.join(shard_dir, name)
            if os.path.getsize(local) == 0:  # empty unscored file: nothing to keep
                uploaded.add(name)
                continue
            try:
                upload_jsonl(settings.s3_bucket, PREFIX + name, local)
            except Exception as e:  # network blip: leave it un-recorded, retry next pass
                print(f"upload of {name} failed ({type(e).__name__}); will retry next pass", flush=True)
                continue
            uploaded.add(name)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(sorted(uploaded), f)
            print(f"uploaded {name}", flush=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(sorted(uploaded), f)

        scored_done = len([n for n in uploaded if n.startswith("scored_")])
        if total and scored_done >= total:
            print("all shards uploaded", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
