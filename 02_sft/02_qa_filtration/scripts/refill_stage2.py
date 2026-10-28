"""Stage 2: fill the shortfall sub-capabilities from the filtration rejects.

Rejects have no quality score, so candidates are re-scored with the same quality
prompt/pipeline; only those scoring >= 8 with a clean question are accepted.
Candidate pool: same folder, question clean, think block present, not a format
defect (missing_think_tag / prefilter). Up to CANDIDATE_X x the shortfall is sampled
per folder (seeded) and scored.
"""
import collections
import dataclasses
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qa_filter.checkpoint import CheckpointStore
from qa_filter.config import load_settings
from qa_filter.s3_io import stream_jsonl_lines
from qa_scoring.pipeline import score_shard
from qa_scoring.prompt import load_prompt
from qa_scoring.score_client import call_scorer

OUTDIR = "merged_output/refill"
CANDIDATE_X = 2.5
SEED = 42
A = re.compile(r"(?i)\b(the|this)\s+document\b|\bthe\s+excerpt\b|\bthe\s+passage\b|\bbased on the (provided|given)\b")
B = re.compile(r"(?i)\bthe\s+(article|text|memo|source)\b|\baccording to the (document|report|filing|article|text|memo)\b|\bthe\s+(provided|attached|above)\s+(data|text|document|information|table)\b")
DOC = re.compile(r"^\s*<document>.*?</document>\s*(.*)$", re.S)
THINK = re.compile(r"<think>.*?</think>\s*(.*)$", re.S)


def flagged(q):
    return bool(A.search(q) or B.search(q))


def qa_hash(q, a):
    return hashlib.md5((q + "\x00" + a).encode("utf-8")).hexdigest()


def collect():
    plan = json.load(open(f"{OUTDIR}/plan.json"))
    short = plan["short"]
    # hashes already used (current domain block + stage-1 replacements)
    used = set()
    with open("merged_output/sft_data_finetune_pool_merged.jsonl", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= 178018:
                break
            m = json.loads(l)["messages"]
            used.add(qa_hash(m[1]["content"], m[2]["content"]))
    with open(f"{OUTDIR}/kept_replacements.jsonl", encoding="utf-8") as f:
        for l in f:
            m = json.loads(l)["messages"]
            used.add(qa_hash(m[1]["content"], m[2]["content"]))
    pool = collections.defaultdict(list)
    n = 0
    for line in stream_jsonl_lines("ft-vaia", "Corpus/Invest_dataset/09032026_SFTQ&Apairs/pc_krishnakant/qa_filtration_run/sft_data_rejected.jsonl"):
        if not line.strip():
            continue
        n += 1
        r = json.loads(line)
        try:
            fo = r["_source"].split("/")[3]
        except Exception:
            continue
        if fo not in short:
            continue
        reason = str(r.get("_reject_reason", ""))
        if reason.startswith("missing_think_tag") or r.get("_judge_prompt_version") == "prefilter":
            continue
        try:
            q = DOC.match(r["messages"][1]["content"]).group(1).strip()
            a = r["messages"][2]["content"]
        except Exception:
            continue
        if not q or flagged(q) or not THINK.search(a):
            continue
        if qa_hash(q, a) in used:
            continue
        pool[fo].append(r)
    print(f"scanned {n:,} rejects; candidate folders {len(pool)}", flush=True)
    rng = random.Random(SEED)
    sample = []
    for fo, need in sorted(short.items()):
        lst = pool.get(fo, [])
        k = min(len(lst), int(need * CANDIDATE_X) + 1)
        sample.extend(rng.sample(lst, k))
        print(f"  {fo.split(' - ',1)[1][:48]:48} need {need:>5} candidates {len(lst):>5} sampled {k:>5}", flush=True)
    for i, r in enumerate(sample):
        r["_reject_ordinal"] = i
    with open(f"{OUTDIR}/reject_candidates.jsonl", "w", encoding="utf-8", newline="\n") as o:
        for r in sample:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("candidates written:", len(sample), flush=True)


def score(only_shards=None, workers=160, tag=""):
    settings = dataclasses.replace(load_settings(), run_state_dir=f"{OUTDIR}/score_state{tag}", max_workers=workers)
    template, version = load_prompt()
    cp = CheckpointStore(settings.run_state_dir)
    os.makedirs(f"{OUTDIR}/score_state{tag}", exist_ok=True)
    recs = [json.loads(l) for l in open(f"{OUTDIR}/reject_candidates.jsonl", encoding="utf-8")]
    print("scoring", len(recs), "candidates with", version, flush=True)

    def scorer(batch, s):
        return call_scorer(batch, s, template=template)

    out = f"{OUTDIR}/reject_scored{tag}.jsonl"
    SH = 5000
    with open(out, "a", encoding="utf-8", newline="\n") as o:
        for sid, start in enumerate(range(0, len(recs), SH)):
            if only_shards is not None and sid not in only_shards:
                continue
            res = score_shard(recs[start:start + SH], sid, start, settings, cp, version, score_fn=scorer)
            if res is None:
                print("shard", sid, "already done", flush=True)
                continue
            for r in res["scored"]:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"shard {sid}: scored {len(res['scored'])} unscored {len(res['unscored'])}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect()
    elif len(sys.argv) > 2:   # score <shard ids comma> <workers> <tag>
        score({int(x) for x in sys.argv[2].split(",")}, int(sys.argv[3]), sys.argv[4])
    else:
        score()
