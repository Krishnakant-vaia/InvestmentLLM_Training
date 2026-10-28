"""Builds the fine-tuning pool: a verbatim subset of the scored corpus.

Pool rule (matches Capability_Pool_Counts.xlsx):
  - every record scoring 9 or 10;
  - plus records scoring 7 or 8 in the thin categories EK, AR, EA, CL;
  - the five named Domain Knowledge sub-capabilities are cut to fixed targets
    (score 10 first, then a seeded random sample of score-9 records).
Lines are copied byte-for-byte from the scored file: no field is added or edited.

  python3 -u scripts/build_finetune_pool.py
"""
import collections
import json
import random
import sys

SCORED = "scoring_state/scored.jsonl"
OUT = "merged_output/sft_data_finetune_pool_178k.jsonl"
SEED = 42
TOPUP_CATS = {"EK", "AR", "EA", "CL"}
DK_TARGETS = {  # folder-name prefix -> target count
    "DK_PRIVATE_MARKETS_TERMINOLOGY": 9605,
    "DK_DEBT_CREDIT_CONVENTIONS": 9605,
    "DK_INDIA_REGULATORY_GROUNDING": 9604,
    "DK_ACCOUNTING_STANDARDS": 9604,
    "DK_ESG_SUSTAINABILITY_FRAMEWORKS": 9604,
}


def key_of(folder):
    return folder.split(" - ", 1)[0]


def eligible(score, cat):
    return score >= 9 or (score >= 7 and cat in TOPUP_CATS)


def fields(line):
    r = json.loads(line)
    folder = r["_source"].split("/")[3]
    return r["_quality_score"], r["_quality_src_index"], folder, folder.split("_")[0]


# ---- pass 1: gather candidates for the capped sub-capabilities ---------------
candidates = collections.defaultdict(list)  # key -> [(score, src_index)]
with open(SCORED, encoding="utf-8") as f:
    for line in f:
        score, idx, folder, cat = fields(line)
        k = key_of(folder)
        if k in DK_TARGETS and eligible(score, cat):
            candidates[k].append((score, idx))

rng = random.Random(SEED)
keep = set()
for k, target in DK_TARGETS.items():
    tens = sorted(i for s, i in candidates[k] if s == 10)
    nines = sorted(i for s, i in candidates[k] if s == 9)
    chosen = tens[:target]
    if len(chosen) < target:
        chosen += rng.sample(nines, target - len(chosen))
    assert len(chosen) == target, (k, len(chosen), target)
    keep.update(chosen)
    print(f"{k}: candidates {len(candidates[k]):,} (score10 {len(tens)}) -> {target:,}", flush=True)

# ---- pass 2: write the pool ----------------------------------------------------
counts = collections.Counter()
seen = set()
with open(SCORED, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8", newline="\n") as out:
    for line in f:
        score, idx, folder, cat = fields(line)
        if not eligible(score, cat):
            continue
        if key_of(folder) in DK_TARGETS and idx not in keep:
            continue
        assert idx not in seen, idx
        seen.add(idx)
        out.write(line if line.endswith("\n") else line + "\n")
        counts[(cat, folder)] += 1

by_cat = collections.Counter()
for (cat, _), n in counts.items():
    by_cat[cat] += n
print("total", f"{sum(by_cat.values()):,}")
for cat in ("DK", "CG", "FR", "WF", "EK", "AR", "EA", "CL"):
    print(f"  {cat}: {by_cat[cat]:,}")
json.dump({"total": sum(by_cat.values()), "by_category": by_cat,
           "by_subcapability": {f: n for (c, f), n in sorted(counts.items())},
           "seed": SEED, "dk_targets": DK_TARGETS},
          open("merged_output/finetune_pool_manifest.json", "w", encoding="utf-8"), indent=2)
