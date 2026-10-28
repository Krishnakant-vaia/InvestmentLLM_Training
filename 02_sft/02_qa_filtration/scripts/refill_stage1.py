"""Stage 1 of the document-reference clean-up.

 1. Flag domain rows whose QUESTION refers to a source document (groups A+B).
 2. For each sub-capability, pick same-folder replacements from the scored corpus
    that are not already in the pool and whose question is clean (score desc,
    seeded shuffle on ties, no duplicate question+answer text).
 3. Write plan.json (needs / shortfalls) and kept_replacements.jsonl (final-format
    records + audit sidecar fields).
"""
import collections
import hashlib
import json
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
N_DOMAIN = 178018
META = "merged_output/sft_data_finetune_pool_178k.jsonl"      # pool WITH metadata + documents (same line order)
SAVED = "merged_output/sft_data_finetune_pool_merged.jsonl"   # current training file
SCORED = "scoring_state/scored.jsonl"
OUTDIR = "merged_output/refill"
SYSPROMPT = open("prompts/systemprompt_SFT.txt", encoding="utf-8").read().replace("\r\n", "\n").strip()

A = re.compile(r"(?i)\b(the|this)\s+document\b|\bthe\s+excerpt\b|\bthe\s+passage\b|\bbased on the (provided|given)\b")
B = re.compile(r"(?i)\bthe\s+(article|text|memo|source)\b|\baccording to the (document|report|filing|article|text|memo)\b|\bthe\s+(provided|attached|above)\s+(data|text|document|information|table)\b")
DOC = re.compile(r"^\s*<document>.*?</document>\s*(.*)$", re.S)
THINK = re.compile(r"<think>.*?</think>\s*(.*)$", re.S)


def flagged(q):
    return bool(A.search(q) or B.search(q))


def qa_hash(q, a):
    return hashlib.md5((q + "\x00" + a).encode("utf-8")).hexdigest()


def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- pass 0: pool membership, flags, existing hashes ----------------------
    inpool = set()
    flag_lines = []                      # line numbers (0-based) in domain block to remove
    need = collections.Counter()         # folder -> number to replace
    pool_count = collections.Counter()
    existing = set()
    removed_src = []
    with open(META, encoding="utf-8") as a, open(SAVED, encoding="utf-8") as b:
        for i, la in enumerate(a):
            lb = b.readline()
            if i >= N_DOMAIN:
                break
            r = json.loads(la)
            m = json.loads(lb)["messages"]
            assert r["messages"][1]["content"].endswith(m[1]["content"])
            fo = r["_source"].split("/")[3]
            inpool.add(r["_quality_src_index"])
            pool_count[fo] += 1
            existing.add(qa_hash(m[1]["content"], m[2]["content"]))
            if flagged(m[1]["content"]):
                flag_lines.append(i)
                need[fo] += 1
                removed_src.append(r["_quality_src_index"])
    print(f"flagged {len(flag_lines):,} of {N_DOMAIN:,}; pool ids {len(inpool):,}", flush=True)

    # ---- pass 1: candidate index from scored corpus ----------------------------
    cands = collections.defaultdict(list)   # folder -> [(score, idx, hash)]
    seen_hash = set(existing)
    with open(SCORED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx = r["_quality_src_index"]
            sc = r["_quality_score"]
            if idx in inpool or sc < 7:
                continue
            fo = r["_source"].split("/")[3]
            if need[fo] == 0:
                continue
            mm = DOC.match(r["messages"][1]["content"])
            if not mm:
                continue
            q = mm.group(1).strip()
            if flagged(q):
                continue
            t = THINK.search(r["messages"][2]["content"])
            if not t:
                continue
            h = qa_hash(q, r["messages"][2]["content"])
            if h in seen_hash:
                continue
            cands[fo].append((sc, idx, h))
    print("candidate folders:", len(cands), flush=True)

    rng = random.Random(SEED)
    chosen = {}      # idx -> folder
    short = {}
    chosen_hash = set()
    for fo, n in sorted(need.items()):
        lst = cands.get(fo, [])
        lst.sort(key=lambda x: (-x[0], x[1]))
        # seeded shuffle within each score tier
        out, tier, cur = [], [], None
        for it in lst + [(None, None, None)]:
            if it[0] != cur and tier:
                rng.shuffle(tier)
                out.extend(tier)
                tier = []
            cur = it[0]
            if it[0] is not None:
                tier.append(it)
        picked = []
        for sc, idx, h in out:
            if h in chosen_hash:
                continue
            picked.append((sc, idx, h))
            chosen_hash.add(h)
            if len(picked) == n:
                break
        for sc, idx, h in picked:
            chosen[idx] = fo
        if len(picked) < n:
            short[fo] = n - len(picked)
    print(f"selected {len(chosen):,} replacements; shortfall folders {len(short)} totalling {sum(short.values()):,}", flush=True)

    # ---- pass 2: materialise chosen records -----------------------------------
    got = 0
    with open(SCORED, encoding="utf-8") as f, open(f"{OUTDIR}/kept_replacements.jsonl", "w", encoding="utf-8", newline="\n") as o:
        for line in f:
            r = json.loads(line)
            idx = r["_quality_src_index"]
            if idx not in chosen:
                continue
            q = DOC.match(r["messages"][1]["content"]).group(1).strip()
            rec = {"messages": [{"role": "system", "content": SYSPROMPT},
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": r["messages"][2]["content"]}],
                   "_audit": {"src_index": idx, "folder": chosen[idx], "score": r["_quality_score"], "from": "kept_unused"}}
            o.write(json.dumps(rec, ensure_ascii=False) + "\n")
            got += 1
    assert got == len(chosen)
    json.dump({"flag_lines": flag_lines, "removed_src_index": removed_src, "need": dict(need), "short": short,
               "pool_count": dict(pool_count), "seed": SEED},
              open(f"{OUTDIR}/plan.json", "w"))
    by = collections.Counter()
    for fo in chosen.values():
        by[fo] += 1
    print("done. replacements by score tier:")
    sc_tier = collections.Counter()
    with open(f"{OUTDIR}/kept_replacements.jsonl", encoding="utf-8") as f:
        for l in f:
            sc_tier[json.loads(l)["_audit"]["score"]] += 1
    print(dict(sorted(sc_tier.items())))


if __name__ == "__main__":
    main()
