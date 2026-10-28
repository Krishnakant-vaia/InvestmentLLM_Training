"""Stage 3: assemble the new training file and validate it (local only; upload is separate).

Domain block  = current domain rows minus flagged ones (original order)
              + stage-1 replacements (kept-but-unused)
              + stage-2 replacements (re-scored rejects scoring >= 8)
Glaive / Smoltalk lines are copied byte-for-byte.
"""
import collections
import hashlib
import json
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUTDIR = "merged_output/refill"
CUR = "merged_output/sft_data_finetune_pool_merged.pre_ref_replace.jsonl"
META = "merged_output/sft_data_finetune_pool_178k.jsonl"
NEW = "merged_output/sft_data_finetune_pool_merged.new.jsonl"
N_DOMAIN = 178018
SEED = 42
SYSPROMPT = open("prompts/systemprompt_SFT.txt", encoding="utf-8").read().replace("\r\n", "\n").strip()
A = re.compile(r"(?i)\b(the|this)\s+document\b|\bthe\s+excerpt\b|\bthe\s+passage\b|\bbased on the (provided|given)\b")
B = re.compile(r"(?i)\bthe\s+(article|text|memo|source)\b|\baccording to the (document|report|filing|article|text|memo)\b|\bthe\s+(provided|attached|above)\s+(data|text|document|information|table)\b")
DOC = re.compile(r"^\s*<document>.*?</document>\s*(.*)$", re.S)
THINK = re.compile(r"<think>.*?</think>\s*(.*)$", re.S)
flagged = lambda q: bool(A.search(q) or B.search(q))
qa_hash = lambda q, a: hashlib.md5((q + "\x00" + a).encode()).hexdigest()


def main():
    plan = json.load(open(f"{OUTDIR}/plan.json"))
    flag = set(plan["flag_lines"])
    short = plan["short"]

    # stage-2 acceptance
    used = set()
    s1 = [json.loads(l) for l in open(f"{OUTDIR}/kept_replacements.jsonl", encoding="utf-8")]
    for r in s1:
        used.add(qa_hash(r["messages"][1]["content"], r["messages"][2]["content"]))
    rej = collections.defaultdict(list)
    rng = random.Random(SEED)
    tier = collections.Counter()
    import glob
    seen_ord = set()
    lines = []
    for fp in sorted(glob.glob(f"{OUTDIR}/reject_scored*.jsonl")):
        with open(fp, encoding="utf-8", newline="\n") as fh:      # split on newline only, not U+2028 etc.
            lines += [x.rstrip("\n") for x in fh if x.strip()]
    torn = 0
    for l in lines:
        try:
            r = json.loads(l)
        except json.JSONDecodeError:      # last line cut off when the scorer was stopped
            torn += 1
            continue
        if r["_reject_ordinal"] in seen_ord:      # shard scored twice -> keep first
            continue
        seen_ord.add(r["_reject_ordinal"])
        tier[r["_quality_score"]] += 1
        if r["_quality_score"] < 8:
            continue
        q = DOC.match(r["messages"][1]["content"]).group(1).strip()
        if flagged(q) or not THINK.search(r["messages"][2]["content"]):
            continue
        fo = r["_source"].split("/")[3]
        rej[fo].append((r["_quality_score"], rng.random(), q, r))
    print("torn lines skipped:", torn)
    print("rejects re-scored, score distribution:", dict(sorted(tier.items())), flush=True)
    s2 = []
    for fo, need in short.items():
        lst = sorted(rej.get(fo, []), key=lambda x: (-x[0], x[1]))
        got = 0
        for sc, _, q, r in lst:
            h = qa_hash(q, r["messages"][2]["content"])
            if h in used:
                continue
            used.add(h)
            s2.append({"messages": [{"role": "system", "content": SYSPROMPT}, {"role": "user", "content": q},
                                    {"role": "assistant", "content": r["messages"][2]["content"]}],
                       "_audit": {"src_index": None, "folder": fo, "score": sc, "from": "rejected_rescored"}})
            got += 1
            if got == need:
                break
    print(f"stage-2 accepted {len(s2):,} of {sum(short.values()):,} thin slots", flush=True)

    # folder of each current domain line (for per-sub-capability validation)
    folder_of = []
    with open(META, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= N_DOMAIN:
                break
            folder_of.append(json.loads(l)["_source"].split("/")[3])
    before = collections.Counter(folder_of)

    after = collections.Counter()
    n_dom = 0
    added = s1 + s2
    with open(CUR, encoding="utf-8") as f, open(NEW, "w", encoding="utf-8", newline="\n") as o:
        for i, l in enumerate(f):
            if i < N_DOMAIN:
                if i in flag:
                    continue
                o.write(l if l.endswith("\n") else l + "\n")
                after[folder_of[i]] += 1
                n_dom += 1
            else:
                if i == N_DOMAIN:
                    for r in added:
                        o.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
                        after[r["_audit"]["folder"]] += 1
                        n_dom += 1
                o.write(l if l.endswith("\n") else l + "\n")
    json.dump({"added": [r["_audit"] for r in added], "domain_rows_after": n_dom,
               "per_folder_before": dict(before), "per_folder_after": dict(after)},
              open(f"{OUTDIR}/audit.json", "w"))
    print(f"domain block: {N_DOMAIN:,} -> {n_dom:,}", flush=True)
    diffs = {fo: (before[fo], after[fo]) for fo in before if before[fo] != after[fo]}
    print("sub-capabilities below original count:", len(diffs), "| rows short:", sum(b - a for b, a in diffs.values()))
    for fo, (b, a) in sorted(diffs.items(), key=lambda x: x[1][0] - x[1][1], reverse=True):
        print(f"   {fo.split(' - ',1)[1][:52]:52} {b:>6,} -> {a:>6,}  (-{b-a})")


if __name__ == "__main__":
    main()
