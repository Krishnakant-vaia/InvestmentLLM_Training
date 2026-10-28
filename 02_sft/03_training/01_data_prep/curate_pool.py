"""Curate the SFT pool: remove the "according to the document" habit, restore documents where a row needs them, balance,
add identity rows, and write DPO pairs and a guard probe. Runs on your machine (no GPU); output is a pool in the SAME
format as the S3 pool, so build_mixture.py and training read it unchanged.

    python 01_data_prep/curate_pool.py join                  # ~15 min: scored corpus (S3, streamed) -> data/curated/matched.jsonl
    python 01_data_prep/curate_pool.py build [options]       # a few minutes; writes data/curated/*  and prints the report

Per domain row (a question with no document in the pool, and reasoning + answer in "<think>..</think>answer" form):
  1. rule_clean: "The document states that X" -> "X" (reasoning and answer);
  2. source_bound: the row still needs a source (the question says "the study", or a "document"/"Section 2.12" is left);
  3. matched: the same question is in the scored corpus with its document and a quality score;
  grounded   = matched, score >= --min-score: trained WITH its document (the original reasoning stays valid);
  ungrounded = not source_bound: trained without a document, with the cleaned text;
  a row that is neither (source-bound and unmatched, or low score and source-bound) is dropped.
Then: duplicates per source chunk limited, the most common answer openers capped, question types capped, identity rows added.

Outputs in data/curated/: sft_pool_v3.jsonl (domain block, tool_calling block, general block, in that order),
dpo_pairs_doc_fix.jsonl (rejected = original, chosen = cleaned), guard_probe.jsonl (held-out, for the eval side only),
curate_report.json (counts for config.yaml data.pool.blocks and for the write-up).
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.config import load_config, resolve  # noqa: E402
from sft_common.curate import (cap_by_key, dedupe, dpo_pair, join_think, opener, rule_clean, source_bound,  # noqa: E402
                               split_think)
from sft_common.identity import identity_rows, leaks_identity, probe_questions  # noqa: E402
from sft_common.s3_fetch import fetch_file  # noqa: E402

OUT = resolve("data/curated")
SCORED_URI = "s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_kept_scored.jsonl"


def qhash(question: str) -> str:
    return hashlib.md5(question.strip().encode()).hexdigest()


def kind_of(messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    return "domain" if "VaiaGPT" in system else "tool_calling" if "functions" in system else "general"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def first_user(messages):
    return next(m["content"] for m in messages if m["role"] == "user")


def scored_question(user: str) -> str:
    """The question part of a scored-corpus user message: what follows the </document> tag."""
    return user.split("</document>")[-1].strip()


# ---- stage 1: join ----------------------------------------------------------------------

def stage_join(cfg: dict) -> None:
    import boto3
    pool = fetch_file(cfg["data"]["pool"]["s3_uri"], OUT / "pool_src.jsonl")
    wanted = {qhash(first_user(r["messages"])) for r in read_jsonl(pool) if kind_of(r["messages"]) == "domain"}
    print(f"[join] {len(wanted)} pool domain questions to find in the scored corpus", flush=True)
    bucket, key = SCORED_URI[5:].split("/", 1)
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
    n = hit = 0
    with open(OUT / "matched.jsonl", "w", encoding="utf-8") as out:
        for line in body.iter_lines():
            if not line:
                continue
            r = json.loads(line)
            n += 1
            user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            h = qhash(scored_question(user))
            if h in wanted and "<document>" in user:
                hit += 1
                out.write(json.dumps({"h": h, "user": user, "score": int(r.get("_quality_score") or 0),
                                      "qtype": r.get("_question_type"), "chunk": r.get("_chunk_id")}) + "\n")
            if n % 50000 == 0:
                print(f"[join] {n} scored rows read, {hit} matched", flush=True)
    print(f"[join] done: {n} scored rows, {hit} matched -> {OUT / 'matched.jsonl'}", flush=True)


# ---- stage 2: build ---------------------------------------------------------------------

def stage_build(cfg: dict, a) -> None:
    rng = random.Random(cfg["data"]["seed"])
    pool = OUT / "pool_src.jsonl"
    if not pool.exists():
        raise SystemExit("run `join` first (it downloads the pool)")
    matched: dict[str, dict] = {}
    if (OUT / "matched.jsonl").exists():
        for m in read_jsonl(OUT / "matched.jsonl"):
            if m["h"] not in matched or m["score"] > matched[m["h"]]["score"]:
                matched[m["h"]] = m
    stats, domain, rest = Counter(), [], {"tool_calling": [], "general": []}
    vaia_system = None
    for i, r in enumerate(read_jsonl(pool)):
        if a.limit and i >= a.limit:
            break
        msgs, kind = r["messages"], kind_of(r["messages"])
        stats[f"pool_{kind}"] += 1
        if kind == "domain":
            vaia_system = vaia_system or msgs[0]["content"]
            domain.append(msgs)
        elif any(m["role"] == "assistant" and leaks_identity(m["content"]) for m in msgs):
            stats["dropped_identity_leak"] += 1                     # a row where the assistant says it is another vendor's model
        else:
            rest[kind].append(r)

    # --- classify every domain row
    grounded, ungrounded, both, dpo = [], [], [], []
    for msgs in domain:
        system, q = msgs[0], first_user(msgs)
        reasoning, answer = split_think(msgs[-1]["content"])
        cr, ca = rule_clean(reasoning), rule_clean(answer)
        info = matched.get(qhash(q))
        ok_score = info is not None and info["score"] >= a.min_score
        stats["matched"] += info is not None
        stats["rule_changed"] += (cr != reasoning) or (ca != answer)
        sb = source_bound(q, reasoning, answer)
        stats["source_bound"] += sb
        chunk = (info or {}).get("chunk") or qhash(q)
        if info is not None and not ok_score:
            stats["dropped_low_score"] += 1
            continue
        g = ([system, {"role": "user", "content": info["user"]}, msgs[-1]], chunk, info["qtype"]) if ok_score else None
        u = ([system, {"role": "user", "content": q}, {"role": "assistant", "content": join_think(cr, ca)}], chunk,
             (info or {}).get("qtype")) if not sb else None
        pair = dpo_pair(q, msgs[-1]["content"], join_think(cr, ca), system["content"]) if u else None
        item = {"g": g, "u": u, "pair": pair, "q": q}
        if g and u:
            both.append(item)
        elif g:
            grounded.append(item)
        elif u:
            ungrounded.append(item)
        else:
            stats["dropped_source_bound_unmatched"] += 1

    # --- guard probe: held-out self-contained questions, removed from training
    rng.shuffle(ungrounded)
    probe_items, ungrounded = ungrounded[:a.probe], ungrounded[a.probe:]
    probe = [{"QA_ID": f"UG-{i:03d}", "Category": "ungrounded", "Question": it["q"]} for i, it in enumerate(probe_items, 1)]
    probe += probe_questions()

    # --- one chunk contributes a few rows, not hundreds
    grounded, both, ungrounded = (dedupe(x, lambda it: (it["g"] or it["u"])[1], a.per_chunk) for x in (grounded, both, ungrounded))
    # --- mix: grounded share of the domain rows (rows that can go either way fill the gap)
    total = len(grounded) + len(both) + len(ungrounded)
    want_g = round(a.grounded_share * total)
    rng.shuffle(both)
    x = max(0, min(len(both), want_g - len(grounded)))
    chosen = [("g", it) for it in grounded + both[:x]] + [("u", it) for it in ungrounded + both[x:]]
    stats["grounded_share_wanted"], stats["grounded_share_reached"] = a.grounded_share, round((len(grounded) + x) / max(total, 1), 3)

    # --- balance: template openers, question types
    rows = [(tag, it, (it[tag])) for tag, it in chosen]
    before = len(rows)
    rows = cap_by_key(rows, lambda t: opener(split_think(t[2][0][-1]["content"])[1]), a.opener_cap, a.seed)
    stats["dropped_by_opener_cap"] = before - len(rows)
    before = len(rows)
    rows = cap_by_key(rows, lambda t: t[2][2] or f"unknown-{id(t)}", a.qtype_cap, a.seed)
    stats["dropped_by_qtype_cap"] = before - len(rows)
    rng.shuffle(rows)
    final_domain = [t[2][0] for t in rows]
    pairs = [t[1]["pair"] for t in rows if t[0] == "u" and t[1]["pair"]]
    ident = identity_rows(vaia_system, a.identity, a.seed)

    # --- write: domain (incl. identity) block, tool_calling block, general block
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = {"domain": len(final_domain) + len(ident), "tool_calling": len(rest["tool_calling"]), "general": len(rest["general"])}
    with open(OUT / "sft_pool_v3.jsonl", "w", encoding="utf-8") as f:
        for msgs in final_domain + ident:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        for k in ("tool_calling", "general"):
            for r in rest[k]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "dpo_pairs_doc_fix.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(OUT / "guard_probe.jsonl", "w", encoding="utf-8") as f:
        for p in probe:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    types = Counter((t[2][2] or "unknown") for t in rows)
    report = {"blocks_for_config": blocks, "domain_rows": {"grounded": sum(t[0] == "g" for t in rows), "ungrounded": sum(t[0] == "u" for t in rows),
                                                              "identity": len(ident)},
              "dpo_pairs": len(pairs), "guard_probe_questions": len(probe), "question_types": dict(types.most_common()),
              "top_openers": Counter(opener(split_think(m[-1]["content"])[1]) for m in final_domain).most_common(8), **dict(stats)}
    (OUT / "curate_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["join", "build"])
    ap.add_argument("--min-score", type=int, default=8, help="matched rows below this quality score are dropped")
    ap.add_argument("--grounded-share", type=float, default=0.30, help="target share of domain rows trained with their document")
    ap.add_argument("--opener-cap", type=float, default=0.01, help="max share of domain rows starting with the same 3 words")
    ap.add_argument("--qtype-cap", type=float, default=0.25, help="max share of domain rows of one question type (known types)")
    ap.add_argument("--per-chunk", type=int, default=3, help="max rows from one source chunk")
    ap.add_argument("--identity", type=int, default=300, help="identity conversations to add")
    ap.add_argument("--probe", type=int, default=300, help="held-out self-contained questions for the guard probe")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, help="only the first N pool rows (quick test)")
    a = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_join(cfg) if a.stage == "join" else stage_build(cfg, a)


if __name__ == "__main__":
    main()
