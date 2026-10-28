"""DPO training pairs -> SFT rows in the format of the 178k SFT pool.

    python dpo_to_sft.py [--check-pool] [--dry-run]

Reads qa_pairs/training/dpo_train.jsonl and writes, beside it under qa_pairs/sft/:
    sft_from_dpo.jsonl                      rows whose answer already carries its <think> block (documents, tear sheets, CIMs)
    sft_from_dpo_numerical_pending.jsonl    the numerical rows, answer unchanged, NO <think> block yet (merged in once the reasoning exists)
    sft_from_dpo_provenance.jsonl           one line per row of both files: file, line, pair_id, origin, source, question_type
    manifest.json                           counts and the checks that were made

Row format = the pool's: {"messages": [system, user, assistant]} and nothing else. The assistant content is the pair's `preferred` word for
word. The system prompt is the pool's domain prompt (the first row's, which --check-pool proves is the most common one in the pool); the DPO system prompt is dropped.
The not-preferred answers are not used.
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone

import config
import s3util

POOL_KEY = "Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_finetune_pool_178k.jsonl"
TRAIN_KEY = f"{config.OUT_ROOT}training/dpo_train.jsonl"
OUT = f"{config.OUT_ROOT}sft/"
MAIN, PENDING, PROV, MANIFEST = "sft_from_dpo.jsonl", "sft_from_dpo_numerical_pending.jsonl", "sft_from_dpo_provenance.jsonl", "manifest.json"


def pool_system_prompt(s3, check_all: bool) -> tuple[str, dict]:
    """The pool's domain system prompt: the first row's. With check_all the whole pool is read and the prompt must be the most common one
    (the pool also holds tool-calling and general rows that carry their own, shorter prompts); its share is reported."""
    body = s3.get_object(Bucket=config.BUCKET, Key=POOL_KEY)["Body"]
    counts, first = Counter(), None
    for line in body.iter_lines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        sys_msg = msgs[0]["content"] if msgs[0]["role"] == "system" else None
        first = first if first is not None else sys_msg
        counts[sys_msg] += 1
        if not check_all:
            break
    info = {"pool_rows_read": sum(counts.values())}
    if check_all:
        top, n_top = counts.most_common(1)[0]
        if top != first:
            raise SystemExit("the first row's system prompt is not the pool's most common one; choose the prompt explicitly")
        info.update({"distinct_system_prompts": len(counts), "rows_with_the_chosen_prompt": n_top, "chosen_prompt_chars": len(first)})
    return first, info


def convert(pair: dict, system: str) -> dict:
    """One SFT row from one DPO pair: the pool's system prompt, the pair's user turn, the preferred answer unchanged."""
    prompt = pair["prompt"]
    if [m["role"] for m in prompt] != ["system", "user"]:
        raise ValueError(f"unexpected prompt roles in {pair.get('pair_id')}: {[m['role'] for m in prompt]}")
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt[1]["content"]},
                         {"role": "assistant", "content": pair["preferred"]}]}


def is_numerical(pair: dict) -> bool:
    return str(pair.get("origin", "")).startswith("numerical")


def split_rows(pairs: list[dict], system: str):
    """(main rows, pending rows, provenance lines) in the order of the DPO file."""
    main, pending, prov = [], [], []
    for p in pairs:
        row = convert(p, system)
        bucket, name = (pending, PENDING) if is_numerical(p) else (main, MAIN)
        bucket.append(row)
        prov.append({"file": name, "line": len(bucket) - 1, "pair_id": p.get("pair_id"), "origin": p.get("origin"), "source": p.get("source"),
                     "question_type": p.get("question_type")})
    return main, pending, prov


def validate(pairs: list[dict], main: list[dict], pending: list[dict], system: str) -> dict:
    """Raises on any problem; returns the checks that passed."""
    assert len(main) + len(pending) == len(pairs), "row count differs from the pair count"
    it_main, it_pending = iter(main), iter(pending)
    for p in pairs:
        row = next(it_pending if is_numerical(p) else it_main)
        m = row["messages"]
        assert list(row) == ["messages"] and [x["role"] for x in m] == ["system", "user", "assistant"], "row shape"
        assert all(list(x) == ["role", "content"] for x in m), "message keys"
        assert m[0]["content"] == system, "system prompt differs from the pool's"
        assert m[1]["content"] == p["prompt"][1]["content"], "user turn changed"
        assert m[2]["content"] == p["preferred"], "answer is not word for word the preferred answer"
    think = lambda rows: sum(r["messages"][2]["content"].lstrip().startswith("<think>") and "</think>" in r["messages"][2]["content"] for r in rows)
    assert think(main) == len(main), "a main row has no <think> block"
    assert think(pending) == 0, "a pending row already has a <think> block"
    return {"rows_in": len(pairs), "main_rows": len(main), "pending_rows": len(pending), "main_rows_with_think_block": think(main),
            "pending_rows_with_think_block": think(pending), "system_prompt_identical_to_pool": True, "answers_word_for_word": True}


def dumps(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-pool", action="store_true", help="read the whole pool to prove it holds one system prompt (a few minutes)")
    ap.add_argument("--dry-run", action="store_true", help="convert and validate; write nothing")
    a = ap.parse_args()
    s3 = s3util.client()
    system, pool_check = pool_system_prompt(s3, a.check_pool)
    pairs = [json.loads(l) for l in s3.get_object(Bucket=config.BUCKET, Key=TRAIN_KEY)["Body"].iter_lines() if l.strip()]
    main, pending, prov = split_rows(pairs, system)
    checks = validate(pairs, main, pending, system)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source_file": TRAIN_KEY, "pool_file": POOL_KEY,
                "pool_check": pool_check, "checks": checks, "files": {MAIN: len(main), PENDING: len(pending), PROV: len(prov)},
                "by_source": {"main": dict(Counter(p["source"] for p in pairs if not is_numerical(p))),
                              "pending_numerical": dict(Counter(p["origin"] for p in pairs if is_numerical(p)))},
                "note": "Numerical rows await a <think> block; merge them into sft_from_dpo.jsonl once the reasoning is written."}
    print(json.dumps({k: manifest[k] for k in ("pool_check", "checks", "files")}, indent=1))
    if a.dry_run:
        print("DRY RUN: nothing written")
        return 0
    s3util.put_text(s3, OUT + MAIN, dumps(main))
    s3util.put_text(s3, OUT + PENDING, dumps(pending))
    s3util.put_text(s3, OUT + PROV, dumps(prov))
    s3util.put_text(s3, OUT + MANIFEST, json.dumps(manifest, indent=2))
    print(f"written under s3://{config.BUCKET}/{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
