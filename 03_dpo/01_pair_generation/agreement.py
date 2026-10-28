"""Second-opinion check of the judge: how often do two judges (e.g. gpt-oss-120b and another model) reach the same verdict?

    python judge.py --in judge_in.jsonl --out judge_out.jsonl                                   # the main judge
    python judge.py --in judge_in.jsonl --out judge_out_2.jsonl --url https://... --model NAME --api-key-env KEY_VAR   # the other one
    python agreement.py judge_out.jsonl judge_out_2.jsonl

Both files must come from the same judge_in.jsonl (same ids and a/b sides). Prints agreement on pass/fail per answer, on the
gate lists, and the ids where the two disagree on whether a pair is usable. Read those disagreements before scaling up:
below ~85% agreement on pass/fail, the judge prompt (or one of the models) is not reliable enough to select data.
"""
import argparse
import json
from pathlib import Path


def read(path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}


def passes(v: dict) -> bool:
    return v["score"] >= 7


def compare(a: dict[int, dict], b: dict[int, dict]) -> dict:
    ids = sorted(set(a) & set(b))
    sides = [(i, s) for i in ids for s in ("a", "b")]
    same_pass = sum(passes(a[i][s]) == passes(b[i][s]) for i, s in sides)
    same_gates = sum(set(a[i][s]["failed_gates"]) - {"G8"} == set(b[i][s]["failed_gates"]) - {"G8"} for i, s in sides)
    # a pair is "usable" when exactly one side passes (the other fails): the situation the selection needs
    usable = lambda r: passes(r["a"]) != passes(r["b"])  # noqa: E731
    disagree = [i for i in ids if usable(a[i]) != usable(b[i])]
    return {"pairs_compared": len(ids), "answers_compared": len(sides),
            "pass_fail_agreement": round(same_pass / max(1, len(sides)), 3),
            "failed_gates_agreement": round(same_gates / max(1, len(sides)), 3),
            "usable_pair_agreement": round(1 - len(disagree) / max(1, len(ids)), 3), "disagreeing_pair_ids": disagree[:30]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first")
    ap.add_argument("second")
    a = ap.parse_args()
    print(json.dumps(compare(read(Path(a.first)), read(Path(a.second))), indent=2))


if __name__ == "__main__":
    main()
