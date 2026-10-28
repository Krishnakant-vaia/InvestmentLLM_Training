"""Analyse the pilot passes: score distribution, run-to-run noise, tie band at the
150K/200K cutoffs (scaled to the full 647,066), composition, failure codes.

  python3 scripts/analyze_scoring_pilot.py scoring_pilot/run1 scoring_pilot/run2 scoring_pilot/run3
"""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FULL = 647066


def load(run_dir):
    rows = {}
    with open(f"{run_dir}/scored.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["_quality_src_index"]] = r
    unscored = 0
    try:
        with open(f"{run_dir}/unscored.jsonl", encoding="utf-8") as f:
            unscored = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    return rows, unscored


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def main(dirs):
    runs = [load(d) for d in dirs]
    base, unscored = runs[0]
    n = len(base)
    print(f"run1: scored={n} unscored={unscored}")

    hist = Counter(r["_quality_score"] for r in base.values())
    print("\nscore histogram (run1)")
    for s in range(10, 0, -1):
        print(f"  {s:>2}: {hist[s]:>5} {pct(hist[s], n):>6} " + "#" * (hist[s] * 60 // max(n, 1)))

    print("\ncutoff bands (run1, scaled to full corpus)")
    cum = 0
    for s in range(10, 0, -1):
        prev = cum
        cum += hist[s]
        scaled_lo, scaled_hi = prev / n * FULL, cum / n * FULL
        marks = [k for k in (150_000, 200_000) if scaled_lo < k <= scaled_hi]
        note = ""
        for k in marks:
            note += f"  <- {k // 1000}K cutoff falls here, tie band ~{scaled_hi - scaled_lo:,.0f} records, need ~{k - scaled_lo:,.0f} of them"
        print(f"  score>={s:>2}: ~{scaled_hi:>9,.0f} records{note}")

    if len(runs) > 1:
        print("\nrun-to-run agreement vs run1")
        for i, (other, _) in enumerate(runs[1:], start=2):
            common = sorted(set(base) & set(other))
            exact = sum(base[k]["_quality_score"] == other[k]["_quality_score"] for k in common)
            within1 = sum(abs(base[k]["_quality_score"] - other[k]["_quality_score"]) <= 1 for k in common)
            binary = sum((base[k]["_quality_score"] >= 7) == (other[k]["_quality_score"] >= 7) for k in common)
            mean_abs = sum(abs(base[k]["_quality_score"] - other[k]["_quality_score"]) for k in common) / max(len(common), 1)
            print(f"  run{i}: n={len(common)} exact={pct(exact, len(common))} +-1={pct(within1, len(common))} "
                  f"pass/fail(>=7)={pct(binary, len(common))} mean|diff|={mean_abs:.2f}")

    print("\ncomposition by _question_type: n, mean score, share scoring >=8")
    by_q = {}
    for r in base.values():
        by_q.setdefault(r.get("_question_type", "?"), []).append(r["_quality_score"])
    for q, v in sorted(by_q.items(), key=lambda kv: -len(kv[1])):
        print(f"  {q:<28} n={len(v):>5} mean={sum(v) / len(v):.2f} >=8: {pct(sum(x >= 8 for x in v), len(v))}")

    print("\nprimary category: n, mean score")
    by_c = {}
    for r in base.values():
        cats = r["_quality_categories"]
        by_c.setdefault(cats[0] if cats else "(none)", []).append(r["_quality_score"])
    for c, v in sorted(by_c.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:<52} n={len(v):>5} mean={sum(v) / len(v):.2f}")

    print("\nfailure reasons among score<=6")
    fails = Counter(r["_quality_reason"] for r in base.values() if r["_quality_score"] <= 6)
    for k, v in fails.most_common():
        print(f"  {k:<32} {v}")

    anomalies = Counter(a for r in base.values() for a in r.get("_quality_anomalies", []))
    print(f"\nanomalies: {dict(anomalies)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["scoring_pilot/run1"])
