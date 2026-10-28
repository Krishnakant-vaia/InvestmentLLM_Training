"""Step 6: scored pairs (ALL of them) -> dpo_pairs.jsonl in the format 03_dpo/02_training/02_training/dpo_train.py reads, plus a bias report.

    python build_dpo_pairs.py --scored scored_pairs.jsonl --out dpo_pairs.jsonl --report report.json

Each line: {"prompt": [system, user], "preferred": "<think>..</think>\\n\\nanswer", "not_preferred": ..., "source": ..., + metadata}.
The user turn is the document followed by the question (the layout of the eval bank's Context + Question); the system turn
is the VaiaGPT prompt every SFT row carries (system_prompt.txt, copied from the SFT training data).
"""
import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import config

SOURCE_LABELS = {"Private Markets Terminology": "pm_terminology", "Credit Rating Replication": "credit_rating",
                 "CIBIL Credit Scoring": "cibil", "tear sheets": "tear_sheets"}
HEDGES = re.compile(r"\b(however|although|unless|whereas|on the other hand|depends on|that said)\b", re.I)
HEADING = re.compile(r"^\s*(#{1,6}\s|\*\*[^*]+\*\*\s*$)", re.M)


def to_dpo_pair(row: dict, system_prompt: str) -> dict:
    """One selected pair in trainer format. Thinking and answer are joined the way the SFT chat template renders them."""
    return {"prompt": [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": "\n\n".join(p for p in (row["document"], row["question"]) if p)}],    # no document: numerical
            "preferred": "\n\n".join(p for p in (row["preferred_thinking"].strip(), row["preferred"].strip()) if p),           # no <think>: numerical
            "not_preferred": "\n\n".join(p for p in (row["not_preferred_thinking"].strip(), row["not_preferred"].strip()) if p),
            "source": SOURCE_LABELS.get(row["folder"], row["folder"]),
            "question_type": row["question_type"], "defect_type": row["defect_type"],
            "chunk_id": row["chunk_id"], "call_id": row["call_id"],
            "passes_all_gates": row.get("passes_all_gates"), "flags": row.get("flags", []),
            "judge_preferred": row.get("judge_preferred"), "judge_not_preferred": row.get("judge_not_preferred")}


def bias_report(rows: list[dict]) -> dict:
    """Cheap checks for the surface tells and distribution skews listed in the bias plan."""
    words = lambda t: len(t.split())  # noqa: E731
    c_len, r_len = [words(r["preferred"]) for r in rows], [words(r["not_preferred"]) for r in rows]
    longer = sum(c > r for c, r in zip(c_len, r_len)) / max(1, len(rows))
    hedge = lambda texts: sum(len(HEDGES.findall(t)) for t in texts) / max(1, sum(words(t) for t in texts)) * 1000  # noqa: E731
    h_c, h_r = hedge([r["preferred"] for r in rows]), hedge([r["not_preferred"] for r in rows])
    head = lambda texts: sum(bool(HEADING.search(t)) for t in texts) / max(1, len(texts))  # noqa: E731
    openings = Counter(" ".join(r["question"].split()[:3]).lower() for r in rows)
    report = {"pairs": len(rows),
              "by_source": dict(Counter(r["folder"] for r in rows)),
              "by_question_type": dict(Counter(r["question_type"] for r in rows)),
              "by_defect": dict(Counter(r["defect_type"] for r in rows)),
              "passes_all_gates": sum(bool(r.get("passes_all_gates")) for r in rows),
              "flag_counts": dict(Counter(f.split(" (")[0].split(" also fails")[0] for r in rows for f in r.get("flags", []))),
              "preferred_longer_share": round(longer, 3),
              "preferred_longer_share_clean_pairs": round(sum(len(r["preferred"].split()) > len(r["not_preferred"].split()) for r in rows if r.get("passes_all_gates"))
                                                       / max(1, sum(bool(r.get("passes_all_gates")) for r in rows)), 3),
              "median_words": {"preferred": st.median(c_len) if rows else 0, "not_preferred": st.median(r_len) if rows else 0},
              "hedge_words_per_1000": {"preferred": round(h_c, 2), "not_preferred": round(h_r, 2)},
              "heading_share": {"preferred": round(head([r["preferred"] for r in rows]), 3), "not_preferred": round(head([r["not_preferred"] for r in rows]), 3)},
              "top_question_openings": openings.most_common(8),
              "distinct_chunks": len({r["chunk_id"] for r in rows}), "warnings": []}
    if rows and not 0.3 <= longer <= 0.7:
        report["warnings"].append(f"preferred is the longer answer in {longer:.0%} of pairs: the model may learn length, not quality")
    if h_r and h_c and not 0.67 <= h_c / h_r <= 1.5:
        report["warnings"].append("hedge-word rate differs between preferred and not-preferred: a surface tell")
    if rows and abs(head([r["preferred"] for r in rows]) - head([r["not_preferred"] for r in rows])) > 0.15:
        report["warnings"].append("headings appear much more in one side: a surface tell")
    if rows and openings.most_common(1)[0][1] / len(rows) > 0.1:
        report["warnings"].append("one question opening covers more than 10% of the questions")
    return report


def build(scored: Path, out: Path, report_path: Path, system_prompt: Path = config.HERE / "prompts" / "system_prompt.txt") -> dict:
    """scored_pairs.jsonl (every pair) -> dpo_pairs.jsonl and the bias report (returned and saved)."""
    system = Path(system_prompt).read_text(encoding="utf-8")
    with open(scored, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(to_dpo_pair(r, system), ensure_ascii=False) + "\n" for r in rows)
    report = bias_report(rows)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--system-prompt", default=str(config.HERE / "prompts" / "system_prompt.txt"))
    a = ap.parse_args()
    report = build(Path(a.scored), Path(a.out), Path(a.report), Path(a.system_prompt))
    print(json.dumps(report, indent=2))
    print(f"[build] {report['pairs']} pairs -> {a.out}")


if __name__ == "__main__":
    main()
