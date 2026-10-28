"""Numerical QA Excel -> one JSONL file per capability, with duplicate questions removed. CPU only; the Excel is never modified.

    python prep_numerical.py [--excel PATH] [--out work/numerical] [--tokenizer-dir PATH]

Input: the merged workbook (sheets "1. Domain Knowledge", "2. Enterprise Knowledge", "3. Financial Reasoning"; columns Pair ID,
Sub-Capability, Priority, Task Type, Question Type, Question, Accepted Answer, Rejected Answer, Source Sheet, Source Row).
Output under --out:
    <sheet-slug>/<capability-slug>.jsonl   the pairs kept, one per line: pair_id, sheet, capability, priority, task_type,
                                           question_type, question, accepted (preferred), rejected (not-preferred), source
    duplicates_removed.jsonl               every removed row with the pair it duplicates (nothing is lost)
    dedupe_report.json                     counts, and the duplicate groups whose accepted answers disagree (review these)
    capabilities.json                      per capability: pairs kept/removed, and token sizes if a tokenizer is given (for the ETA)

A duplicate is a question with the same text after lower-casing and collapsing whitespace. The first occurrence is kept (sheet
order, then row order); later ones are removed, across sheets too.
"""
import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import config

DEFAULT_EXCEL = config.HERE.parent.parent / "Finance_Numerical_QA_Dataset_Merged.xlsx"      # DPO_Training/
SHEET_PATTERN = re.compile(r"^\d+\.\s")                                  # "1. Domain Knowledge"; Index and Summaries are skipped
COLUMNS = {"Pair ID": "pair_id", "Sub-Capability": "capability", "Priority": "priority", "Task Type": "task_type",
           "Question Type": "question_type", "Question": "question", "Source Sheet": "source_sheet", "Source Row": "source_row"}


def slug(text: str) -> str:
    """File-name-safe name: bracketed tags such as [NEW - Debt] and punctuation are dropped."""
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:70]


def norm(text) -> str:
    return " ".join(str(text).lower().split())


def read_pairs(excel: Path) -> list[dict]:
    """Every pair of the workbook in sheet and row order."""
    import pandas as pd                                                  # only the local prep needs it, not the GPU box
    book = pd.ExcelFile(excel)
    rows = []
    for sheet in book.sheet_names:
        if not SHEET_PATTERN.match(sheet):
            continue
        df = book.parse(sheet)
        accepted = next(c for c in df.columns if c.startswith("Accepted Answer"))
        for rec in df.to_dict("records"):
            row = {new: rec[old] for old, new in COLUMNS.items()}
            row.update(sheet=sheet, preferred=rec[accepted], not_preferred=rec["Rejected Answer"])
            rows.append({k: ("" if v != v else v) for k, v in row.items()})           # NaN -> ""
    return rows


def dedupe(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(kept, removed, conflicts). `removed` rows carry `duplicate_of`; `conflicts` lists groups whose accepted answers differ."""
    first: dict[str, dict] = {}
    kept, removed, groups = [], [], defaultdict(list)
    for r in rows:
        key = norm(r["question"])
        if key in first:
            removed.append({**r, "duplicate_of": first[key]["pair_id"]})
        else:
            first[key] = r
            kept.append(r)
        groups[key].append(r)
    conflicts = [{"question": g[0]["question"], "pair_ids": [x["pair_id"] for x in g],
                  "preferred_answers_differ": len({norm(x["preferred"]) for x in g}) > 1}
                 for g in groups.values() if len(g) > 1 and len({norm(x["preferred"]) for x in g}) > 1]
    return kept, removed, conflicts


def pair_line(r: dict) -> dict:
    return {"pair_id": r["pair_id"], "sheet": r["sheet"], "capability": r["capability"], "priority": r["priority"],
            "task_type": r["task_type"], "question_type": r["question_type"], "question": r["question"],
            "preferred": r["preferred"], "not_preferred": r["not_preferred"], "source": f"{r['source_sheet']} row {r['source_row']}"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def prepare(excel: Path, out: Path, count_tokens=None) -> dict:
    rows = read_pairs(excel)
    kept, removed, conflicts = dedupe(rows)
    by_cap: dict[tuple, list[dict]] = defaultdict(list)
    for r in kept:
        by_cap[(r["sheet"], r["capability"])].append(r)
    removed_by_cap = Counter((r["sheet"], r["capability"]) for r in removed)
    caps = []
    for (sheet, capability), items in by_cap.items():
        path = Path(slug(sheet)) / f"{slug(capability)}.jsonl"
        write_jsonl(out / path, [pair_line(r) for r in items])
        entry = {"sheet": sheet, "capability": capability, "file": path.as_posix(), "pairs": len(items),
                 "removed_as_duplicates": removed_by_cap[(sheet, capability)],
                 "question_types": dict(Counter(r["question_type"] for r in items))}
        if count_tokens:
            tq, ta, tr = ([count_tokens(str(r[k])) for r in items] for k in ("question", "preferred", "not_preferred"))
            entry["tokens"] = {"question_median": st.median(tq), "preferred_median": st.median(ta), "not_preferred_median": st.median(tr),
                               "total": sum(tq) + sum(ta) + sum(tr)}
        caps.append(entry)
    for sheet, capability in set(removed_by_cap) - set(by_cap):                       # a capability that was only duplicates
        caps.append({"sheet": sheet, "capability": capability, "file": None, "pairs": 0, "removed_as_duplicates": removed_by_cap[(sheet, capability)]})
    write_jsonl(out / "duplicates_removed.jsonl", [{**pair_line(r), "duplicate_of": r["duplicate_of"]} for r in removed])
    report = {"pairs_read": len(rows), "pairs_kept": len(kept), "duplicates_removed": len(removed),
              "removed_by_sheet": dict(Counter(r["sheet"] for r in removed)),
              "kept_by_sheet": dict(Counter(r["sheet"] for r in kept)), "kept_by_question_type": dict(Counter(r["question_type"] for r in kept)),
              "duplicate_groups_whose_preferred_answers_differ": conflicts}
    (out / "dedupe_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "capabilities.json").write_text(json.dumps(sorted(caps, key=lambda c: (c["sheet"], c["capability"])), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_judge_input(rows: list[dict], seed: int = 3407):
    """(judge lines, key) for the numerical judge: question, the golden as reference_solution, and the two answers on random sides."""
    import random
    rng, lines, key = random.Random(seed), [], []
    for i, r in enumerate(rows):
        accepted_first = rng.random() < 0.5
        a, b = (r["preferred"], r["not_preferred"]) if accepted_first else (r["not_preferred"], r["preferred"])
        lines.append({"id": i, "question": r["question"], "reference_solution": r["preferred"], "answer_a": a, "answer_b": b})
        key.append({"pair_id": r["pair_id"], "preferred_side": "a" if accepted_first else "b"})
    return lines, key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", default=str(DEFAULT_EXCEL))
    ap.add_argument("--out", default=str(config.HERE / "work" / "numerical"))
    ap.add_argument("--tokenizer-dir", help="a tokenizer folder: adds token sizes per capability to capabilities.json (for the ETA)")
    a = ap.parse_args()
    count = None
    if a.tokenizer_dir:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
        count = lambda t: len(tok(t)["input_ids"])  # noqa: E731
    report = prepare(Path(a.excel), Path(a.out), count)
    print(json.dumps({k: v for k, v in report.items() if k != "duplicate_groups_whose_preferred_answers_differ"}, indent=2))
    print(f"duplicate groups whose accepted answers differ (to review): {len(report['duplicate_groups_whose_preferred_answers_differ'])}")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
