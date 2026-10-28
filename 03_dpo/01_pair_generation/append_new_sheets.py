"""Append the Workflow Understanding and Content Generation workbooks to the merged numerical workbook, after cleaning the giveaways.

    python append_new_sheets.py [--merged PATH] [--workflow PATH] [--content PATH] [--dry-run]

Makes a backup copy of the merged workbook first (never overwritten), then adds two sheets in the merged workbook's own columns:
"4. Workflow Understanding" and "5. Content Generation". The new files carry no Pair ID, Priority, Task Type, Question Type or source
row: IDs are generated (WU-4.1-0001, CG-5.1-0001), Priority and Task Type come from the Index sheet, Source Sheet/Row point back to the
file and row, and Question Type is "Not labelled" (every question has figures; the files give only counts per capability).

Cleaning (the originals are kept in two extra columns, filled only where a pair changed). The not-preferred answers gave themselves away:
  * Workflow: "Working (flawed):" -> the preferred answer's "Task:" line plus "Worked solution:"; the preferred answer's
    "Next step in workflow: ..." line is removed (the not-preferred answers never had one).
  * Both: bracketed notes that explain the mistake, such as "(Scope 2 omitted)" or "(inverted)", are removed from the not-preferred
    answer (the Given/question text is never touched).
Duplicates are not removed here; prep_numerical.py does that across all sheets.
"""
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

import prep_numerical

HERE = Path(__file__).resolve().parent.parent.parent      # DPO_Training/: where the Excel files live
HEADERS = ["Pair ID", "Sub-Capability", "Priority", "Task Type", "Question Type", "Question", "Accepted Answer (Golden — Worked Solution)",
           "Rejected Answer", "Source Sheet", "Source Row"]
EXTRA = ["Original Accepted Answer (before cleaning)", "Original Rejected Answer (before cleaning)"]
JOBS = [("workflow", "4. Workflow Understanding", "WU"), ("content", "5. Content Generation", "CG"),
        ("agentic", "6. Agentic Reasoning", "AR"), ("learning", "7. Continuous Learning", "CL")]
ERROR_WORDS = {"inverted", "omitted", "ignored", "reversed", "unweighted", "misapplied", "flipped", "skipped", "swapped", "double-counted"}
TRAILING_NOTE = re.compile(r";[ \t]*[\w ]{0,30}\b(?:" + "|".join(sorted(ERROR_WORDS)) + r")\b\.?")     # "; pool ignored."
PAREN = re.compile(r"[ \t]*\(([^()]*)\)")


def _is_error_note(text: str) -> bool:
    """A bracket that explains the mistake: two or more words, or one known error word. "(Thu)", "(6)", "(3,4,5)" stay."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    return len(words) >= 2 or (len(words) == 1 and words[0].lower() in ERROR_WORDS)


def strip_error_notes(text: str) -> str:
    """Remove explanatory brackets outside the 'Given:' line (the question text is never touched)."""
    out = []
    for line in str(text).split("\n"):
        if line.startswith("Given:"):
            out.append(line)
            continue
        line = PAREN.sub(lambda m: "" if _is_error_note(m.group(1)) else m.group(0), line)
        out.append(TRAILING_NOTE.sub("", line).rstrip())
    return "\n".join(out)


def clean_workflow(accepted: str, rejected: str) -> tuple[str, str]:
    accepted = re.sub(r"\nNext step in workflow:[^\n]*", "", str(accepted))
    task = re.search(r"^Task:[^\n]*", accepted, re.M)
    rejected = str(rejected)
    if "Working (flawed):" in rejected:
        rejected = rejected.replace("Working (flawed):", (task.group(0) + "\n" if task else "") + "Worked solution:", 1)
    return accepted, strip_error_notes(rejected)


def clean_content(accepted: str, rejected: str) -> tuple[str, str]:
    return str(accepted), strip_error_notes(rejected)


def read_new(path: Path, prefix: str, clean, index: dict) -> list[list]:
    book = pd.ExcelFile(path)
    rows = []
    for sheet in (s for s in book.sheet_names if s != "Summary"):
        df = book.parse(sheet)
        acc = next(c for c in df.columns if c.startswith("Accepted Answer"))
        number = sheet.split()[0]                                              # "4.1"
        for i, r in enumerate(df.to_dict("records")):
            a, j = clean(r[acc], r["Rejected Answer"])
            meta = index.get(prep_numerical.slug(r["Sub-Capability"]), ("", ""))
            changed = (a != str(r[acc]), j != str(r["Rejected Answer"]))
            rows.append([f"{prefix}-{number}-{i + 1:04d}", r["Sub-Capability"], meta[0], meta[1], "Not labelled", r["Question"], a, j,
                         sheet, i + 2] + [str(r[acc]) if changed[0] else "", str(r["Rejected Answer"]) if changed[1] else ""])
    return rows


def index_lookup(book) -> dict:
    ws = book["Index"]
    head = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(head)}
    return {prep_numerical.slug(str(row[col["Sub-Capability (from taxonomy)"]])): (row[col["Priority"]], row[col["Task Type"]])
            for row in ws.iter_rows(min_row=2, values_only=True) if row[col["Sub-Capability (from taxonomy)"]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", default=str(HERE / "Finance_Numerical_QA_Dataset_Merged.xlsx"))
    ap.add_argument("--workflow", default=str(HERE / "Workflow_Understanding_QA_Pairs_FULL_2850 (2).xlsx"))
    ap.add_argument("--content", default=str(HERE / "Content_Generation_QA_Pairs_FULL_3350.xlsx"))
    ap.add_argument("--agentic", default=str(HERE / "Agentic_Reasoning_QA_Pairs_FULL_2450.xlsx"))
    ap.add_argument("--learning", default=str(HERE / "Continuous_Learning_QA_Pairs_FULL_1300.xlsx"))
    ap.add_argument("--only", help="comma-separated jobs to append (workflow, content, agentic, learning); default: those whose sheet is not in the workbook yet")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    merged = Path(a.merged)
    book = load_workbook(merged)
    index = index_lookup(book)
    sources = {"workflow": (Path(a.workflow), clean_workflow), "content": (Path(a.content), clean_content),
               "agentic": (Path(a.agentic), clean_content), "learning": (Path(a.learning), clean_content)}
    plan = []
    only = set(a.only.split(",")) if a.only else None
    for key, sheet, prefix in JOBS:
        if (only is not None and key not in only) or (only is None and sheet in book.sheetnames):
            continue
        if sheet in book.sheetnames:
            raise SystemExit(f"{sheet} is already in {merged.name}: nothing appended (run on a fresh copy to redo)")
        path, clean = sources[key]
        rows = read_new(path, prefix, clean, index)
        plan.append((sheet, rows))
        print(f"{sheet}: {len(rows)} pairs, {sum(bool(r[10]) for r in rows)} preferred and {sum(bool(r[11]) for r in rows)} not-preferred answers cleaned")
    if not plan:
        raise SystemExit("every sheet is already in the workbook: nothing to append")
    if a.dry_run:
        return
    backup = merged.with_name(f"{merged.stem}.backup-{datetime.now():%Y%m%d-%H%M}{merged.suffix}")
    shutil.copy2(merged, backup)
    ref = book["3. Financial Reasoning"]
    for sheet, rows in plan:
        ws = book.create_sheet(sheet, index=book.sheetnames.index("Summaries") if "Summaries" in book.sheetnames else None)
        ws.append(HEADERS + EXTRA)
        for c, h in enumerate(HEADERS + EXTRA, 1):
            cell, src = ws.cell(1, c), ref.cell(1, min(c, len(HEADERS)))
            cell.font, cell.fill, cell.alignment, cell.border = src.font.copy(), src.fill.copy(), src.alignment.copy(), src.border.copy()
            ws.column_dimensions[get_column_letter(c)].width = (ref.column_dimensions[get_column_letter(min(c, len(HEADERS)))].width or 30) if c <= 10 else 40
        for r in rows:
            ws.append(r)
        ws.freeze_panes = ref.freeze_panes
    book.save(merged)
    print(f"appended to {merged}\nbackup: {backup}")


if __name__ == "__main__":
    main()
