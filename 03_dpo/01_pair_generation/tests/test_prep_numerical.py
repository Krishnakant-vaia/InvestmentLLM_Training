"""The numerical Excel preparation: reading, duplicate removal (nothing lost), per-capability files."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import prep_numerical as pn  # noqa: E402

COLS = ["Pair ID", "Sub-Capability", "Priority", "Task Type", "Question Type", "Question", "Accepted Answer (Golden - Worked Solution)",
        "Rejected Answer", "Source Sheet", "Source Row"]


def make_excel(path: Path) -> None:
    def rows(prefix, cap, items):
        return [[f"{prefix}-{i:04d}", cap, "High", "Worked numeric", "Numerical", q, a, r, "src", i + 1] for i, (q, a, r) in enumerate(items)]
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="Index", index=False)
        pd.DataFrame(rows("DK", "Accounting [NEW - Debt]", [("What is 2+2?", "4", "5"), ("What is 3+3?", "6", "7"),
                                                              ("  what IS 2+2? ", "4", "9")]), columns=COLS).to_excel(w, sheet_name="1. Domain Knowledge", index=False)
        pd.DataFrame(rows("FR", "DCF", [("What is 3+3?", "six", "7"), ("What is 4+4?", "8", "9")]), columns=COLS).to_excel(w, sheet_name="3. Financial Reasoning", index=False)
        pd.DataFrame({"y": [1]}).to_excel(w, sheet_name="Summaries", index=False)


def test_slug_drops_tags_and_punctuation():
    assert pn.slug("Credit rating methodology fluency (Moody's/S&P/Fitch) [NEW - Debt]") == "credit-rating-methodology-fluency-moody-s-s-p-fitch"
    assert pn.slug("1. Domain Knowledge") == "1-domain-knowledge"


def test_duplicates_are_removed_first_copy_kept_across_sheets_and_nothing_is_lost(tmp_path):
    excel, out = tmp_path / "in.xlsx", tmp_path / "out"
    make_excel(excel)
    report = pn.prepare(excel, out)
    assert report["pairs_read"] == 5 and report["pairs_kept"] == 3 and report["duplicates_removed"] == 2
    kept = [json.loads(l) for f in sorted(out.glob("*/*.jsonl")) for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
    assert sorted(p["pair_id"] for p in kept) == ["DK-0000", "DK-0001", "FR-0001"]            # "2+2" kept once, "3+3" kept from the first sheet
    removed = [json.loads(l) for l in (out / "duplicates_removed.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    assert {(r["pair_id"], r["duplicate_of"]) for r in removed} == {("DK-0002", "DK-0000"), ("FR-0000", "DK-0001")}
    assert report["removed_by_sheet"] == {"1. Domain Knowledge": 1, "3. Financial Reasoning": 1}


def test_groups_whose_accepted_answers_disagree_are_reported(tmp_path):
    excel, out = tmp_path / "in.xlsx", tmp_path / "out"
    make_excel(excel)
    report = pn.prepare(excel, out)
    groups = report["duplicate_groups_whose_preferred_answers_differ"]
    assert [g["pair_ids"] for g in groups] == [["DK-0001", "FR-0000"]]                        # 6 vs six; the "2+2" copies agree


def test_one_file_per_capability_with_the_trainer_names(tmp_path):
    excel, out = tmp_path / "in.xlsx", tmp_path / "out"
    make_excel(excel)
    pn.prepare(excel, out, count_tokens=lambda t: len(t.split()))
    caps = json.loads((out / "capabilities.json").read_text(encoding="utf-8"))
    assert [(c["capability"], c["pairs"], c["removed_as_duplicates"]) for c in caps] == [("Accounting [NEW - Debt]", 2, 1), ("DCF", 1, 1)]
    first = json.loads((out / caps[0]["file"]).read_text(encoding="utf-8").split("\n")[0])
    assert set(first) == {"pair_id", "sheet", "capability", "priority", "task_type", "question_type", "question", "preferred", "not_preferred", "source"}
    assert caps[0]["tokens"]["total"] > 0
