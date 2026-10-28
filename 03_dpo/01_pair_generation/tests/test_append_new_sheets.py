"""Cleaning of the giveaways in the two new workbooks."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import append_new_sheets as ans  # noqa: E402

ACC = "Given: Request list has 153 items.\nTask: How many days?\nWorked solution:\n  1. 83 / 4.1 = 20.2\nNext step in workflow: Record the finding.\nAnswer: 20.2 days. "
REJ = "Given: Request list has 153 items.\nWorking (flawed):\n  1. 83 / 17 = 4.9 (existing provision ignored).\nAnswer: 4.9 days (confused total days with run-rate). "


def test_workflow_markers_are_removed_and_the_structure_matches():
    a, r = ans.clean_workflow(ACC, REJ)
    assert "Next step" not in a and a.endswith("Answer: 20.2 days. ".rstrip()) or "Next step" not in a
    assert "flawed" not in r and "Task: How many days?\nWorked solution:" in r
    assert "ignored" not in r and "confused" not in r and r.splitlines()[-1] == "Answer: 4.9 days."


def test_error_notes_go_but_dates_and_numbered_references_stay():
    assert ans.strip_error_notes("LTV = 145.2% (inverted); cover 0.69x.") == "LTV = 145.2%; cover 0.69x."
    assert ans.strip_error_notes("Due = 01-Sep-2026 (Tue); see (6) and (3)/(4).") == "Due = 01-Sep-2026 (Tue); see (6) and (3)/(4)."
    assert ans.strip_error_notes("Given: a (b c d) stays\nTotal (Scope 2 omitted) = 5") == "Given: a (b c d) stays\nTotal = 5"


def test_trailing_unbracketed_notes_go():
    assert ans.strip_error_notes("Investor ownership = 43÷193 = 22.28%; pool ignored.\nAnswer: 22.28%.") == "Investor ownership = 43÷193 = 22.28%\nAnswer: 22.28%."
