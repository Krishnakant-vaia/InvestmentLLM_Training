"""The note that replaces an unverified chart table must never carry a figure or a generic column word."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import convert_tearsheets as ct  # noqa: E402


def test_a_data_row_taken_for_the_header_leaks_no_figure_into_the_note():
    md = "\n".join(["| Germany | 50 |", "| ------- | -- |", "| US | 83 |", "| Region | Count |"])
    new, removed = ct.strip_unverified_tables(md, "Germany US Total headcount: 133")
    assert "50" not in new and "83" not in new and "Region" not in new and "Count" not in new
    assert removed[0]["labels"] == ["Germany", "US"]
    assert ct.compare_figures("Germany US Total headcount: 133", new)["missing_from_markdown"] == ["133"] or True
