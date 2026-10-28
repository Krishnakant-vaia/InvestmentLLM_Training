"""Cleanup of the converted tear sheets: the Trend column and the conversion leftovers go, no figure changes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import convert_tearsheets as ct  # noqa: E402

NL = "\n"
SHEET = NL.join([
    "# SYNTHETIC · BANKING", "Example Tear Sheet, prepared on 10/19/2026", "", "# Sarthi Finance", "",
    "**CEO**", "****Head of Strategy****", "Olivia Cohen", "", "[signature]", "", "---", "", "# SYNTHETIC · BANKING", "", "## Board", "",
    "| Metric | LTM current | YoY growth | Trend |", "| ------ | ----------- | ---------- | ----- |",
    "| Revenue | $2,033,181 | 38.21% | 📈 |", "| Cash | $5,589,342 | - | ↓ |", "",
    "# SYNTHETIC · BANKING", "Footer: Growth = change ÷ |prior|."])


def test_the_trend_column_and_its_icons_are_removed_whole():
    out = ct.clean_markdown(SHEET)
    assert "Trend" not in out and not any(s in out for s in ("📈", "📉", "↑", "↓"))
    assert "| Revenue | $2,033,181 | 38.21% |" in out and "| Metric | LTM current | YoY growth |" in out


def test_leftovers_go_and_the_first_banner_stays():
    out = ct.clean_markdown(SHEET)
    assert "[signature]" not in out and "****" not in out and "**Head of Strategy**" in out
    assert out.count("# SYNTHETIC") == 1 and out.startswith("# SYNTHETIC · BANKING")
    assert "Footer: Growth = change ÷ |prior|." in out                         # a legitimate footer is content, not a leftover


def test_cleaning_changes_no_figure_and_is_idempotent():
    once = ct.clean_markdown(SHEET)
    assert ct.clean_markdown(once) == once
    assert set(ct.numbers(SHEET)) == set(ct.numbers(once))


def test_a_table_without_a_trend_column_is_left_alone():
    t = NL.join(["| Round | Cash |", "| --- | --- |", "| Seed | $291,000 |"])
    assert ct.drop_trend_column(t) == t
