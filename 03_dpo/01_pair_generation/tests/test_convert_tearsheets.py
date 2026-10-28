"""Tear sheet conversion: file selection, the figure check, and key rotation (no network)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import convert_tearsheets as ct  # noqa: E402


def test_collects_the_startup_and_sector_sheets_and_skips_the_example(tmp_path):
    for rel in ("02_B_Tear_Sheet.pdf", "01_A_Tear_Sheet.pdf", "Tear sheet example 1.pdf", "sectors/Banking_NBFC/40_X_Tear_Sheet.pdf",
                "sectors/Banking_NBFC/structured.json", "sectors/all_sectors_structured.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    names = [p.name for p in ct.collect_pdfs(tmp_path)]
    assert names == ["01_A_Tear_Sheet.pdf", "02_B_Tear_Sheet.pdf", "40_X_Tear_Sheet.pdf"]
    assert ct.flat_name(Path("48_Ganga_Hydro_&_Storage_Tear_Sheet.pdf")) == "48_Ganga_Hydro_and_Storage_Tear_Sheet.md"


def test_figures_match_across_formats_and_a_dropped_or_invented_figure_is_reported():
    pdf = "LATEST VALUATION\n$21,984,590\nOWNERSHIP\n6.65%\nRESERVES $1M\nprepared on 10/19/2026"
    md = "| LATEST VALUATION | $21,984,590 |\n| OWNERSHIP | 6.65% |\n| RESERVES | $1M |\nprepared on 10/19/2026"
    assert ct.compare_figures(pdf, md)["clean"]
    broken = ct.compare_figures(pdf, md.replace("6.65%", "6.5%"))
    assert not broken["clean"] and broken["missing_from_markdown"] == ["6.65"] and broken["not_in_pdf"] == ["6.5"]


def test_markdown_comes_from_the_full_text_or_else_the_pages():
    assert ct.markdown_of({"markdown_full": "  # A\n"}) == "# A\n"
    assert ct.markdown_of({"markdown_full": "", "markdown": {"pages": [{"markdown": "p1"}, {"markdown": "p2"}]}}) == "p1\n\np2\n"
    with pytest.raises(ValueError):
        ct.markdown_of({"markdown_full": None, "markdown": None})


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body or {}
        self.ok = code < 400

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeHttp:
    def __init__(self, dead):
        self.dead, self.used = dead, []

    def post(self, url, headers, files, data, timeout):
        key = headers["Authorization"].split()[-1]
        self.used.append(key)
        return Resp(402) if key in self.dead else Resp(200, {"id": "job1"})

    def get(self, url, headers, params=None, timeout=0):
        if params:
            return Resp(200, {"markdown_full": "# ok"})
        return Resp(200, {"job": {"status": "COMPLETED"}})


def test_a_key_without_credit_is_dropped_and_the_next_one_is_used(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    http = FakeHttp(dead={"k1", "k2"})
    p = ct.Parser(["k1", "k2", "k3"], "cost_effective", session=http)
    assert ct.markdown_of(p.parse(pdf)) == "# ok\n"
    assert http.used == ["k1", "k2", "k3"] and p.i == 2
    p.parse(pdf)
    assert http.used[-1] == "k3"                                            # the dead keys are not tried again


def test_every_key_dead_stops_cleanly(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(ct.NoKeyLeft):
        ct.Parser(["k1"], "fast", session=FakeHttp(dead={"k1"})).parse(pdf)


def test_a_table_holding_a_figure_the_pdf_never_prints_becomes_a_note_and_a_real_table_stays():
    pdf = "Revenue $2,033,181 LTM 38.21% Total headcount: 157 India US UK"
    md = ("| Metric | LTM |\n| --- | --- |\n| Revenue | $2,033,181 |\n| Growth | 38.21% |\n\nTotal headcount: 157\n\n"
          "| Region | Count | Percentage |\n| ------ | ----- | ---------- |\n| India | 80 | 50.95% |\n| US | 57 | 36.30% |\n| UK | 20 | 12.74% |\n\nend")
    new, removed = ct.strip_unverified_tables(md, pdf)
    assert "| Revenue | $2,033,181 |" in new and "50.95" not in new and "Total headcount: 157" in new
    assert "India, US, UK" in new and "no values shown" in new
    assert len(removed) == 1 and removed[0]["labels"] == ["India", "US", "UK"] and "80" in removed[0]["figures_removed"]
    assert ct.compare_figures(pdf, new)["clean"]
