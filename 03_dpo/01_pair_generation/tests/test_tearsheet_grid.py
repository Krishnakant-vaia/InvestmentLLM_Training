"""Tear-sheet generator: the grid gives every sheet each compatible (type, defect) once; the sheets become one chunk each."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assign  # noqa: E402
import config  # noqa: E402
import generate  # noqa: E402
import prep_tearsheets  # noqa: E402
from score_pairs import COMPATIBLE  # noqa: E402

CH = [{"chunk_id": "A", "n_quantities": 50, "text": "x"}, {"chunk_id": "B", "n_quantities": 3, "text": "y"}]


def test_each_sheet_gets_18_pairs_one_per_call():
    calls = assign.assign_grid(CH)
    assert len(calls) == 36 and all(len(c["types"]) == 1 for c in calls)
    for cid in "AB":
        mine = [(c["types"][0], c["defects"][0]) for c in calls if c["chunk_id"] == cid]
        assert sorted(mine) == sorted((t, d) for t in config.TEARSHEET_TYPES for d in COMPATIBLE[t]) and len(mine) == 18
    assert len({c["call_id"] for c in calls}) == 36


def test_styles_and_determinism():
    calls = assign.assign_grid(CH)
    assert calls == assign.assign_grid(CH)
    assert all(c["style"] in config.TEARSHEET_STYLES for c in calls if c["types"] == ["Drafting"])
    assert {c["style"] for c in calls if c["types"] != ["Drafting"]} == {config.TEARSHEET_NEUTRAL_STYLE}


def test_sheets_become_one_chunk_each():
    rows = prep_tearsheets.sheet_chunks({"documents/tear sheets/01_A_Co_Tear_Sheet.md": "Revenue $1,000 up 5.5%", "x/02_B.md": "  "})
    assert len(rows) == 1 and rows[0]["folder"] == "tear sheets" and rows[0]["n_chunks"] == 1 and rows[0]["n_quantities"] == 2
    assert rows[0]["chunk_id"] == "01_A_Co_Tear_Sheet"


def test_the_tearsheet_prompt_loads_and_fills():
    t = generate.load_prompt(config.GENERATION_PROMPT_TEARSHEET)
    for ph in ("{chunk_text}", "{question_types}", "{defect_types}"):
        assert ph in t, ph


def test_table_quotes_pass_in_tear_sheets_only_and_invented_figures_fail():
    import score_pairs as sp
    doc = " ".join("| Our investment | Ownership |\r\n| --- | --- |\r\n| $897,000 | 6.28% |".split())
    assert sp.quote_in_document("Our investment | $897,000", doc, "tear sheets")
    assert not sp.quote_in_document("Our investment | $897,000", doc, "CIBIL Credit Scoring")
    assert not sp.quote_in_document("Our investment | $898,000", doc, "tear sheets")


def _cim_chunks():
    return [{"chunk_id": f"C{i}", "n_quantities": 40 if i % 2 == 0 else 2, "text": ("FY25 net debt " + ("[SECTION 1: x] [SECTION 4: y]" if i % 3 == 0 else "[SECTION 5: z]"))} for i in range(30)]


def test_balanced_cim_assignment_five_per_chunk_with_rules():
    chunks = _cim_chunks()
    calls = assign.assign_balanced(chunks, **config.CIM_ASSIGN)
    by = {}
    for c in calls:
        assert len(c["types"]) == 1
        by.setdefault(c["chunk_id"], []).append((c["types"][0], c["defects"][0]))
    assert all(len(v) == 5 and len(set(v)) == 5 for v in by.values())            # no (type, defect) twice in a chunk
    for ch in chunks:
        mine = by[ch["chunk_id"]]
        if ch["n_quantities"] < config.MIN_QUANTITIES_FOR_NUMERIC:
            assert all(t not in config.NUMERIC_TYPES_NEED_FIGURES for t, _ in mine)
        if ch["text"].count(config.CIM_SECTION_MARK) < 2:
            assert all(d != "cross_section_conflict" for _, d in mine)
    assert calls == assign.assign_balanced(chunks, **config.CIM_ASSIGN)
    assert len({c["call_id"] for c in calls}) == len(calls)
