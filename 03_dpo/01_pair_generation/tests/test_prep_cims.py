import re

import assign
import config
import prep_cims

FOOT = "Strictly Private and Confidential  Synthetic document  Page 1"


def body(word, n=700):
    return " ".join([word] * n)


def pages():
    return [
        'Book Number\nThis Memorandum has been prepared on behalf of Acme Pharma Limited (the "Company") solely to assist',
        "Table of Contents",
        "Table of Contents\n1 Executive Summary 4",
        f"Section 1\nExecutive Summary\n{FOOT}",
        f"Executive Summary\n{body('exec')}\n{FOOT}",
        f"Section 2\nInvestment Considerations\n{FOOT}",
        f"Investment Considerations\n {body('consider')}\n{FOOT}",
        f"Section 3\nBusiness History and Description\n{FOOT}",
        f"Section 3-0: Industry and Market Overview\nExhibit 22: Key risks and mitigants {body('risk')}\n{FOOT}",
        f"Section 3-A: US Generics\nSEGMENT PROFILE {body('seg')}\n{FOOT}",
        f"Section 4\nFinancial Review\n{FOOT}",
        f"Financial Review\nRevenue FY2025 {body('fin', 2300)}\n{FOOT}",
        f"Exhibit 40: margin {body('fin2', 2300)}\n{FOOT}",
        f"Section 5\nManagement\n{FOOT}",
        f"Senior Management\n{body('mgmt')}\n{FOOT}",
    ]


def test_each_section_is_a_labelled_chunk_and_the_footer_and_glyphs_are_gone():
    chunks = prep_cims.cim_chunks(pages(), "Acme_CIM")
    by = {c["chunk_id"]: c for c in chunks}
    assert {"Acme_exec_summary", "Acme_investment_considerations", "Acme_industry_and_risks", "Acme_segment_a",
            "Acme_management"} <= set(by)
    assert by["Acme_exec_summary"]["text"].startswith("CIM of Acme Pharma Limited")
    assert "[SECTION 1: Executive Summary]" in by["Acme_exec_summary"]["text"]
    assert "[SECTION 3-A: Business segment A: US Generics]" in by["Acme_segment_a"]["text"]
    assert all("Strictly Private" not in c["text"] and not re.search("[-]", c["text"]) for c in chunks)
    assert all(c["folder"] == config.CIM_FOLDER for c in chunks)
    assert "Key risks and mitigants" in by["Acme_industry_and_risks"]["text"]


def test_a_long_section_is_split_at_page_boundaries_and_nothing_is_dropped():
    chunks = prep_cims.cim_chunks(pages(), "Acme_CIM")
    fin = [c for c in chunks if c["chunk_id"].startswith("Acme_financial_review_p")]
    assert len(fin) == 2 and all("(part" in c["text"] for c in fin)
    assert "fin " in fin[0]["text"] and "fin2 " in fin[1]["text"]


def test_two_section_chunks_hold_two_labels_within_the_size_cap():
    chunks = prep_cims.cim_chunks(pages(), "Acme_CIM")
    both = [c for c in chunks if c["chunk_id"] in ("Acme_exec_plus_financials", "Acme_considerations_plus_financials")]
    assert len(both) == 2
    for c in both:
        assert c["text"].count(config.CIM_SECTION_MARK) == 2
        assert len(c["text"]) < 2 * prep_cims.PART_CHARS + 2000


def test_cross_section_conflict_only_lands_on_two_section_chunks():
    chunks = [{"chunk_id": f"c{i:03d}", "folder": "cims", "n_quantities": 20,
               "text": ("[SECTION 1: x] FY2025E CAGR\n[SECTION 4: y]" if i % 3 == 0 else "[SECTION 4: y] FY2025E CAGR segment")}
              for i in range(120)]
    by_id = {c["chunk_id"]: c for c in chunks}
    calls = assign.assign_balanced(chunks, **config.CIM_ASSIGN)
    used = [c for c in calls if "cross_section_conflict" in c["defects"]]
    assert used and all(by_id[c["chunk_id"]]["text"].count(config.CIM_SECTION_MARK) == 2 for c in used)


def test_basis_defects_are_assignable_to_cim_text_that_never_says_adjusted():
    chunks = [{"chunk_id": f"c{i:03d}", "folder": "cims", "n_quantities": 20,
               "text": "[SECTION 4: Financial Review] Revenue FY2025 and FY2026E, CAGR 12.5%, net debt, segment table"} for i in range(60)]
    calls = assign.assign_balanced(chunks, **config.CIM_ASSIGN)
    assert {d for c in calls for d in c["defects"]} & config.ADJUSTMENT_DEFECTS


def test_cim_judge_is_short_loads_and_expected_gates_use_only_its_gates():
    import judge
    import score_pairs
    text = judge.load_template(config.JUDGE_PROMPT_CIM)
    assert len(text.strip().splitlines()) <= 50
    cim_defects = set(config.CIM_DEFECT_TARGETS)
    assert cim_defects == set(score_pairs.EXPECTED_GATES_CIM)
    assert all(g in {"G1", "G2", "G3", "G4", "G5", "G6", "G7"} for gs in score_pairs.EXPECTED_GATES_CIM.values() for g in gs)


def test_not_preferred_failing_a_gate_of_its_defect_is_clean_on_a_cim_row_and_flagged_on_a_default_row():
    import score_pairs
    good, bad = {"score": 9, "reason": "PASS_ALL_GATES", "failed_gates": []}, {"score": 2, "reason": "F06_NUMERICAL_ERROR", "failed_gates": ["G3", "G8"]}
    cim = {"defect_type": "calculation_slip", "folder": config.CIM_FOLDER}
    assert score_pairs.judge_flags(cim, good, bad) == []
    other = {"defect_type": "wrong_definition", "folder": config.CIM_FOLDER}
    assert score_pairs.judge_flags(other, good, {"score": 3, "reason": "F08_PERIOD_SCOPE_ERROR", "failed_gates": ["G3", "G8"]}) == []
    assert score_pairs.judge_flags({"defect_type": "wrong_definition", "folder": "x"}, good,
                                   {"score": 3, "reason": "F08_PERIOD_SCOPE_ERROR", "failed_gates": ["G3", "G8"]})


def test_facts_and_markdown_round_trip():
    data = {"revenue": [716.33, 765.89], "ebitda": [100.5], "net_income": [], "segments": {"A": [10.25, 11.5]}, "net_debt_fy25": 141.26,
            "leverage_fy25": 1.2348, "cagr_rev_hist": 0.0710, "cagr_rev_proj": 0.0785}
    f = prep_cims.facts_of(data)
    assert 716.33 in f and 10.25 in f and any(abs(x - 7.10) < 1e-9 for x in f) and 1.2348 in f
    pages = ["Section 1: Executive Summary\nbody one", "page two text", "Strictly Private and Confidential - Page 3"]
    assert prep_cims.pages_of_md(prep_cims.md_of(pages)) == ["Section 1: Executive Summary\nbody one", "page two text"]


def test_figure_check_flags_an_input_found_nowhere():
    import score_pairs
    row = {"facts": [716.34, 765.89], "document": "Revenue FY2022 $716.3 million; EBITDA 120.5",
           "numeric_check": {"expression": "(765.89 - 716.34) / 716.34", "final_value": 0.0692}}
    assert score_pairs.figure_flags(row) == []
    row["numeric_check"] = {"expression": "(777.12 - 716.34) / 716.34"}
    assert score_pairs.figure_flags(row) and score_pairs.figure_flags({**row, "facts": []}) == []


def test_final_value_matches_ratio_for_percent_and_rounding():
    import score_pairs
    assert score_pairs.final_matches(0.195, 19.5, "%") and score_pairs.final_matches(1.235, 1.2, "x")
    assert not score_pairs.final_matches(0.195, 19.5, "x") and not score_pairs.final_matches(1.5, 1.2, "x")
