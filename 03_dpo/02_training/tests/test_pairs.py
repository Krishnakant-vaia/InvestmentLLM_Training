"""The data layer: source checks, grouping, <think> format, length guard, and the leak-free split."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pairs import shape, source  # noqa: E402
from sft_common import dpo_data  # noqa: E402


def row(i, origin="numerical-1", chunk="DK-1.1-0001", q=None, chosen="step\nAnswer: 5", rejected="Answer: 6", src="Accounting"):
    q = q or f"[Deccan Dairy, file ref {i}] Compute LCR: HQLA ₹{8000 + i} crore, outflows ₹{5000 + i} crore."
    return {"prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": q}], "preferred": chosen, "not_preferred": rejected,
            "source": src, "origin": origin, "pair_id": f"{origin}:{chunk}:{i}", "chunk_id": chunk, "call_id": chunk, "line": i}


def test_source_refuses_a_file_that_the_manifest_does_not_describe():
    rows = [row(i) for i in range(3)]
    manifest = {"total_pairs": 3, "batches": [{"origin": "numerical-1", "pairs": 3}]}
    assert source.check_against_manifest(rows, manifest) == []
    assert any("manifest says 5" in p for p in source.check_against_manifest(rows, {**manifest, "total_pairs": 5}))
    assert any("repeated pair_id" in p for p in source.check_against_manifest(rows + [rows[0]], None))
    assert any("missing" in p for p in source.check_against_manifest([{**rows[0], "preferred": ""}], None))
    assert any("per origin" in p for p in source.check_against_manifest(rows, {**manifest, "batches": [{"origin": "numerical-2", "pairs": 3}]}))


def test_domain_comes_from_the_sheet_prefix_and_documents_keep_their_label():
    assert shape.domain_of(row(1, chunk="WU-4.1-0001")) == "workflow_understanding"
    assert shape.domain_of(row(1, chunk="AR-6.2-0003")) == "agentic_reasoning"
    assert shape.domain_of(row(1, origin="full-1", chunk="0e3a1745f0ab", src="credit_rating")) == "credit_rating"


def test_questions_from_one_template_share_a_group_and_other_templates_do_not():
    a, b = row(1), row(2, q="[Orion Logistics, file ref 9] Compute LCR: HQLA ₹1 crore, outflows ₹2 crore.")
    other = row(3, q="[Vertex Auto, file ref 5] Compute DSCR: EBITDA ₹10 crore, debt service ₹4 crore.")
    ga, gb, go = (shape.group_of(r, "template") for r in (a, b, other))
    assert ga == gb and ga != go
    assert shape.group_of(a, "prompt") != shape.group_of(b, "prompt")                 # the old, leaky key
    d1, d2 = row(1, origin="full-1", chunk="c1"), row(2, origin="full-1", chunk="c1", q="a different question about the same document")
    assert shape.group_of(d1, "template") == shape.group_of(d2, "template")          # one document = one group


def test_workings_move_into_think_on_both_sides_or_not_at_all():
    c, r = shape.workings_in_think("1+1=2\nAnswer: 2", "1+1=3\nAnswer: 3")
    assert c == "<think>\n1+1=2\n</think>\n\nAnswer: 2" and r == "<think>\n1+1=3\n</think>\n\nAnswer: 3"
    assert shape.workings_in_think("1+1=2\nAnswer: 2", "just 3") == ("1+1=2\nAnswer: 2", "just 3")           # one side cannot: neither changes
    already = "<think>x</think>\n\nAnswer: 2"
    assert shape.workings_in_think(already, "t\nAnswer: 3") == (already, "t\nAnswer: 3")


def test_length_guard_drops_the_widest_gaps_first_and_only_when_asked():
    rows = [row(i, chosen="word " * (10 + i), rejected="word " * 5) for i in range(6)] + [row(10 + i, chosen="w", rejected="word " * 5) for i in range(4)]
    kept, n = shape.cap_preferred_longer(rows, 0.5)
    gone = {rows[5]["pair_id"], rows[4]["pair_id"]}                                  # 6/10 and 5/9 are above 50%; 4/8 is not
    assert n == 2 and len(kept) == 8 and not gone & {r["pair_id"] for r in kept}
    out, report = shape.shape(rows, {"split_by": "template"})
    assert len(out) == 10 and "length guard" not in "\n".join(report)
    out, report = shape.shape(rows, {"max_preferred_longer_share": 0.5})
    assert sum(len(r["preferred"].split()) > len(r["not_preferred"].split()) for r in out) / len(out) <= 0.5


def test_validation_gets_whole_template_groups_and_no_group_is_on_both_sides():
    class Tok:                                                                     # minimal stand-in: ids = characters
        pass
    pairs = []
    for t in range(40):                                                            # 40 templates x 5 pairs
        for k in range(5):
            pairs.append({"prompt_ids": [t, k], "chosen_ids": [1], "rejected_ids": [2], "source": "s", "line": len(pairs), "group": f"tpl{t}"})
    train, val = dpo_data._split_by_prompt(pairs, 0.1, 500, 0)
    assert val and {p["group"] for p in train}.isdisjoint({p["group"] for p in val})
    assert len(val) % 5 == 0 and len(train) + len(val) == 200


def test_remarks_are_removed_from_both_sides_the_same_way():
    from pairs.clean import clean_answer
    assert clean_answer("Total = 5 (cure period mis-converted).\nAnswer: 5 (inverted).") == "Total = 5.\nAnswer: 5."
    assert clean_answer("Due (Tue); see (6) and (3,4,5); pool ignored.") == "Due (Tue); see (6) and (3,4,5)"
    assert clean_answer("Given: a (b c d) stays\nx (two words) = 1") == "Given: a (b c d) stays\nx = 1"
    both = [clean_answer(t) for t in ("Net = 9 (disallowed allowance release).", "Net = 9 (allowance ignored).")]
    assert both[0] == both[1] == "Net = 9."                                                  # one rule for preferred and not-preferred


def test_sheet_8_pairs_form_one_domain():
    assert shape.domain_of(row(1, chunk="EA-8.3-0010")) == "enterprise_ai_applications"
