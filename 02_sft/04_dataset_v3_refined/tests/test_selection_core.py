import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selection_core as core


def lab(standalone=True, quality=5, depth=2, reason="OK"):
    return {"standalone": standalone, "quality": quality, "depth": depth, "reason": reason}


def row(q, folder="DK_ACCOUNTING_STANDARDS", qt="High-Order Reasoning", source="s/a.md"):
    return {"question": q, "_source": source, "_chunk_id": "c", "_question_type": qt,
            "_difficulty": "Advanced", "capability_folder": folder, "capability": folder.split("_")[0]}


def big(rows, folder="DK_BIG", n=2000):
    """Pad a folder past the small-folder threshold with unlabelled rows."""
    return rows + [row("pad", folder=folder)] * n


def test_extract_row_takes_text_after_document_and_folder_from_path():
    rec = {"messages": [{"role": "system", "content": "x"},
                        {"role": "user", "content": "<document>body</document>\n\nWhat is CRAR?"}],
           "_source": "Corpus/Invest_dataset/run2/DK_REGULATORY_CAPITAL_FLUENCY - desc/text/doc.md",
           "_question_type": "Multi‑Concept Relationship", "_difficulty": "advanced"}
    r = core.extract_row(rec)
    assert r["question"] == "What is CRAR?"
    assert r["capability_folder"] == "DK_REGULATORY_CAPITAL_FLUENCY"
    assert r["capability"] == "DK"
    assert r["_question_type"] == "Multi-Concept Relationship"
    assert r["_difficulty"] == "Advanced"


def test_normalize_question_ignores_case_and_punctuation():
    assert core.normalize_question("What is EBITDA?") == core.normalize_question("what is  EBITDA")


def test_render_prompt_fills_single_question():
    assert core.render_prompt("Q: {{question}}", "What is LTV?") == "Q: What is LTV?"


def test_parse_label_accepts_valid_reply_with_surrounding_text():
    raw = 'ok {"standalone": false, "quality": 3, "depth": 1, "reason": "DOC_ONLY_SUBJECT"} done'
    assert core.parse_label(raw) == lab(standalone=False, quality=3, depth=1, reason="DOC_ONLY_SUBJECT")


@pytest.mark.parametrize("bad", [
    lab(quality=6), lab(depth=0), lab(reason="NOPE"), lab() | {"standalone": "yes"},
])
def test_parse_label_rejects_invalid_fields(bad):
    with pytest.raises(ValueError):
        core.parse_label(json.dumps(bad))


def test_parse_label_rejects_non_json():
    with pytest.raises(ValueError):
        core.parse_label("I think this question is fine.")


def test_dedup_batches_only_passing_rows_grouped_by_source():
    rows = big([row("a1", folder="DK_BIG", source="a"), row("a2", folder="DK_BIG", source="a"),
                row("a3", folder="DK_BIG", source="a"), row("b1", folder="DK_BIG", source="b")])
    labels = [lab(), lab(quality=3), lab(), lab()] + [None] * 2000
    batches = core.dedup_batches(rows, labels)
    # a2 fails the quality floor in a large folder; b has only one passing question
    assert [idxs for _, idxs in batches] == [[0, 2]]


def test_parse_dedup_maps_and_validates():
    raw = json.dumps({"results": [{"id": 0, "dup_of": None}, {"id": 1, "dup_of": 0}]})
    assert core.parse_dedup(raw, 2) == [None, 0]
    with pytest.raises(ValueError):
        core.parse_dedup(json.dumps({"results": [{"id": 0, "dup_of": 0}]}), 1)
    with pytest.raises(ValueError):
        core.parse_dedup(json.dumps({"results": [{"id": 0, "dup_of": None}]}), 2)


def test_select_filters_ranks_and_cuts():
    rows = big([row(f"question number {i}") for i in range(6)], folder="DK_ACCOUNTING_STANDARDS")
    labels = [lab(quality=4, depth=1), lab(quality=5, depth=1), lab(quality=5, depth=3),
              lab(quality=3), lab(standalone=False), lab()] + [None] * 2000
    kept, stats = core.select(rows, labels, duplicates={5}, target=2, template_cap=1.0)
    assert [r["question"] for r in kept] == ["question number 2", "question number 1"]
    assert stats["funnel"]["duplicate"] == 1
    assert stats["funnel"]["unlabelled"] == 2000
    assert stats["funnel"]["cut_by_target"] == 1


def test_small_folder_keeps_quality_3_and_beats_target():
    rows = big([row("a b c d e one", folder="DK_BIG"), row("rare question", folder="EA_TINY")])
    labels = [lab(quality=5), lab(quality=3)] + [None] * 2000
    kept, _ = core.select(rows, labels, target=1, template_cap=1.0)
    assert [r["question"] for r in kept] == ["rare question"]


def test_template_cap_limits_shared_openings_per_type():
    rows = [row(f"form a thesis on topic {i}", folder="EA_TINY") for i in range(10)]
    rows += [row(f"unique opening {i} here now", folder="EA_TINY") for i in range(10)]
    kept, stats = core.select(rows, [lab()] * 20, target=100, template_cap=0.15)
    assert len([r for r in kept if r["question"].startswith("form a thesis")]) == 3  # floor(0.15 * 20)
    assert stats["funnel"]["template_capped"] == 7
