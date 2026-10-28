"""Tests for the content-generation prompt and the code that supports it (style line, evidence quotes, adjustment defects).

    cd DPO_Training/QA_pairs_prep && python -m pytest tests/test_content_prompt.py -q
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import assign  # noqa: E402
import config  # noqa: E402
import generate  # noqa: E402
import score_pairs  # noqa: E402

PROMPT = HERE / "prompts" / "DPO_content_generation_prompts.txt"
DOC = "DOC-1 QUALITY OF EARNINGS. Reported EBITDA is 92.4. The add-back of 2.3 is accepted. Underwritten EBITDA is 94.7."


def test_prompt_loads_and_formats_with_every_placeholder():
    template = generate.load_prompt(PROMPT)
    text = template.format(chunk_text=DOC, question_types="Drafting", defect_types="headline_number", style=config.STYLES[2])
    assert DOC in text and config.STYLES[2] in text
    assert "{{" not in text and "}}" not in text and "{chunk_text}" not in text and "{style}" not in text   # every placeholder was filled
    assert "Answer format" not in text and "eval" not in text.lower()


def test_prompt_describes_every_defect_the_three_types_can_be_assigned():
    template = generate.load_prompt(PROMPT)
    defects = set()
    for t in ("Drafting", "Adjusted Metric", "Short Answer"):
        defects |= set(score_pairs.COMPATIBLE[t])
    for d in defects:
        assert f"- {d} (breaks" in template and f"- {d}:" in template, d


@pytest.mark.parametrize("name", ["DPO_content_generation_prompts.txt", "DPO_content_generation_prompts_sng.txt"])
def test_each_content_prompt_formats_with_the_pipeline_keywords(name):
    """generate.run_call fills the prompt with str.format(chunk_text, question_types, defect_types, style); a single brace in the
    JSON example or an empty {} makes every call fail, so both prompt files must format and describe every assignable defect."""
    template = generate.load_prompt(HERE / "prompts" / name)
    text = template.format(chunk_text=DOC, question_types="Drafting", defect_types="headline_number", style=config.STYLES[0])
    assert DOC in text and config.STYLES[0] in text and "{{" not in text and "}}" not in text
    for t in ("Drafting", "Adjusted Metric", "Short Answer"):
        for d in score_pairs.COMPATIBLE[t]:
            assert d in template, (name, d)


def test_drafting_allows_the_figure_defects():
    for d in ("headline_number", "wrong_adjustment", "calculation_slip", "wrong_definition"):
        assert d in score_pairs.COMPATIBLE["Drafting"]


def content_chunks(n):
    return [{"chunk_id": f"c{i:03d}", "folder": "f", "n_quantities": 12,
             "text": "Reported EBITDA 92.4 with one add-back of 2.3." if i % 2 == 0 else "A plain description of the business and its customers."}
            for i in range(n)]


def test_assign_gives_every_call_a_style_and_keeps_adjustment_defects_off_files_without_adjustments(monkeypatch):
    monkeypatch.setattr(config, "DEFECT_TARGETS", {d: 0.08 for d in (
        "omitted_element", "advice_not_draft", "generic_content", "unsupported_claim", "overstated_certainty", "headline_number",
        "wrong_adjustment", "calculation_slip", "wrong_definition", "no_workings", "off_by_one", "padded")})
    chunks = content_chunks(60)
    calls = assign.assign(chunks, passes=1, per_pass=3, type_mix={"Drafting": 1, "Adjusted Metric": 1, "Short Answer": 1})
    by_id = {c["chunk_id"]: c for c in chunks}
    assert calls and all(c["style"] in config.STYLES for c in calls)
    for c in calls:
        if not config.ADJUSTMENT_WORDS.search(by_id[c["chunk_id"]]["text"]):
            assert not set(c["defects"]) & config.ADJUSTMENT_DEFECTS
    assert {d for c in calls for d in c["defects"]} & config.ADJUSTMENT_DEFECTS      # and they are used where they can apply


def test_style_stream_does_not_change_which_types_and_defects_are_assigned():
    chunks = [{"chunk_id": f"c{i:03d}", "folder": "f", "n_quantities": 12, "text": "x"} for i in range(60)]
    a = assign.assign(chunks)
    strip = lambda calls: [(c["call_id"], c["types"], c["defects"]) for c in calls]    # noqa: E731
    assert strip(a) == strip(assign.assign(chunks))


def pair_row(evidence=None, document=DOC):
    row = {"document": document, "question": "Q?", "question_type": "Short Answer", "defect_type": "off_by_one",
           "preferred": "word " * 20, "not_preferred": "word " * 20 + "extra",
           "preferred_thinking": "<think>" + "word " * 40 + "</think>", "not_preferred_thinking": "<think>" + "word " * 40 + "</think>",
           "numeric_check": {}}
    if evidence is not None:
        row["evidence"] = evidence
    return row


def test_evidence_quotes_must_be_in_the_document():
    assert not [f for f in score_pairs.code_flags(pair_row(["Reported EBITDA is 92.4.", "The add-back of 2.3\nis accepted."])) if "evidence" in f]
    assert "evidence quote not found in the document" in score_pairs.code_flags(pair_row(["Underwritten EBITDA is 96.7."]))
    assert "evidence missing" in score_pairs.code_flags(pair_row([]))
    assert not [f for f in score_pairs.code_flags(pair_row()) if "evidence" in f]      # no key (web-corpus prompt): not checked


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat(self, user, **_):
        self.prompts.append(user)
        return {"content": self.reply, "finish_reason": "stop", "usage": {}, "message_keys": ["content"]}


def test_generate_passes_the_style_and_keeps_the_evidence_key():
    obj = {"question_type": "Short Answer", "defect_type": "off_by_one", "question": "How many?", "preferred_thinking": "a", "preferred": "9 of 12.",
           "not_preferred_thinking": "b", "not_preferred": "10 of 12.", "numeric_check": {}, "evidence": ["Reported EBITDA is 92.4."], "extra": 1}
    llm = FakeLLM(json.dumps([obj]))
    call = {"call_id": "c-p0-0", "chunk_id": "c", "types": ["Short Answer"], "defects": ["off_by_one"], "style": config.STYLES[4]}
    chunk = {"chunk_id": "c", "text": DOC, "source_key": "k", "folder": "f"}
    out = generate.run_call(call, chunk, llm, generate.load_prompt(PROMPT))
    assert config.STYLES[4] in llm.prompts[0]
    pair = out["pairs"][0]
    assert pair["evidence"] == ["Reported EBITDA is 92.4."] and "extra" not in pair and pair["preferred_thinking"].startswith("<think>")


def test_the_web_corpus_prompt_still_formats_with_a_style_keyword():
    template = generate.load_prompt()                      # qa_prompt.txt has no {style}
    template.format(chunk_text="x", question_types="Terminology", defect_types="generic_content", style="ignored")
