"""Hard-negative rewrite: parsing, the code checks, and the accept decision."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import rewrite_run as rw  # noqa: E402

REC = {"kind": "numerical", "preferred": "CAGR = (593/358)^(1/3)-1 = 18.3%.\nDraft: x\nAnswer: 18.3%.", "original": "Revenue grew 21.9%.\nAnswer: 21.9%.",
       "question": "q", "domain": "d", "id": 0}
GOOD = "CAGR = ((593-358)/358)/3 = 21.9%.\nDraft: y\nAnswer: 21.9%."


def test_parse_numerical_and_document_shapes():
    assert rw.parse_rewrite('```json\n{"answer": " x "}\n```', "numerical") == "x"
    assert rw.parse_rewrite('{"think": "t", "answer": "a"}', "document") == "<think>t</think>\n\na"
    assert rw.parse_rewrite('{"answer": "a"}', "document") is None and rw.parse_rewrite("not json", "numerical") is None


def test_checks_catch_admissions_length_structure_and_copying():
    c = rw.code_checks(REC, GOOD)
    assert all(c[k] for k in ("no_admission", "length_ok", "same_lines", "different_final"))
    assert not rw.code_checks(REC, GOOD.replace("21.9% .", "x") + " (inverted)")["no_admission"]
    assert not rw.code_checks(REC, "Answer: 21.9%.")["length_ok"]
    assert not rw.code_checks(REC, "a\nb\nc\nd\ne\nAnswer: 21.9% x y z w")["same_lines"]
    assert not rw.code_checks(REC, REC["preferred"])["different_final"]


def test_accept_only_when_every_check_and_the_judge_agree():
    ok = {"score": 9}
    bad = {"score": 2}
    r = {**REC, "rewrite": GOOD, "checks": rw.code_checks(REC, GOOD), "judge_preferred": ok, "judge_rewrite": bad}
    rw.decide(r)
    assert r["accepted"] and r["rejected_because"] == []
    r.update(judge_rewrite=ok)
    rw.decide(r)
    assert not r["accepted"] and "judge passes the rewrite" in r["rejected_because"][0]
    r.update(rewrite=None, checks=None)
    rw.decide(r)
    assert r["rejected_because"] == ["no valid rewrite"]


def test_target_length_and_feedback_tell_the_rewriter_what_to_fix():
    assert "between" in rw.target_length(REC)
    doc = {**REC, "kind": "document", "preferred": "<think>one two three four five six seven eight nine ten</think>\n\nab cd ef gh"}
    t = rw.target_length(doc)
    assert "reasoning should be about 10 words" in t and "final answer about 4 words" in t
    fb = rw.feedback_for(REC, "Answer: 18.3% (inverted).", rw.code_checks(REC, "Answer: 18.3% (inverted)."))
    assert "inverted" in fb and "length" in fb and "number of lines" in fb
    same = rw.feedback_for(REC, "Answer: 18.3%.", rw.code_checks(REC, "Answer: 18.3%."))
    assert "same as the preferred" in same
    assert rw.feedback_for(REC, GOOD, rw.code_checks(REC, GOOD)) == ""
    prompt = rw.build_prompt(open(rw.PROMPT, encoding="utf-8").read(), REC, "too short")
    assert "{{" not in prompt and "PREVIOUS ATTEMPT WAS REJECTED: too short" in prompt


def test_a_rewrite_that_reveals_the_mistake_is_not_accepted():
    r = {**REC, "rewrite": GOOD, "checks": rw.code_checks(REC, GOOD), "judge_preferred": {"score": 9}, "judge_rewrite": {"score": 2}, "reveals_mistake": True}
    rw.decide(r)
    assert not r["accepted"] and any("reveals" in w for w in r["rejected_because"])


def test_the_rewriter_is_retried_with_feedback_until_the_checks_pass():
    class Fake:
        def __init__(self):
            self.prompts = []

        def chat(self, prompt, **kw):
            self.prompts.append(prompt)
            reply = {"answer": "Answer: 18.3%."} if len(self.prompts) == 1 else {"answer": GOOD}                # first attempt copies the preferred result
            return {"content": __import__("json").dumps(reply)}
    llm = Fake()
    out = rw.rewrite_all([REC], llm, 1)[0]
    assert out["attempts"] == 2 and out["rewrite"] == GOOD and "PREVIOUS ATTEMPT WAS REJECTED" in llm.prompts[1] and "PREVIOUS" not in llm.prompts[0]
