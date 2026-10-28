"""Eval bank: loading the workbook, the prompt barrier, the train-overlap check, the eval job."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft_common import eval_bank as eb  # noqa: E402

COLS = ["QA_ID", "Sub-Capability", "Question Type", "Context", "Question", "Options", "Gold Answer",
        "Gold Answer (Medium Length)", "Answer Format", "Tolerance", "Difficulty", "Scoring Rubric", "Source/Basis"]


def row(qid, q, ctx="", opts="", gold="SECRET-GOLD", rubric="SECRET-RUBRIC"):
    return dict(zip(COLS, [qid, "Sub", "MCQ", ctx, q, opts, gold, "", "MCQ-Letter", "Exact", "Easy", rubric, "src"]))


@pytest.fixture
def workbook(tmp_path):
    p = tmp_path / "bank.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"How to use": ["x"]}).to_excel(w, sheet_name="README", index=False)
        pd.DataFrame([row("DK-001", "Which inventory method does IFRS prohibit?", opts="A) FIFO | B) LIFO"),
                      row("DK-002", "What is WACC?")], columns=COLS).to_excel(w, sheet_name="1. Domain Knowledge", index=False)
        pd.DataFrame([row("EK-001", "What was the EBITDA margin?", ctx="Revenue: 200. EBITDA: 36.")],
                     columns=COLS).to_excel(w, sheet_name="2. Enterprise Knowledge", index=False)
        pd.DataFrame([{"QA_ID": "DK-001", "Model Response": "x"}]).to_excel(w, sheet_name="Results Log", index=False)
    return p


def test_loads_only_category_sheets(workbook):
    rows = eb.load_bank(workbook)
    assert [r["QA_ID"] for r in rows] == ["DK-001", "DK-002", "EK-001"]
    assert rows[2]["Category"] == "2. Enterprise Knowledge" and rows[1]["Context"] == ""


def test_prompt_never_contains_gold_answers_or_rubrics(workbook):
    for r in eb.load_bank(workbook):
        p = eb.question_prompt(r)
        assert r["Question"] in p and "SECRET" not in p
    p = eb.question_prompt(eb.load_bank(workbook)[0])
    assert "Options: A) FIFO | B) LIFO" in p and p.endswith("Answer format: MCQ-Letter")


def test_barrier_finds_training_material_and_passes_a_clean_machine(tmp_path):
    clean = tmp_path / "clean"
    (clean / "outputs" / "eval_bank").mkdir(parents=True)
    (clean / "models" / "m").mkdir(parents=True)
    (clean / "models" / "m" / "model.safetensors").write_bytes(b"w")          # the merged model is fine
    (clean / "outputs" / "eval_bank" / "eval_bank_responses.jsonl").write_text("{}")
    assert eb.barrier_violations(clean) == []
    dirty = tmp_path / "dirty"
    (dirty / "data" / "processed").mkdir(parents=True)
    (dirty / "data" / "processed" / "train.jsonl").write_text("{}")
    (dirty / "outputs" / "sft" / "checkpoint-200").mkdir(parents=True)
    assert eb.barrier_violations(dirty) == ["data/processed/train.jsonl", "outputs/sft/checkpoint-200"] or \
        eb.barrier_violations(dirty) == ["data\\processed\\train.jsonl", "outputs\\sft\\checkpoint-200"]


def test_train_overlap_flags_copied_questions_only(workbook, tmp_path):
    rows = eb.load_bank(workbook)
    long_q = "Under US GAAP which inventory costing method is permitted that IFRS and Ind AS both prohibit today"
    rows[0]["Question"] = long_q
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps({"text": t}) for t in [
        "<|im_start|>user\n" + long_q.upper() + "?<|im_end|>",      # copied (case/punctuation differ)
        "something unrelated about bonds and yields",
    ]) + "\n", encoding="utf-8")
    rep = eb.train_overlap(rows, train, workers=2, batch=1)
    assert rep["DK-001"]["flagged"] and rep["DK-001"]["fraction"] == 1.0
    assert not rep["EK-001"]["flagged"] and not rep["DK-002"]["flagged"]
