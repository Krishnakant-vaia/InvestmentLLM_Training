"""The training-set wrapper: selection labels, the file format, and adding batches safely."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import make_training_set as mts  # noqa: E402
from test_checkpoints import FakeS3  # noqa: E402

GOOD = {"score": 9, "reason": "PASS_ALL_GATES", "failed_gates": []}
BAD = {"score": 2, "reason": "F04_UNSUPPORTED_CLAIM", "failed_gates": ["G2", "G8"]}


def row(i=0, chosen=GOOD, rejected=BAD, defect="unsupported_claim", flags=(), qtype="Terminology"):
    return {"document": "DOC", "question": f"Q{i}?", "question_type": qtype, "defect_type": defect, "folder": "Credit Rating Replication",
            "preferred_thinking": "<think>a</think>", "preferred": "preferred answer text", "not_preferred_thinking": "<think>b</think>",
            "not_preferred": "not preferred answer text", "chunk_id": f"c{i}", "call_id": f"c{i}-p0-0", "judge_preferred": chosen,
            "judge_not_preferred": rejected, "flags": list(flags), "passes_all_gates": not flags}


def test_labels():
    assert mts.label(row(), 1.0)[0] == "SEND"
    assert mts.label(row(chosen={**GOOD, "score": 8}), 1.0)[0] == "SEND"                        # 8 is enough
    assert mts.label(row(chosen={**BAD}), 1.0)[0] == "EXCLUDE"                                    # preferred answer fails the judge
    assert mts.label(row(flags=["not-preferred answer has a tell"]), 1.0)[0] == "EXCLUDE"
    assert mts.label(row(flags=["numeric: not-preferred equals preferred"]), 1.0)[0] == "EXCLUDE"
    assert mts.label(row(rejected=GOOD), 1.0)[0] == "HOLD"                                        # the mistake was not caught
    assert mts.label(row(rejected=GOOD, defect="padded"), 1.0)[0] == "SEND"                       # padded counter-pairs are meant to pass
    assert mts.label(row(), 2.5)[0] == "HOLD" and mts.label(row(), 0.4)[0] == "HOLD"              # beyond the trainer's 2x rule
    assert mts.label({**row(), "judge_preferred": None}, 1.0) == ("EXCLUDE", ["no judge verdict"])
    assert mts.label(row(flags=["length outside the defect's rule"]), 1.0)[0] == "SEND"           # informational flags do not block


def test_a_batch_is_added_once_in_trainer_format_with_provenance_and_a_manifest():
    s3 = FakeS3()
    rows = [row(0), row(1, rejected=GOOD), row(2, chosen=BAD), row(3, rejected=GOOD, defect="padded")]
    entry = mts.add_batch(s3, "run-a", rows, lambda t: len(t.split()), "SYSTEM")
    assert entry["pairs"] == 2 and entry["labels"] == {"SEND": 2, "HOLD": 1, "EXCLUDE": 1} and entry["batch"] == 1
    lines = [json.loads(l) for l in s3.objects[mts.TRAIN_KEY].decode().split("\n") if l]
    assert [p["pair_id"] for p in lines] == ["run-a:c0-p0-0:Terminology", "run-a:c3-p0-0:Terminology"]
    p = lines[0]
    assert [m["role"] for m in p["prompt"]] == ["system", "user"] and p["preferred"].startswith("<think>") and p["origin"] == "run-a"
    assert p["preferred_score"] == 9 and p["not_preferred_score"] == 2 and p["source"] == "credit_rating"
    manifest = json.loads(s3.objects[mts.MANIFEST_KEY])
    assert manifest["total_pairs"] == 2 and manifest["batches"][0]["origin"] == "run-a" and manifest["rule"]["length_cap_on_preferred_longer_share"] is None
    with pytest.raises(SystemExit, match="already in the training set"):
        mts.add_batch(s3, "run-a", rows, lambda t: len(t.split()), "SYSTEM")                      # never twice
    mts.add_batch(s3, "run-b", [row(7)], lambda t: len(t.split()), "SYSTEM")                      # a second source appends
    assert len([l for l in s3.objects[mts.TRAIN_KEY].decode().split("\n") if l]) == 3
    assert json.loads(s3.objects[mts.MANIFEST_KEY])["total_pairs"] == 3


def test_dry_run_writes_nothing():
    s3 = FakeS3()
    entry = mts.add_batch(s3, "run-a", [row(0)], lambda t: len(t.split()), "SYSTEM", dry_run=True)
    assert entry["pairs"] == 1 and s3.objects == {}


def test_numerical_rows_become_pairs_without_document_or_think():
    s3 = FakeS3()
    judged = {"pair_id": "FR-0001", "capability": "DCF", "question": "What is 2+2?", "question_type": "Numerical", "preferred": "4",
              "not_preferred": "5", "judge_preferred": GOOD, "judge_not_preferred": BAD, "flags": []}
    s3.objects[f"{mts.config.OUT_ROOT}numerical-x/sheet/dcf/final/judged_pairs.jsonl"] = (json.dumps(judged) + "\n").encode()
    rows = mts.numerical_rows(s3, "numerical-x")
    entry = mts.add_batch(s3, "numerical-x", rows, lambda t: len(t.split()), "SYSTEM")
    assert entry["pairs"] == 1
    p = json.loads(s3.objects[mts.TRAIN_KEY].decode().strip())
    assert p["prompt"][1]["content"] == "What is 2+2?" and p["preferred"] == "4" and p["not_preferred"] == "5" and p["source"] == "DCF"
