"""One vocabulary: the renames, the flag texts, and the guarantee that a migration is reversible."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import migrate_names as mn  # noqa: E402
from schema import normalize_row, rename_flag, to_old_names  # noqa: E402

SCORED = {"call_id": "c", "chosen": "A", "chosen_thinking": "<think>a</think>", "rejected": "B", "rejected_thinking": "<think>b</think>",
          "judge_chosen": {"score": 9}, "judge_rejected": {"score": 2},
          "flags": ["chosen identical to rejected", "judge: chosen answer fails (F06; gates ['G3'])", "length outside the defect's rule"]}
TRAINING = {"prompt": [], "chosen": "A", "rejected": "B", "chosen_judge": {"score": 9}, "rejected_judge": {"score": 2}, "preferred_score": 9, "not_preferred_score": 2}
NUMERICAL = {"pair_id": "DK-1", "accepted": "A", "rejected": "B", "judge_preferred": {"score": 9}, "judge_not_preferred": {"score": 2}, "flags": []}


def test_every_old_vocabulary_becomes_preferred_and_not_preferred_and_keeps_key_order():
    s = normalize_row(SCORED)
    assert list(s) == ["call_id", "preferred", "preferred_thinking", "not_preferred", "not_preferred_thinking", "judge_preferred", "judge_not_preferred", "flags"]
    assert s["flags"] == ["preferred identical to not-preferred", "judge: preferred answer fails (F06; gates ['G3'])", "length outside the defect's rule"]
    t = normalize_row(TRAINING)
    assert t["preferred"] == "A" and t["not_preferred"] == "B" and t["judge_preferred"] == {"score": 9} and "chosen_judge" not in t
    n = normalize_row(NUMERICAL, accepted_is_answer=True)
    assert n["preferred"] == "A" and n["not_preferred"] == "B" and "accepted" not in n


def test_accepted_is_renamed_only_where_it_is_an_answer():
    result = {"accepted": True, "rejected_because": [], "preferred": "A"}                  # a rewrite result: accepted is a yes/no
    assert normalize_row(result) == result
    assert normalize_row(result, accepted_is_answer=False)["accepted"] is True


def test_renaming_twice_changes_nothing_and_a_clash_is_refused():
    once = normalize_row(SCORED)
    assert normalize_row(once) == once
    with pytest.raises(ValueError, match="both"):
        normalize_row({"chosen": "A", "preferred": "A2"})


def test_flag_texts_keep_their_tails_and_unrelated_flags_are_untouched():
    assert rename_flag("numeric: rejected equals chosen") == "numeric: not-preferred equals preferred"
    assert rename_flag("judge: not-preferred answer passes (defect did not land)") == "judge: not-preferred answer passes (defect did not land)"


@pytest.mark.parametrize("kind,row", [("scored", SCORED), ("training", TRAINING), ("numerical", NUMERICAL)])
def test_a_migrated_file_reverses_to_the_original_rows(tmp_path, kind, row):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out" / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in (row, {**row, "call_id": "d"})), encoding="utf-8")
    r = mn.migrate_file(src, dst, kind)
    assert r["rows"] == 2 and r["rows_renamed"] == 2
    new = mn.read_rows(dst)
    assert all("chosen" not in x and "rejected" not in x and "accepted" not in x for x in new)
    assert [to_old_names(x, kind) for x in new] == mn.read_rows(src)


def test_a_numerical_flag_that_already_used_the_new_wording_survives_the_round_trip(tmp_path):
    row = {**NUMERICAL, "flags": ["judge: preferred answer fails", "judge: not-preferred answer passes"]}
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    mn.migrate_file(src, tmp_path / "out.jsonl", "numerical")                       # must not raise
    assert mn.read_rows(tmp_path / "out.jsonl")[0]["flags"] == row["flags"]


def test_the_migration_covers_the_right_files_and_skips_the_rewrite_results():
    k = mn.kind_of
    assert k("training/dpo_train.jsonl") == "training" and k("full-1/final/scored_pairs.jsonl") == "scored"
    assert k("full-1/generation/pairs_raw.jsonl") == "raw" and k("full-1/pairs_raw.jsonl") == "raw"
    assert k("numerical-3/6-agentic-reasoning/x/final/judged_pairs.jsonl") == "numerical"
    assert k("numerical-input/4-workflow-understanding/x.jsonl") == "numerical" and k("numerical-input/duplicates_removed.jsonl") == "numerical"
    assert k("rewrite-2/final/results.jsonl") is None and k("full-1/judge/judge_out.jsonl") is None and k("numerical-input/capabilities.json") is None
