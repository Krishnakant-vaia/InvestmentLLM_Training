"""The eval barrier on training machines (sft_common/barrier.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft_common.barrier import find_eval_material, main  # noqa: E402


def make(root: Path, *rels):
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def test_clean_training_machine_passes(tmp_path):
    make(tmp_path, "sft/02_training/sft_train.py", "sft/data/processed/train.jsonl", "sft/models/pretuned/model.safetensors",
         "venv/lib/site-packages/vllm/eval_bank_like.py")                  # inside a venv: ignored
    assert find_eval_material([tmp_path]) == []
    assert main([str(tmp_path)]) == 0


def test_any_trace_of_the_evals_is_found(tmp_path):
    make(tmp_path, "eval_input/eval_bank.xlsx", "sft/04_evaluation/run_eval_bank.py", "sft/sft_common/eval_bank.py",
         "Downloads/VAIA_Eval_Dataset_finalised.xlsx", "sft/outputs/eval_bank/train_overlap.json")
    found = {Path(p).relative_to(tmp_path).as_posix() for p in find_eval_material([tmp_path])}
    assert {"eval_input", "sft/04_evaluation", "sft/sft_common/eval_bank.py", "Downloads/VAIA_Eval_Dataset_finalised.xlsx",
            "sft/outputs/eval_bank"} <= found
    assert main([str(tmp_path)]) == 1
