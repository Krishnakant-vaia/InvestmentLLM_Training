"""Where a DPO run is saved: under 100626_Dataset_DPO/DPO_Run_1/, one folder per run; smoke runs are named apart."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sft_common.config import load_config  # noqa: E402
from sft_common.runs import Run, dpo_view, is_dpo_run  # noqa: E402
from sft_common.tracking import make_run_name  # noqa: E402

BASE = "s3://ft-vaia/Corpus/Invest_dataset/100626_Dataset_DPO/DPO_Run_1/"


def test_runs_are_saved_under_the_dpo_run_folder_in_the_dataset_bucket_folder():
    view = dpo_view(load_config(ROOT / "config.yaml"))
    smoke, full = Run(view, make_run_name("dpo-smoke")), Run(view, make_run_name("dpo"))
    for run in (smoke, full):
        assert run.store.uri().startswith(BASE) and is_dpo_run(run.name)
        assert run.store.uri("checkpoints/checkpoint-25").startswith(BASE + run.name + "/checkpoints/")
        assert run.store.uri("final_adapter").endswith(run.name + "/final_adapter")
    assert smoke.name.startswith("dpo-smoke-") and not full.name.startswith("dpo-smoke-")
    assert "092826_SFT_FineTune_Model/DPO_Run_1" not in smoke.store.uri() + full.store.uri()


def test_inputs_are_read_from_where_they_are():
    cfg = load_config(ROOT / "config.yaml")["dpo"]
    assert cfg["base"]["s3_uri"].endswith("Fine_Tune_Run_2/merged_models/sft-2026-10-01_23-20_final/")
    assert cfg["data"]["s3_uri"].endswith("100626_Dataset_DPO/qa_pairs/training/dpo_train.jsonl")
    assert cfg["data"]["manifest_s3_uri"].endswith("qa_pairs/training/manifest.json")


def test_a_checkpoint_every_100_steps_with_an_eval_at_each_one():
    d = load_config(ROOT / "config.yaml")["dpo"]
    assert d["save_steps"] == 100 and d["eval_steps"] == 100


def test_the_aws_check_writes_its_probe_in_the_dpo_folder_and_never_deletes():
    src = (ROOT / "01_data_prep" / "check_aws.py").read_text(encoding="utf-8")
    assert "delete_object" not in src and 'cfg.get("dpo")' in src
