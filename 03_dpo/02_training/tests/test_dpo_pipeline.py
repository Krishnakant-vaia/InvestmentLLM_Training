"""DPO around the trainer: run stages, the Daytona dpo job and its eval barrier, merging, and S3 model fetch (no network)."""
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sft_common.barrier import find_eval_material
from sft_common.config import load_config
from sft_common.runs import Run, dpo_view, is_dpo_run, stage_cfg
from sft_common.s3_fetch import fetch_model
from sft_common.s3_sync import merged_store


def launcher():
    spec = importlib.util.spec_from_file_location("dr", ROOT / "06_daytona" / "daytona_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dpo_runs_use_the_dpo_folders_and_sft_runs_are_untouched():
    cfg = load_config()
    assert is_dpo_run("dpo-2026-10-06_12-30") and not is_dpo_run("sft-2026-10-01_23-20")
    assert stage_cfg(cfg, "sft-2026-10-01_23-20") is cfg and stage_cfg(cfg, None) is cfg
    view = stage_cfg(cfg, "dpo-2026-10-06_12-30")
    assert view["storage"]["s3_root"] == cfg["dpo"]["s3_root"] != cfg["storage"]["s3_root"]
    assert view["training"] is cfg["dpo"] and dpo_view(view)["training"] is cfg["dpo"]            # idempotent
    run = Run(view, "dpo-x")
    assert run.train_dir.parts[-2:] == ("dpo", "dpo-x") and run.store.uri().startswith(cfg["dpo"]["s3_root"])
    assert merged_store(view["storage"]["s3_root"], "dpo-x", None).uri() == f"{cfg['dpo']['s3_root']}merged_models/dpo-x_final"


def test_dpo_job_steps_barrier_first_and_no_pool_download():
    dr, cfg = launcher(), load_config()
    steps = dr.job_steps("dpo", cfg, "dpo-x")
    assert steps[0] == dr.BARRIER_STEP
    text = "\n".join(c for _, c in steps)
    assert "build_mixture" not in text and "download_s3" not in text           # dpo_train.py fetches its own inputs
    assert "check_aws" in text and "require_fast_kernels" in text
    cmd = dr.training_command("dpo", cfg, "dpo-x", resume=True, max_steps=10)
    assert cmd == "python 02_training/dpo_train.py --run dpo-x --resume --max-steps 10"
    assert dr.training_command("dpo", cfg, "dpo-x", resume=False) == "python 02_training/dpo_train.py --run dpo-x"
    assert dr.ttl_minutes("dpo", cfg) == cfg["dpo"]["ttl_minutes"] and dr.job_group("dpo") == ("dpo",)
    assert "dpo" not in dr.job_group("train") and dr.RUN_PREFIX["dpo"] == "dpo"


def test_dpo_sandbox_bundle_has_the_trainer_and_no_trace_of_the_evals(tmp_path):
    dr = launcher()
    z = zipfile.ZipFile(io.BytesIO(dr.make_zip(ROOT, "dpo")))
    names = z.namelist()
    assert {"02_training/dpo_train.py", "sft_common/dpo_data.py", "sft_common/dpo_loss.py", "config.yaml"} <= set(names)
    assert not any("04_evaluation" in n or n.endswith(("eval_bank.py", "judge.py")) for n in names)
    z.extractall(tmp_path)
    assert find_eval_material([tmp_path]) == []                                # the barrier step would pass on this machine


def test_dpo_env_has_no_judge_keys():
    dr = launcher()
    env = dr.filtered_env("AWS_ACCESS_KEY_ID=a\nWANDB_API_KEY=w\nCLOUDFLARE_API_TOKEN=c\nDAYTONA_API_KEY=d\n", "dpo")
    assert "AWS_ACCESS_KEY_ID" in env and "WANDB_API_KEY" in env and "CLOUDFLARE" not in env and "DAYTONA" not in env


def test_config_keeps_one_pair_inside_the_memory_envelope_sft_proved():
    d = load_config()["dpo"]
    assert 2 * d["max_pair_tokens"] <= d["tokens_per_micro_batch"] <= 12000         # a longest pair alone = the 12k tokens SFT proved
    assert d["lora"]["dropout"] == 0 and d["data"]["val_fraction"] < 0.5


class FakeS3:
    def __init__(self, keys):
        self.keys = keys

    def list_objects_v2(self, Bucket, Prefix):  # noqa: N803
        return {"KeyCount": sum(k.startswith(Prefix) for k in self.keys)}

    def get_paginator(self, _):
        keys = self.keys

        class P:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                return [{"Contents": [{"Key": k, "Size": 1} for k in keys if k.startswith(Prefix)]}]
        return P()

    def download_file(self, bucket, key, dest):
        Path(dest).write_bytes(b"x")


def test_unfinished_model_upload_is_refused(tmp_path):
    s3 = FakeS3(["m/final/model.safetensors", "m/final/config.json"])             # no _COMPLETE marker
    with pytest.raises(SystemExit, match="_COMPLETE"):
        fetch_model("s3://b/m/final/", tmp_path / "model", client=s3)


def test_finished_model_is_downloaded_once(tmp_path, monkeypatch):
    import sft_common.s3_fetch as f
    monkeypatch.setattr(f, "fix_generation_config", lambda d: None)               # needs a real tokenizer; not what is tested
    s3 = FakeS3(["m/final/model.safetensors", "m/final/config.json", "m/final/_COMPLETE", "m/final/logs/x.txt"])
    out = fetch_model("s3://b/m/final/", tmp_path / "model", client=s3)
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.safetensors"]    # direct children only, no marker, no logs/
    s3.keys.clear()
    assert fetch_model("s3://b/m/final/", out, client=s3) == out                  # already there: nothing is asked of S3
