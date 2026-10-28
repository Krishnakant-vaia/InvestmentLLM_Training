"""S3 storage and runs, against a fake S3 client (no AWS calls)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from sft_common import runs
from sft_common.s3_sync import COMPLETE_MARKER, RunStore, is_adapter_file, merged_name, merged_store

ROOT_URI = "s3://ft-vaia/Corpus/Invest_dataset/092826_SFT_FineTune_Model/"


class FakeS3:
    """In-memory stand-in for the boto3 S3 client methods RunStore uses."""

    def __init__(self, fail_times=0):
        self.objects: dict[str, bytes] = {}
        self.fail_times = fail_times            # number of upload_file calls that raise first

    def upload_file(self, local, bucket, key):
        if self.fail_times:
            self.fail_times -= 1
            raise OSError("network down")
        self.objects[key] = Path(local).read_bytes()

    def put_object(self, Bucket, Key, Body):  # noqa: N803 - boto3 keyword names
        self.objects[Key] = Body

    def download_file(self, bucket, key, dest):
        Path(dest).write_bytes(self.objects[key])

    def get_paginator(self, _):
        objects = self.objects

        class P:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {"Contents": [{"Key": k} for k in sorted(objects) if k.startswith(Prefix)]}
        return P()


def store(client, run="sft-1"):
    return RunStore(ROOT_URI, run, client=client, wait=0)


def make_ckpt(tmp_path, step):
    d = tmp_path / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"w")
    (d / "trainer_state.json").write_text("{}")
    return d


def test_keys_follow_the_run_layout():
    s = store(FakeS3())
    assert s.key("checkpoints/checkpoint-500/x") == "Corpus/Invest_dataset/092826_SFT_FineTune_Model/sft-1/checkpoints/checkpoint-500/x"
    assert s.uri("merged") == "s3://ft-vaia/Corpus/Invest_dataset/092826_SFT_FineTune_Model/sft-1/merged"


def test_checkpoint_upload_marks_complete_and_is_listed(tmp_path):
    fake = FakeS3()
    s = store(fake)
    assert s.upload_checkpoint(make_ckpt(tmp_path, 500))
    assert s.key(f"checkpoints/checkpoint-500/{COMPLETE_MARKER}") in fake.objects
    assert s.checkpoint_steps() == [500]


def test_half_uploaded_checkpoint_is_not_listed(tmp_path):
    fake = FakeS3()
    s = store(fake)
    fake.objects[s.key("checkpoints/checkpoint-1000/adapter_model.safetensors")] = b"partial"
    assert s.checkpoint_steps() == []                      # no marker -> not a resume candidate


def test_upload_failure_retries_then_reports_false_without_raising(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert store(FakeS3(fail_times=1)).upload_file(f, "logs/a.txt")          # 1 failure, then success
    assert store(FakeS3(fail_times=10)).upload_file(f, "logs/a.txt") is False


def test_env_file_is_never_uploaded(tmp_path):
    (tmp_path / ".env").write_text("AWS_SECRET_ACCESS_KEY=secret")
    (tmp_path / "config.yaml").write_text("a: 1")
    fake = FakeS3()
    assert store(fake).upload_dir(tmp_path, "config")
    assert not any(k.endswith(".env") for k in fake.objects) and any(k.endswith("config.yaml") for k in fake.objects)


def test_download_dir_round_trip(tmp_path):
    fake = FakeS3()
    s = store(fake)
    s.upload_dir(make_ckpt(tmp_path / "src", 500).parent, "final_adapter")
    assert s.download_dir("final_adapter", tmp_path / "dst") == 2
    assert (tmp_path / "dst" / "checkpoint-500" / "trainer_state.json").exists()


def test_merged_models_live_next_to_the_runs_not_inside():
    assert merged_name("sft-1", 3000) == "sft-1_checkpoint-3000" and merged_name("sft-1", None) == "sft-1_final"
    assert merged_store(ROOT_URI, "sft-1", 3000, client=FakeS3()).uri() == \
        "s3://ft-vaia/Corpus/Invest_dataset/092826_SFT_FineTune_Model/merged_models/sft-1_checkpoint-3000"


def test_merge_downloads_only_adapter_files_from_a_checkpoint(tmp_path):
    fake = FakeS3()
    s = store(fake)
    ck = make_ckpt(tmp_path / "src", 3000)
    for name in ("adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        (ck / name).write_bytes(b"x")
    assert s.upload_checkpoint(ck)
    n = s.download_dir("checkpoints/checkpoint-3000", tmp_path / "dst", keep=is_adapter_file)
    got = sorted(p.name for p in (tmp_path / "dst").iterdir())
    assert n == 3 and got == ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json"]


def test_merged_model_upload_is_marked_complete_and_never_touches_a_run_folder(tmp_path):
    fake = FakeS3()
    (tmp_path / "model.safetensors").write_bytes(b"w")
    m = merged_store(ROOT_URI, "sft-1", 3000, client=fake)
    assert m.upload_model(tmp_path)
    assert m.key(COMPLETE_MARKER) in fake.objects and m.key("model.safetensors") in fake.objects
    for bad in (store(fake), RunStore(ROOT_URI, "sft-1/checkpoints", client=fake)):   # run folders
        with pytest.raises(ValueError):
            bad.upload_model(tmp_path)
    assert all("/merged_models/sft-1_checkpoint-3000/" in k for k in fake.objects)


def test_runs_share_folders_and_remember_the_last_one(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "LAST_RUN_FILE", tmp_path / "last_run.txt")
    cfg = {"training": {"output_dir": str(tmp_path / "sft")}, "merging": {"merged_dir": str(tmp_path / "merged")},
           "evaluation": {"out_dir": str(tmp_path / "eval")}, "storage": {"s3_root": ROOT_URI}}
    r = runs.new_run(cfg, "sft-x")
    assert r.adapter_dir == tmp_path / "sft" / "sft-x" / "final" and r.merged_dir == tmp_path / "merged" / "sft-x"
    assert runs.existing_run(cfg).name == "sft-x"          # merge/eval pick up the same run
    assert r.store.uri().endswith("/sft-x")


def test_s3_sync_callback_uploads_checkpoint_and_logs(tmp_path):
    from sft_common.callbacks import S3SyncCallback
    fake = FakeS3()
    run = SimpleNamespace(store=store(fake), logs_dir=tmp_path / "logs", train_dir=tmp_path)
    make_ckpt(tmp_path, 500)
    state = SimpleNamespace(global_step=500, is_world_process_zero=True, log_history=[{"loss": 1.2}])
    cb = S3SyncCallback(run)
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path)), state, None)
    assert run.store.checkpoint_steps() == [500]
    assert json.loads(fake.objects[run.store.key("logs/log_history.json")]) == [{"loss": 1.2}]
    make_ckpt(tmp_path, 1000)                              # e.g. its upload failed earlier
    cb.sync_all(state)
    assert run.store.checkpoint_steps() == [500, 1000]
