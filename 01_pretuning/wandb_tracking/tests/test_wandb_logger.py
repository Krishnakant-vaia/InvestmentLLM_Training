import pytest

from wandb_logger import (
    build_wandb_config,
    log_run_to_wandb,
    validate_report,
    validate_artifact_uris,
    MODEL_NAME,
    CHUNK_TARGET_TOKENS,
    PACK_SEQ_LEN,
    VAL_FRACTION,
)


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.references = []

    def add_reference(self, uri):
        self.references.append(uri)


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.artifacts_logged = []
        self.finished = False

    def log(self, data):
        self.logged.append(data)

    def log_artifact(self, artifact):
        self.artifacts_logged.append(artifact)

    def finish(self, exit_code=0):
        self.finished = True
        self.exit_code = exit_code


class RaisingWandbRun(FakeWandbRun):
    """A fake run whose .log() always raises, to test crash handling."""

    def log(self, data):
        raise RuntimeError("simulated network failure")


class FakeWandbModule:
    def __init__(self):
        self.run = FakeWandbRun()
        self.init_calls = []

    def init(self, project, config, job_type):
        self.init_calls.append({"project": project, "config": config, "job_type": job_type})
        return self.run

    def Artifact(self, name, type):
        return FakeArtifact(name, type)


class RaisingWandbModule(FakeWandbModule):
    """A fake module whose run raises on .log(), to test crash handling."""

    def __init__(self):
        super().__init__()
        self.run = RaisingWandbRun()


def test_build_wandb_config_includes_tunable_constants():
    config = build_wandb_config()
    assert config["model_name"] == MODEL_NAME
    assert config["chunk_target_tokens"] == CHUNK_TARGET_TOKENS
    assert config["pack_seq_len"] == PACK_SEQ_LEN
    assert config["val_fraction"] == VAL_FRACTION

def test_build_wandb_config_allows_overrides():
    config = build_wandb_config(pack_seq_len=4096, val_fraction=0.1)
    assert config["pack_seq_len"] == 4096
    assert config["val_fraction"] == 0.1
    assert config["model_name"] == MODEL_NAME  # untouched defaults still present

def test_log_run_to_wandb_logs_config_and_report_without_artifact_when_no_uris():
    fake_wandb = FakeWandbModule()
    config = {"pack_seq_len": 8192}
    report = {"chunk_counts": {"train": 10, "val": 1}}

    log_run_to_wandb(fake_wandb, "test-project", config, report, artifact_uris=[])

    assert fake_wandb.init_calls[0]["project"] == "test-project"
    assert fake_wandb.init_calls[0]["config"] == config
    assert fake_wandb.run.logged == [report]
    assert fake_wandb.run.finished is True
    assert fake_wandb.run.artifacts_logged == []

def test_log_run_to_wandb_registers_dataset_artifact_when_uris_given():
    fake_wandb = FakeWandbModule()
    uris = ["s3://bucket/prefix/train.jsonl", "s3://bucket/prefix/val.jsonl"]

    log_run_to_wandb(fake_wandb, "test-project", {}, {}, artifact_uris=uris)

    assert len(fake_wandb.run.artifacts_logged) == 1
    artifact = fake_wandb.run.artifacts_logged[0]
    assert artifact.name == "pretuning-dataset"
    assert artifact.type == "dataset"
    assert artifact.references == uris

def test_log_run_to_wandb_uses_custom_artifact_name():
    fake_wandb = FakeWandbModule()
    uris = ["s3://bucket/prefix/train.jsonl"]

    log_run_to_wandb(fake_wandb, "test-project", {}, {}, artifact_uris=uris, artifact_name="my-corpus-slice")

    assert fake_wandb.run.artifacts_logged[0].name == "my-corpus-slice"

def test_log_run_to_wandb_finishes_run_with_error_exit_code_on_exception():
    fake_wandb = RaisingWandbModule()

    with pytest.raises(RuntimeError, match="simulated network failure"):
        log_run_to_wandb(fake_wandb, "test-project", {}, {"a": 1}, artifact_uris=[])

    assert fake_wandb.run.finished is True
    assert fake_wandb.run.exit_code == 1

def test_log_run_to_wandb_finishes_run_with_success_exit_code_on_success():
    fake_wandb = FakeWandbModule()

    log_run_to_wandb(fake_wandb, "test-project", {}, {"a": 1}, artifact_uris=[])

    assert fake_wandb.run.exit_code == 0

def test_validate_report_accepts_a_dict():
    report = {"chunk_counts": {"train": 10}}
    assert validate_report(report) == report

def test_validate_report_rejects_non_dict():
    with pytest.raises(ValueError, match="report-json must contain a JSON object"):
        validate_report(["not", "a", "dict"])

def test_validate_artifact_uris_accepts_s3_uris():
    validate_artifact_uris(["s3://bucket/train.jsonl", "s3://bucket/val.jsonl"])  # no exception

def test_validate_artifact_uris_accepts_empty_list():
    validate_artifact_uris([])  # no exception

def test_validate_artifact_uris_rejects_non_s3_uri():
    with pytest.raises(ValueError, match="must be an s3:// URI"):
        validate_artifact_uris(["s3://bucket/train.jsonl", "/local/path/val.jsonl"])
