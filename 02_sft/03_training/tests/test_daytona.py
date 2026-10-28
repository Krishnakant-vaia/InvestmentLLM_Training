"""Daytona launcher: what is uploaded, that secrets are handled, and the job steps (no network)."""
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def launcher():
    spec = importlib.util.spec_from_file_location("dr", ROOT / "06_daytona" / "daytona_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_upload_has_code_but_no_data_secrets_or_caches(tmp_path):
    dr = launcher()
    for rel in ["01_data_prep/a.py", "config.yaml", "sft_common/x.py", ".env", "data/raw/big.jsonl", "outputs/sft/ckpt/w.bin",
                "models/pretuned/model.safetensors", "wandb/run/log", "sft_common/__pycache__/x.pyc", ".git/HEAD",
                "_tmp_test/f.txt", "tests/t.py"]:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    names = {p.relative_to(tmp_path).as_posix() for p in dr.project_files(tmp_path)}
    assert names == {"01_data_prep/a.py", "config.yaml", "sft_common/x.py", "tests/t.py"}


def test_zip_of_the_real_project_excludes_secrets():
    dr = launcher()
    names = zipfile.ZipFile(io.BytesIO(dr.make_zip(ROOT))).namelist()
    assert "config.yaml" in names and "02_training/sft_train.py" in names and "sft_common/batching.py" in names
    assert ".env" not in names and not any(n.startswith(("data/", "outputs/", "models/", "wandb/")) for n in names)


def test_sandbox_env_drops_only_the_daytona_lines():
    dr = launcher()
    text = "AWS_ACCESS_KEY_ID=a\n# Daytona\nDAYTONA_API_KEY=secret\nDAYTONA_API_URL=u\nWANDB_API_KEY=w\n"
    out = dr.filtered_env(text)
    assert "DAYTONA_API_KEY" not in out and "DAYTONA_API_URL" not in out and "secret" not in out
    assert "AWS_ACCESS_KEY_ID=a" in out and "WANDB_API_KEY=w" in out


def test_benchmark_job_steps_packed_mode_adds_the_packing_check():
    dr = launcher()
    cfg = {"daytona": {"benchmark_sizes": "1,2,4,8"}, "training": {"batching_mode": "packed"}}
    steps = dr.job_steps("benchmark", cfg)
    cmds = [c for _, c in steps]
    assert cmds[0] == "python3 -m sft_common.barrier /root"                   # the eval barrier comes first
    assert cmds[1] == "nvidia-smi" and any("check_aws.py" in c for c in cmds)
    assert cmds[-1].endswith("python 02_training/sft_train.py --check-packing --benchmark 1,2,4,8")
    assert all(". /root/venv/bin/activate" in c for c in cmds[4:])            # pip/python run inside the venv


def test_benchmark_job_steps_padded_mode_omits_the_packing_check():
    # --check-packing is only accepted by sft_train.py when batching_mode is packed; padded runs
    # must not pass it, or the sandbox job fails immediately (as it did on 2026-09-29 before this fix).
    dr = launcher()
    cfg = {"daytona": {"benchmark_sizes": "1,2,4,8"}, "training": {"batching_mode": "padded"}}
    cmds = [c for _, c in dr.job_steps("benchmark", cfg)]
    assert cmds[-1].endswith("python 02_training/sft_train.py --benchmark 1,2,4,8")
    assert "--check-packing" not in cmds[-1]


def test_job_steps_rejects_unknown_job():
    import pytest
    dr = launcher()
    with pytest.raises(ValueError):
        dr.job_steps("nope", {"daytona": {"benchmark_sizes": "1"}, "training": {"batching_mode": "padded"}})


def test_config_has_the_daytona_settings_the_launcher_reads():
    from sft_common.config import load_config
    d = load_config()["daytona"]
    assert set(d["gpu"].split(",")) <= {"H100", "H200", "B300", "RTX_PRO_6000"} and isinstance(d["spot"], bool)
    assert d["ttl_minutes"] > 0 and d["benchmark_sizes"]


# ---- Qwen3.5 fast-kernel guard ------------------------------------------------------

def test_missing_fast_kernels_are_reported_and_stop_the_run():
    import pytest
    from types import SimpleNamespace
    from sft_common.kernels import missing_fast_kernels, require_fast_kernels
    ok = SimpleNamespace(chunk_gated_delta_rule=object(), fused_recurrent_gated_delta_rule=object())
    assert missing_fast_kernels(ok) == []
    require_fast_kernels(ok)                                                   # no exception
    # transformers sets these to None when flash-linear-attention is not installed
    broken = SimpleNamespace(chunk_gated_delta_rule=None, fused_recurrent_gated_delta_rule=None)
    assert missing_fast_kernels(broken) == ["chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"]
    with pytest.raises(SystemExit, match="flash-linear-attention"):
        require_fast_kernels(broken)


def test_daytona_job_checks_the_kernels_right_after_the_install():
    dr = launcher()
    steps = dr.job_steps("benchmark", {"daytona": {"benchmark_sizes": "1"}, "training": {"batching_mode": "padded"}})
    titles = [t for t, _ in steps]
    assert titles.index("check Qwen3.5 fast kernels") == titles.index("install requirements") + 1   # fail in minutes


def test_flash_linear_attention_is_pinned_in_the_requirements():
    text = (ROOT / "requirements" / "training.txt").read_text(encoding="utf-8")
    assert any(l.strip().startswith("flash-linear-attention==") for l in text.splitlines())


# ---- smoke / train jobs, detached running, following a job ---------------------------------

from types import SimpleNamespace  # noqa: E402

CFG = {"daytona": {"benchmark_sizes": "1", "smoke_steps": 20, "ttl_minutes": 240, "train_ttl_minutes": 2400},
       "training": {"batching_mode": "padded"}}


def test_ttl_is_long_for_training_only():
    dr = launcher()
    assert dr.ttl_minutes("train", CFG) == 2400 and dr.ttl_minutes("benchmark", CFG) == dr.ttl_minutes("smoke", CFG) == 240


def test_training_commands():
    dr = launcher()
    assert dr.training_command("smoke", CFG, "smoke-x", False) == "python 02_training/sft_train.py --run smoke-x --max-steps 20"
    assert dr.training_command("train", CFG, "sft-x", False) == "python 02_training/sft_train.py --run sft-x"
    assert dr.training_command("train", CFG, "sft-x", True).endswith("--run sft-x --resume")
    assert dr.training_command("benchmark", CFG, "b", False) is None
    setup = dr.job_steps("train", CFG)                                   # only setup: training itself is detached
    assert setup == dr.job_steps("smoke", CFG) and not any("sft_train.py" in c for _, c in setup)


def test_detached_wrapper_survives_the_caller_and_records_the_exit_code():
    dr = launcher()
    line = dr.detached("python 02_training/sft_train.py --run r")
    assert "setsid nohup" in line and line.rstrip().endswith("&")           # background, session-detached
    assert dr.LOG_FILE in line and f"echo $? > {dr.EXIT_FILE}" in line     # output file + exit code file
    assert "PYTHONUNBUFFERED=1" in line and "/root/venv/bin/activate" in line


def test_gone_states_include_spot_eviction():
    dr = launcher()
    assert not dr.gone(SimpleNamespace(state="started", spot_evicted_at=None))
    assert dr.gone(SimpleNamespace(state="destroyed", spot_evicted_at=None))
    assert dr.gone(SimpleNamespace(state="started", spot_evicted_at="2026-09-29T10:00:00Z"))   # evicted spot
    assert dr.gone(SimpleNamespace(state=SimpleNamespace(value="stopped"), spot_evicted_at=None))


class FakeSandbox:
    """Serves a job log in pieces, then the exit code; can drop calls to mimic network trouble or an eviction."""

    def __init__(self, log_parts, exit_code="0", fail_first=0, dead_after=None, state="started"):
        self.log, self.parts, self.exit_code = "", list(log_parts), exit_code
        self.fail_first, self.calls, self.dead_after, self.state = fail_first, 0, dead_after, state
        self.spot_evicted_at = None
        self.process = SimpleNamespace(exec=self._exec)

    def _exec(self, cmd, timeout=None):
        self.calls += 1
        if self.calls <= self.fail_first or (self.dead_after is not None and self.calls > self.dead_after):
            raise ConnectionError("sandbox unreachable")
        if cmd.startswith("tail -c +"):
            start = int(cmd.split("+")[1].split()[0]) - 1
            if self.parts:
                self.log += self.parts.pop(0)
            return SimpleNamespace(result=self.log.encode()[start:].decode(), exit_code=0)
        if cmd.startswith("cat "):
            return SimpleNamespace(result=self.exit_code if not self.parts else "", exit_code=0)
        raise AssertionError(cmd)

    def refresh_data(self):
        pass


def test_follow_streams_the_whole_log_and_returns_the_exit_code():
    dr = launcher()
    seen = []
    sb = FakeSandbox(["step 1\n", "step 2\n", "done \u00e9\n"], exit_code="0")
    outcome, code = dr.follow(sb, seen.append, poll=0, sleep=lambda s: None)
    assert (outcome, code) == ("done", 0)
    assert "".join(seen) == "step 1\nstep 2\ndone \u00e9\n"                  # nothing lost or repeated, unicode intact
    assert dr.follow(FakeSandbox(["boom\n"], exit_code="1"), lambda t: None, poll=0, sleep=lambda s: None) == ("done", 1)


def test_follow_survives_network_blips_but_reports_a_confirmed_eviction():
    dr = launcher()
    blip = FakeSandbox(["a\n", "b\n"], fail_first=6)                       # 6 failed polls, sandbox still started
    seen = []
    assert dr.follow(blip, seen.append, poll=0, misses=3, sleep=lambda s: None) == ("done", 0)
    assert "".join(seen) == "a\nb\n"
    evicted = FakeSandbox(["a\n"], dead_after=2, state="destroyed")
    assert dr.follow(evicted, lambda t: None, poll=0, misses=3, sleep=lambda s: None) == ("lost", None)


def test_config_has_the_training_job_settings():
    from sft_common.config import load_config
    d = load_config()["daytona"]
    assert d["train_ttl_minutes"] > d["ttl_minutes"] and d["smoke_steps"] >= 2


def test_probe_job_needs_no_data_and_returns_its_results():
    dr = launcher()
    steps = dr.job_steps("probe", CFG)
    titles = [t for t, _ in steps]
    assert titles[-1].startswith("memory probe") and steps[-1][1].endswith("python 02_training/memory_probe.py")
    assert not any("download the pool" in t or "AWS" in t for t in titles)            # synthetic inputs only
    assert "check Qwen3.5 fast kernels" in titles
    assert dr.RESULT_FILES["probe"] == ["outputs/probe.json"] and dr.RUN_PREFIX["probe"] == "probe"


def test_merge_job_needs_no_data_and_merges_the_given_checkpoint():
    dr = launcher()
    steps = dr.job_steps("merge", CFG, "sft-x", 3000)
    titles = [t for t, _ in steps]
    assert steps[-1][1].endswith("python 03_merging/merge_model.py --run sft-x --checkpoint 3000")
    assert "merged_models/sft-x_checkpoint-3000" in titles[-1]
    assert "check AWS access" in titles and not any("download the pool" in t for t in titles)
    assert dr.job_steps("merge", CFG, "sft-x")[-1][1].endswith("merge_model.py --run sft-x")   # final adapter
    assert dr.training_command("merge", CFG, "sft-x", False) is None                           # nothing detached


def test_eval_job_uses_its_own_env_and_never_touches_training_data_prep():
    dr = launcher()
    steps = dr.job_steps("eval", CFG, "sft-x", 3000)
    cmds = " ".join(c for _, c in steps)
    assert steps[-1][1].endswith(f"run_eval_bank.py --workbook {dr.EVAL_INPUT} --run sft-x --checkpoint 3000")
    assert "--overlap" not in steps[-1][1] and "no training data" in steps[-1][0]            # default: no train data at all
    assert steps[-1][1].endswith("--overlap") is False
    assert dr.job_steps("eval", CFG, "sft-x", 3000, overlap=True)[-1][1].endswith("--checkpoint 3000 --overlap")
    assert dr.job_steps("eval", CFG, "sft-x", 3000, "bank2")[-1][1].endswith("--checkpoint 3000 --tag bank2")
    assert dr.EVAL_VENV in cmds and "requirements/training.txt" not in cmds and "build_mixture" not in cmds
    assert not dr.EVAL_INPUT.startswith(dr.CODE_DIR)                  # the workbook is not part of the code
    assert dr.job_group("eval") == ("eval",) and "eval_bank_responses.jsonl" in dr.RESULT_FILES["eval"][0]


def test_eval_sandbox_gets_only_the_keys_it_needs():
    dr = launcher()
    env = "AWS_ACCESS_KEY_ID=a\nWANDB_API_KEY=w\nCLOUDFLARE_API_TOKEN=c\nDAYTONA_API_KEY=d\nAZURE_CLIENT_SECRET=z\n"
    assert dr.filtered_env(env, "eval") == "AWS_ACCESS_KEY_ID=a\n"                 # AWS only: to fetch the model
    assert dr.filtered_env(env, "train") == "AWS_ACCESS_KEY_ID=a\nWANDB_API_KEY=w\nAZURE_CLIENT_SECRET=z\n"   # no judge keys
    assert dr.filtered_env(env, "merge") == dr.filtered_env(env, "train")          # every training-type job alike
    assert "CLOUDFLARE_API_TOKEN=c" in dr.filtered_env(env)                         # no job given: only DAYTONA_* dropped


def test_training_sandboxes_get_no_eval_code_and_start_with_the_barrier(tmp_path):
    dr = launcher()
    for rel in ("02_training/sft_train.py", "04_evaluation/run_eval_bank.py", "04_evaluation/evaluate.py",
                "sft_common/eval_bank.py", "sft_common/judge.py", "sft_common/barrier.py", "tests/test_eval_bank.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    names = lambda job: {p.relative_to(tmp_path).as_posix() for p in dr.project_files(tmp_path, job)}  # noqa: E731
    for job in ("train", "smoke", "check", "benchmark", "probe", "merge"):
        assert names(job) == {"02_training/sft_train.py", "sft_common/barrier.py"}, job
        assert dr.job_steps(job, CFG, "sft-x", 3000 if job == "merge" else None)[0] == dr.BARRIER_STEP
    assert "04_evaluation/run_eval_bank.py" in names("eval") and "sft_common/eval_bank.py" in names("eval")
    assert dr.job_steps("eval", CFG, "sft-x", 3000)[0] != dr.BARRIER_STEP


def test_merge_and_training_sandboxes_are_told_apart():
    dr = launcher()
    sbs = [SimpleNamespace(state="started", labels={"sft-run": "r", "sft-job": "train"}),
           SimpleNamespace(state="started", labels={"sft-run": "r", "sft-job": "merge"})]
    fake = SimpleNamespace(list=lambda q: sbs)
    assert [s.labels["sft-job"] for s in dr.live_sandboxes(fake, "r", dr.job_group("merge"))] == ["merge"]
    assert [s.labels["sft-job"] for s in dr.live_sandboxes(fake, "r", dr.job_group("train"))] == ["train"]


def test_check_job_is_benchmark_then_smoke_in_one_sandbox():
    dr = launcher()
    steps = dr.job_steps("check", CFG, "smoke-x")
    titles = [t for t, _ in steps]
    bench = dr.job_steps("benchmark", CFG)
    assert steps[: len(bench)] == bench                                       # everything the benchmark does, first
    assert titles[-1].startswith("smoke") and steps[-1][1].endswith("--run smoke-x --max-steps 20")
    assert dr.RUN_PREFIX["check"] == "smoke" and dr.RESULT_FILES["check"] == dr.RESULT_FILES["benchmark"]
