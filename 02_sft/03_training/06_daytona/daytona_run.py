"""Run a job on a Daytona GPU sandbox: create it, upload the code, run the steps, bring the
results back, delete the sandbox.

    python 06_daytona/daytona_run.py probe                     # peak GPU memory vs sequence length, ~20 min
    python 06_daytona/daytona_run.py benchmark                 # micro-batch size test (1,2,4,8), ~35 min
    python 06_daytona/daytona_run.py check                     # benchmark + smoke test in one sandbox, ~60 min
    python 06_daytona/daytona_run.py smoke                     # 20 real training steps: proves Trainer + S3 saving
    python 06_daytona/daytona_run.py train                     # the full fine-tuning run (many hours)
    python 06_daytona/daytona_run.py train --resume --run NAME # continue a run (checkpoints come from S3)
    (DPO training is launched from 03_dpo/02_training/06_daytona/daytona_run.py; merge/eval/tokens here accept dpo-... runs)
    python 06_daytona/daytona_run.py attach --run NAME         # re-watch a running job (e.g. after closing the laptop)
    python 06_daytona/daytona_run.py merge --run NAME --checkpoint N  # merge checkpoint N (own sandbox; training untouched; dpo-... runs merge into the SFT model)
    python 06_daytona/daytona_run.py eval --run NAME --checkpoint N --workbook X.xlsx   # answer the eval bank (own sandbox)
    python 06_daytona/daytona_run.py tokens --run NAME --questions Q.jsonl            # output-length statistics (own sandbox)
    options: --gpu H100,H200   --on-demand   --keep   --auto-resume N   --force

How training survives problems:
  * `smoke`, `train` and `dpo` run DETACHED on the sandbox (their output goes to a log file there). If this
    script is closed or the laptop sleeps, training keeps going: use `attach` to watch it again.
    Ctrl-C likewise leaves the sandbox running.
  * Every checkpoint is uploaded to S3 as it is saved. If a spot sandbox is taken away, this script
    creates a new one and resumes from the last S3 checkpoint, up to --auto-resume times (default 3).
  * A run name can only have one running sandbox (labels): starting a second one is refused unless --force.
  * Lifetime caps (config.yaml -> daytona): Daytona deletes the sandbox after ttl_minutes / train_ttl_minutes
    even if this script dies, so a forgotten sandbox cannot run up costs forever.

Needs DAYTONA_API_KEY, DAYTONA_API_URL, DAYTONA_TARGET in .env (see .env.example). The sandbox gets the
code and a copy of .env WITHOUT the DAYTONA_* lines (it does not need them).
Console output goes to outputs/daytona/<name>/console.log; benchmark results also to
<storage.s3_root>/benchmarks/<name>/, and training consoles to <storage.s3_root>/<run>/logs/.
"""
import argparse
import io
import os
import shlex
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
import sft_common  # noqa: E402
from sft_common.config import ROOT, load_config
from sft_common.runs import is_dpo_run, stage_cfg
from sft_common.s3_sync import MERGED_ROOT, RunStore, merged_name
from sft_common.tracking import make_run_name

# Never sent to the sandbox: data and outputs are rebuilt there; models are fetched from S3.
EXCLUDE_DIRS = {".git", "data", "outputs", "models", "wandb", "__pycache__", ".pytest_cache", "node_modules"}
EXCLUDE_FILES = {".env"}                       # replaced by a filtered copy (see filtered_env)
VENV = "/root/venv"                            # the sandbox's own Python environment (see job_steps)
EVAL_VENV = "/root/venv-eval"                  # eval job: separate environment for vLLM
EVAL_INPUT = "/root/eval_input/eval_bank.xlsx" # eval job: where the eval workbook is uploaded (never into the code)
TOKENS_INPUT = "/root/eval_input/token_questions.jsonl"   # tokens job: the question set, uploaded the same way
EVAL_JOBS = ("eval", "tokens")                 # eval-side jobs: own venv, eval code allowed, no training material
CODE_DIR = "/root/sft"                         # where the code is unpacked on the sandbox
LOG_FILE, EXIT_FILE = "/root/job.log", "/root/job.exit"   # a detached job's output and, when done, its exit code
CHUNK = 200_000                                # most log bytes fetched per poll
RESULT_FILES = {"benchmark": ["outputs/sft/checks/logs/benchmark.json", "outputs/sft/checks/logs/startup_report.txt"],
                "check": ["outputs/sft/checks/logs/benchmark.json", "outputs/sft/checks/logs/startup_report.txt"],
                "probe": ["outputs/probe.json"],
                "eval": ["outputs/eval_bank/eval_bank_responses.jsonl", "outputs/eval_bank/train_overlap.json"],
                "tokens": ["outputs/token_stats/token_stats.jsonl", "outputs/token_stats/token_stats_summary.json"]}   # with --tag: see result_files()
RUN_PREFIX = {"benchmark": "benchmark", "probe": "probe", "check": "smoke", "smoke": "smoke", "train": "sft", "dpo": "dpo"}   # run name = <prefix>-YYYY-MM-DD_HH-MM
# Sandbox states in which a job is no longer running (a spot eviction ends in one of these).
GONE = {"destroyed", "destroying", "stopped", "stopping", "error", "build_failed", "archived", "archiving"}


# ---- pure helpers (unit-tested) ---------------------------------------------------------------

# The eval side of the code: never sent to a training-type sandbox (barrier between training and evals).
EVAL_CODE_DIRS = {"04_evaluation"}
EVAL_CODE_FILES = {"eval_bank.py", "test_eval_bank.py", "test_eval.py", "judge.py"}


def project_files(root: Path, job: str | None = None) -> list[Path]:
    """Files to upload: the code, without data, outputs, models, secrets or caches.
    Every job except `eval` also gets no eval code at all."""
    out = []
    no_eval = job not in EVAL_JOBS
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if (p.is_file() and not (set(rel.parts[:-1]) & EXCLUDE_DIRS) and p.name not in EXCLUDE_FILES
                and p.suffix != ".pyc" and not rel.parts[0].startswith("_tmp")
                and not (no_eval and (set(rel.parts[:-1]) & EVAL_CODE_DIRS or p.name in EVAL_CODE_FILES))):
            out.append(p)
    return out


EVAL_DROPS = ("WANDB_", "CLOUDFLARE_", "AZURE_", "OPENAI_", "HF_")   # an eval sandbox needs only AWS (to fetch the model)
TRAIN_DROPS = ("CLOUDFLARE_",)                                      # the judge's keys: eval-only, never on a training machine


def filtered_env(text: str, job: str | None = None) -> str:
    """.env for the sandbox: everything except the DAYTONA_* lines (the sandbox never talks to Daytona).
    An eval sandbox also loses the wandb / Cloudflare / other keys it has no use for (least privilege);
    a training-type sandbox loses the eval judge's (Cloudflare) keys."""
    drop = ("DAYTONA_",) + (EVAL_DROPS if job in EVAL_JOBS else TRAIN_DROPS if job is not None else ())
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith(drop)) + "\n"


def ttl_minutes(job: str, cfg: dict) -> int:
    """Hard lifetime cap for a job's sandbox. Training needs far more than a benchmark."""
    d = cfg["daytona"]
    if job == "dpo":
        return cfg["dpo"]["ttl_minutes"]
    return d["train_ttl_minutes"] if job == "train" else d["ttl_minutes"]


BARRIER_STEP = ("barrier: no eval code, eval workbooks or eval outputs on this machine", "python3 -m sft_common.barrier /root")


def job_steps(job: str, cfg: dict, run: str | None = None, checkpoint: int | None = None, tag: str = "",
              overlap: bool = False, model_uri: str = "") -> list[tuple[str, str]]:
    """(title, shell command) list run to completion, in order, in the code folder on the sandbox.

    For `smoke` and `train` these are only the setup steps; the training itself is started
    afterwards, detached (see training_command / detached). Every job except `eval` starts with
    the barrier check: it stops the job if any trace of the evals is on the machine.
    """
    steps = _job_steps(job, cfg, run, checkpoint, tag, overlap, model_uri)
    return steps if job in EVAL_JOBS else [BARRIER_STEP] + steps


def _job_steps(job: str, cfg: dict, run: str | None = None, checkpoint: int | None = None, tag: str = "",
               overlap: bool = False, model_uri: str = "") -> list[tuple[str, str]]:
    d = cfg["daytona"]
    # The sandbox's Python is Ubuntu's system Python, which refuses system-wide pip installs, so
    # everything runs inside our own virtual environment (VENV); every later step activates it.
    in_venv = f". {VENV}/bin/activate && "
    setup = [
        ("GPU", "nvidia-smi"),
        ("system packages (git for the transformers install, venv support)",
         "command -v git >/dev/null && python3 -m venv --help >/dev/null 2>&1 || "
         "(apt-get update -qq && apt-get install -y -qq git python3-venv python3-pip)"),
        ("python environment", f"test -x {VENV}/bin/python || python3 -m venv {VENV}"),
        ("install requirements", in_venv + "pip install -q -U pip && "
                                           "pip install -q -r requirements/data_prep.txt -r requirements/training.txt"),
        ("check Qwen3.5 fast kernels", in_venv + "python -c \"from sft_common.kernels import require_fast_kernels as r; r(); "
                                              "print('fast kernels OK')\""),
        ("check AWS access", in_venv + "python 01_data_prep/check_aws.py"),
        ("download the pool + build train/val",
         in_venv + "python 01_data_prep/download_s3.py && python 01_data_prep/build_mixture.py"),
    ]
    if job == "probe":       # no dataset needed: synthetic sequences through the real model
        return setup[:5] + [("memory probe (peak GPU memory vs sequence length)", in_venv + "python 02_training/memory_probe.py")]
    if job == "check":       # benchmark, then a smoke test, in ONE sandbox (setup and data prep are ~25 min each time)
        bench = _job_steps("benchmark", cfg)[len(setup):]
        return setup + bench + [("smoke: real training steps (saves 2 checkpoints to S3)",
                                 in_venv + training_command("smoke", cfg, run or "smoke-check", False))]
    if job == "benchmark":
        # --check-packing only applies to (and is only accepted with) batching_mode: packed.
        packing_flag = "--check-packing " if cfg["training"]["batching_mode"] == "packed" else ""
        return setup + [("packing check + micro-batch size benchmark" if packing_flag else "micro-batch size benchmark",
                         in_venv + f"python 02_training/sft_train.py {packing_flag}--benchmark {d['benchmark_sizes']}")]
    if job in ("smoke", "train"):
        return setup
    if job == "dpo":         # no pool/train.jsonl: dpo_train.py fetches the merged SFT model and the pairs file itself
        return setup[:6]
    if job == "merge":       # no dataset needed; reads the checkpoint from S3, writes only merged/ on S3
        what = f"--checkpoint {checkpoint}" if checkpoint is not None else ""
        return setup[:6] + [(f"merge into {MERGED_ROOT}/{merged_name(run, checkpoint)} + sanity check + upload",
                             in_venv + f"python 03_merging/merge_model.py --run {run} {what}".rstrip())]
    if job == "eval":        # own venv (vLLM brings its own torch); the eval workbook is uploaded separately
        in_eval = f". {EVAL_VENV}/bin/activate && "
        what = f"--checkpoint {checkpoint}" if checkpoint is not None else ""
        return setup[:2] + [
            ("python environment (eval)", f"test -x {EVAL_VENV}/bin/python || python3 -m venv {EVAL_VENV}"),
            ("install eval requirements (vLLM)", in_eval + "pip install -q -U pip && pip install -q -r requirements/evaluation.txt"),
            ("check AWS access", in_eval + "python 01_data_prep/check_aws.py"),
            (f"answer the eval bank with {MERGED_ROOT}/{merged_name(run, checkpoint)}"
             + (" (+ train overlap check: training data is copied here temporarily)" if overlap
                else " (no training data on this sandbox)"),
             in_eval + f"python 04_evaluation/run_eval_bank.py --workbook {EVAL_INPUT} --run {run} {what}"
             + (" --overlap" if overlap else "") + (f" --tag {tag}" if tag else "")),
        ]
    if job == "tokens":      # same environment as eval; the question set is uploaded separately (TOKENS_INPUT)
        in_eval = f". {EVAL_VENV}/bin/activate && "
        what = f" --checkpoint {checkpoint}" if checkpoint is not None else ""
        return setup[:2] + [
            ("python environment (eval)", f"test -x {EVAL_VENV}/bin/python || python3 -m venv {EVAL_VENV}"),
            ("install eval requirements (vLLM)", in_eval + "pip install -q -U pip && pip install -q -r requirements/evaluation.txt"),
            ("check AWS access", in_eval + "python 01_data_prep/check_aws.py"),
            (f"output-length statistics for {model_uri or MERGED_ROOT + '/' + merged_name(run, checkpoint)}",
             in_eval + f"python 04_evaluation/token_stats.py --questions {TOKENS_INPUT} --run {run}{what}" + (f" --tag {tag}" if tag else "")
             + (f" --model-uri {shlex.quote(model_uri)}" if model_uri else "")),
        ]
    raise ValueError(f"unknown job {job!r}")


def training_command(job: str, cfg: dict, run: str, resume: bool, max_steps: int | None = None) -> str | None:
    """The detached command of a job (None for jobs that have none, like the benchmark)."""
    if job == "dpo":
        return (f"python 02_training/dpo_train.py --run {run}" + (" --resume" if resume else "")
                + (f" --max-steps {max_steps}" if max_steps else ""))
    if job == "smoke":
        return f"python 02_training/sft_train.py --run {run} --max-steps {cfg['daytona']['smoke_steps']}"
    if job == "train":
        return f"python 02_training/sft_train.py --run {run}" + (" --resume" if resume else "")
    return None


def detached(cmd: str) -> str:
    """Shell line that starts `cmd` in the background, immune to this script closing, with its
    output in LOG_FILE and its exit code written to EXIT_FILE when it finishes."""
    inner = f"export PYTHONUNBUFFERED=1; . {VENV}/bin/activate && cd {CODE_DIR} && {cmd}; echo $? > {EXIT_FILE}"
    return f"rm -f {EXIT_FILE}; : > {LOG_FILE}; setsid nohup bash -c {shlex.quote(inner)} > {LOG_FILE} 2>&1 < /dev/null &"


def state_of(sb) -> str:
    """A sandbox's state as a lowercase string ('started', 'destroyed', ...)."""
    s = getattr(sb, "state", None)
    return str(getattr(s, "value", s)).lower()


def gone(sb) -> bool:
    """True if the sandbox no longer runs jobs (stopped, destroyed, or evicted as a spot instance)."""
    return state_of(sb) in GONE or getattr(sb, "spot_evicted_at", None) is not None


def follow(sb, emit, poll: float = 30.0, misses: int = 4, sleep=time.sleep) -> tuple[str, int | None]:
    """Stream a detached job's log until it finishes. Returns ("done", exit code) or ("lost", None).

    "lost" means the sandbox stopped answering AND is confirmed gone (spot eviction); a few failed
    polls alone (network blips) are just retried, so a running job is never abandoned by mistake.
    """
    offset, failures = 0, 0

    def drain() -> bool:
        nonlocal offset
        text = sb.process.exec(f"tail -c +{offset + 1} {LOG_FILE} | head -c {CHUNK}").result or ""
        if text:
            emit(text)
            offset += len(text.encode("utf-8", "replace"))
        return bool(text)

    while True:
        try:
            drain()
            code = (sb.process.exec(f"cat {EXIT_FILE} 2>/dev/null").result or "").strip()
            failures = 0
            if code:                                   # the job finished: read whatever is left of its log
                while drain():
                    pass
                return "done", int(code)
        except Exception:  # noqa: BLE001 - network blip, or the sandbox is gone
            failures += 1
            if failures >= misses:
                try:
                    sb.refresh_data()
                    if gone(sb):
                        return "lost", None
                except Exception:  # noqa: BLE001 - cannot even ask (our network is down): keep trying, never assume it is gone
                    pass
        sleep(poll)


def s3_summary(cfg: dict, run: str) -> dict:
    """What is on S3 for a run: proof that checkpoints, adapter, logs and inputs were really saved."""
    store = RunStore(cfg["storage"]["s3_root"], run)
    keys = store._keys("")
    top = lambda name: sum(k.startswith(name + "/") for k in keys)  # noqa: E731
    return {"checkpoints": store.checkpoint_steps(), "final_adapter_files": top("final_adapter"),
            "log_files": top("logs"), "config_files": top("config"), "data_files": top("data")}


# ---- sandbox plumbing (needs the Daytona SDK and network) -------------------------------------

def make_zip(root: Path, job: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in project_files(root, job):
            z.write(p, p.relative_to(root).as_posix())
        pkg = Path(sft_common.__file__).resolve().parent     # shared package at the repo root, outside the step
        if not pkg.is_relative_to(root):                    # unpacked as <code dir>/sft_common, same as before
            for p in project_files(pkg, job):
                z.write(p, f"sft_common/{p.relative_to(pkg).as_posix()}")
    return buf.getvalue()


def create_sandbox(daytona, cfg: dict, gpu: str, spot: bool, ttl: int, labels: dict):
    from daytona import CreateSandboxFromImageParams, GpuType, Resources
    print(f"[daytona] requesting {gpu} ({'spot' if spot else 'on-demand'}, lifetime cap {ttl} min) ...", flush=True)
    return daytona.create(CreateSandboxFromImageParams(
        image=cfg["daytona"]["image"], spot=spot, resources=Resources(gpu=1, gpu_type=getattr(GpuType, gpu)),
        auto_stop_interval=0,                    # never idle-stop a long job
        auto_delete_interval=0,                  # delete on stop
        ttl_minutes=ttl,                         # hard lifetime cap: the leak backstop
        labels=labels,                           # lets us find (and refuse to duplicate) this run's sandbox
    ), timeout=900)


def live_sandboxes(daytona, run: str, jobs: tuple[str, ...] | None = None) -> list:
    """Sandboxes labelled with this run that are still running a job (only these jobs, if given):
    a merge sandbox and the training sandbox of the same run never count as each other."""
    from daytona import ListSandboxesQuery
    found = [s for s in daytona.list(ListSandboxesQuery(labels={"sft-run": run}, limit=20)) if not gone(s)]
    return [s for s in found if jobs is None or (getattr(s, "labels", None) or {}).get("sft-job") in jobs]


def job_group(job: str) -> tuple[str, ...]:
    """Jobs that must not run twice at once for the same run (training vs. merging are separate)."""
    return (job,) if job in ("merge", "eval", "tokens", "dpo") else ("benchmark", "probe", "check", "smoke", "train")


def run_step(sb, session: str, title: str, cmd: str, log, poll: float = 5.0) -> int:
    """Run one command in the session, stream its output live, return its exit code."""
    from daytona import SessionExecuteRequest
    line = f"\n===== {title} =====\n$ {cmd}\n"
    print(line, end="", flush=True)
    log.write(line)
    started = sb.process.execute_session_command(session, SessionExecuteRequest(command=cmd, run_async=True))
    seen = 0
    while True:
        logs = sb.process.get_session_command_logs(session, started.cmd_id)
        text = logs.output or ((logs.stdout or "") + (logs.stderr or ""))
        if len(text) > seen:
            print(text[seen:], end="", flush=True)
            log.write(text[seen:])
            log.flush()
            seen = len(text)
        code = sb.process.get_session_command(session, started.cmd_id).exit_code
        if code is not None:
            return code
        time.sleep(poll)


def upload_code(sb, job: str | None = None) -> None:
    sb.process.exec(f"mkdir -p {CODE_DIR}")
    sb.fs.upload_file(make_zip(ROOT, job), "/root/sft.zip")
    sb.process.exec(f"cd {CODE_DIR} && python3 -m zipfile -e /root/sft.zip .")
    sb.fs.upload_file(filtered_env((ROOT / ".env").read_text(encoding="utf-8"), job).encode(), f"{CODE_DIR}/.env")


def finish_sandbox(sb, ttl: int, keep: bool) -> None:
    if keep:
        print(f"[daytona] sandbox {sb.id} left running. DELETE IT when done (Daytona dashboard) "
              f"or it is removed after its {ttl}-minute cap.", flush=True)
        return
    try:
        sb.delete()
        print(f"[daytona] sandbox {sb.id} deleted", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[daytona] COULD NOT DELETE sandbox {sb.id} ({e}). Delete it in the Daytona dashboard; "
              f"otherwise it is removed after {ttl} min.", flush=True)


def attempt(daytona, cfg, a, job: str, run: str, resume: bool, n: int, out_dir: Path) -> tuple[str, int]:
    """One sandbox from creation to deletion. Returns (outcome, exit code):
    outcome "done" (finished, see the code), "lost" (spot eviction: safe to resume), "detached" (Ctrl-C)."""
    spot = cfg["daytona"]["spot"] and not a.on_demand
    ttl = ttl_minutes(job, cfg)
    sb = None
    for gpu in [g.strip() for g in (a.gpu or cfg["daytona"]["gpu"]).split(",")]:   # first GPU type with capacity wins
        try:
            sb = create_sandbox(daytona, cfg, gpu, spot, ttl, {"sft-job": job, "sft-run": run})
            break
        except Exception as e:  # noqa: BLE001 - e.g. "No available GPU spot runners"
            print(f"[daytona] no {gpu}: {e}", flush=True)
    if sb is None:
        return "done", 2
    (out_dir / "sandbox.txt").write_text(f"sandbox {sb.id}\ngpu {gpu}\nspot {spot}\nattempt {n}\n")
    print(f"[daytona] sandbox {sb.id} created; console log: {out_dir / 'console.log'}", flush=True)
    keep, code, outcome = a.keep, 1, "done"
    try:
        with open(out_dir / "console.log", "a", encoding="utf-8") as log:
            upload_code(sb, job)
            print(f"[daytona] code uploaded to {CODE_DIR}", flush=True)
            if job == "eval":             # the eval workbook goes only to this sandbox, outside the code folder
                sb.process.exec(f"mkdir -p {Path(EVAL_INPUT).parent.as_posix()}", timeout=60)
                sb.fs.upload_file(Path(a.workbook).read_bytes(), EVAL_INPUT)
                print(f"[daytona] eval workbook uploaded to {EVAL_INPUT}", flush=True)
            if job == "tokens":           # the question set, like the workbook: only to this sandbox, outside the code
                sb.process.exec(f"mkdir -p {Path(TOKENS_INPUT).parent.as_posix()}", timeout=60)
                sb.fs.upload_file(Path(a.questions).read_bytes(), TOKENS_INPUT)
                print(f"[daytona] question set uploaded to {TOKENS_INPUT}", flush=True)
            sb.process.create_session("job")
            for title, cmd in job_steps(job, cfg, run, getattr(a, "checkpoint", None), getattr(a, "tag", ""),
                                        getattr(a, "overlap", False), getattr(a, "model_uri", "") or ""):
                code = run_step(sb, "job", title, f"cd {CODE_DIR} && {cmd}", log)
                if code != 0:
                    print(f"\n[daytona] step failed ({title}), exit code {code}. Stopping.", flush=True)
                    return "done", code
            main_cmd = training_command(job, cfg, run, resume, getattr(a, "max_steps", None))
            if main_cmd:
                sb.process.exec(detached(main_cmd), timeout=60)
                print(f"\n===== {job} started (detached; Ctrl-C leaves it running; reattach: "
                      f"python 06_daytona/daytona_run.py attach --run {run}) =====\n$ {main_cmd}", flush=True)
                emit = lambda text: (print(text, end="", flush=True), log.write(text), log.flush())  # noqa: E731
                outcome, code = follow(sb, emit)
                if outcome == "lost":
                    print(f"\n[daytona] sandbox {sb.id} is gone (spot eviction?).", flush=True)
            tag = getattr(a, "tag", "")
            for rel in RESULT_FILES.get(job, []):    # bring the results back (an eval tag is a file-name suffix)
                if job in ("eval", "tokens") and tag:
                    stem, ext = rel.rsplit(".", 1)
                    rel = f"{stem}_{tag}.{ext}"
                try:
                    data = sb.fs.download_file(f"{CODE_DIR}/{rel}")
                    if data:
                        (out_dir / Path(rel).name).write_bytes(data)
                        print(f"[daytona] saved {out_dir / Path(rel).name}", flush=True)
                except Exception:  # noqa: BLE001 - the file may not exist if an earlier step failed
                    pass
    except KeyboardInterrupt:
        if job in ("smoke", "train", "dpo"):
            keep, outcome = True, "detached"
            print(f"\n[daytona] detached: {job} keeps running in sandbox {sb.id}. Re-watch it with:\n"
                  f"    python 06_daytona/daytona_run.py attach --run {run}", flush=True)
        else:
            print("\n[daytona] interrupted", flush=True)
    except Exception as e:  # noqa: BLE001 - e.g. a spot sandbox taken away mid-setup
        print(f"\n[daytona] lost contact with the sandbox ({type(e).__name__}: {e}).", flush=True)
        try:
            sb.refresh_data()
            outcome = "lost" if gone(sb) else "detached"     # "lost" only when Daytona confirms it is gone
        except Exception:  # noqa: BLE001 - cannot ask (our network is down): assume it still runs, never delete it
            outcome = "detached"
        if outcome == "detached":
            print(f"[daytona] sandbox {sb.id} NOT deleted; it may still be running. Re-watch: "
                  f"python 06_daytona/daytona_run.py attach --run {run}", flush=True)
    finally:
        if job in ("benchmark", "probe", "check"):  # test results go to S3 next to the runs (kept even if the sandbox was lost)
            store = RunStore(cfg["storage"]["s3_root"], f"benchmarks/{out_dir.name}")
        elif job == "tokens" and getattr(a, "model_uri", None):    # a baseline model: consoles in baselines/, never a merged_models/ folder
            store = RunStore(cfg["storage"]["s3_root"], f"baselines/{run}/logs")
        elif job in ("merge", "eval", "tokens"):    # consoles inside the merged model's own folder: <name>/logs/, <name>/eval/logs/
            store = RunStore(cfg["storage"]["s3_root"], f"{MERGED_ROOT}/{merged_name(run, getattr(a, 'checkpoint', None))}"
                                                        + ("/logs" if job == "merge" else "/eval/logs_tokens" if job == "tokens"
                                                           else "/eval/logs" + (f"_{a.tag}" if getattr(a, "tag", "") else "")))
        else:                             # training consoles live with the run
            store = RunStore(cfg["storage"]["s3_root"], f"{run}/logs")
        for f in (p for p in out_dir.glob("*") if p.is_file()):
            store.upload_file(f, f.name if job in ("benchmark", "probe", "check", "merge", "eval", "tokens")
                              else f"{f.stem}_attempt{n}{f.suffix}")
        print(f"[daytona] results -> {store.uri()}", flush=True)
        if outcome != "detached":
            finish_sandbox(sb, ttl, keep)
        else:
            finish_sandbox(sb, ttl, True)
    return outcome, code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["benchmark", "probe", "check", "smoke", "train", "merge", "eval", "tokens", "attach"])
    ap.add_argument("--checkpoint", type=int, help="merge/eval: checkpoint step (default: the final adapter / model)")
    ap.add_argument("--tag", default="", help="eval: name of this eval bank; outputs get this suffix so earlier evals are kept")
    ap.add_argument("--overlap", action="store_true",
                    help="eval: also check the questions against train.jsonl (copies the training data into the "
                         "eval sandbox, so it is OFF by default)")
    ap.add_argument("--workbook", help="eval: the eval bank .xlsx on this machine (uploaded only to the eval sandbox)")
    ap.add_argument("--model-uri", help="tokens: measure this S3 model folder (e.g. the pre-tuned model); --run is then just a label")
    ap.add_argument("--questions", help="tokens: the question set .jsonl on this machine (uploaded only to that sandbox)")
    ap.add_argument("--run", help="run name (train/smoke: default a new sft-/smoke-<timestamp>; attach: required)")
    ap.add_argument("--resume", action="store_true", help="continue the run from its newest S3 checkpoint")
    ap.add_argument("--max-steps", type=int, help="dpo: smoke test of N optimizer steps (2 checkpoints + 2 evals inside), then stop")
    ap.add_argument("--auto-resume", type=int, default=3, metavar="N",
                    help="if a spot sandbox is taken away, create a new one and resume, up to N times (default 3)")
    ap.add_argument("--gpu", help="GPU, or several in order of preference, e.g. H100,H200 "
                                  "(default: config.yaml daytona.gpu)")
    ap.add_argument("--on-demand", action="store_true", help="on-demand instead of spot")
    ap.add_argument("--keep", action="store_true", help="do not delete the sandbox at the end")
    ap.add_argument("--force", action="store_true", help="start even if this run already has a running sandbox")
    a = ap.parse_args()

    cfg = load_config()
    for k in ("DAYTONA_API_KEY", "DAYTONA_API_URL"):
        if not os.environ.get(k):
            raise SystemExit(f"{k} missing in .env")
    from daytona import Daytona, DaytonaConfig
    daytona = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"], api_url=os.environ["DAYTONA_API_URL"],
                                    target=os.environ.get("DAYTONA_TARGET") or "us"))

    if a.job == "attach":
        if not a.run:
            raise SystemExit("attach needs --run <run name>")
        cfg = stage_cfg(cfg, a.run)
        job = "dpo" if is_dpo_run(a.run) else "train"
        found = live_sandboxes(daytona, a.run, job_group(job))
        if not found:
            raise SystemExit(f"no running sandbox for run {a.run}")
        sb = found[0]
        out_dir = ROOT / "outputs" / "daytona" / a.run
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[daytona] attached to sandbox {sb.id}", flush=True)
        with open(out_dir / "console.log", "a", encoding="utf-8") as log:
            outcome, code = follow(sb, lambda t: (print(t, end="", flush=True), log.write(t), log.flush()))
        if outcome == "done":
            finish_sandbox(sb, ttl_minutes(job, cfg), a.keep)
        else:
            print(f"\n[daytona] sandbox {sb.id} is GONE (spot eviction?). Resume from the last S3 checkpoint:\n"
                  f"    python 06_daytona/daytona_run.py {job} --run {a.run} --resume --auto-resume 8", flush=True)
            return 3
        return code or 0

    if a.job == "tokens" and not (a.questions and Path(a.questions).is_file()):
        raise SystemExit("tokens needs --questions <path to the question set .jsonl>")
    if a.job in ("merge", "eval", "tokens") and not a.run:
        raise SystemExit(f"{a.job} needs --run <run name> (and --checkpoint N for a mid-run checkpoint)")
    if a.job == "eval" and not (a.workbook and Path(a.workbook).is_file()):
        raise SystemExit("eval needs --workbook <path to the eval bank .xlsx>")
    run = a.run or make_run_name(RUN_PREFIX[a.job])
    if a.job == "dpo" and not is_dpo_run(run):
        raise SystemExit("DPO run names start with 'dpo-' (e.g. dpo-2026-10-06_12-30)")
    cfg = stage_cfg(cfg, run)             # DPO runs (training, merge, eval) live under dpo.s3_root
    if not a.force and live_sandboxes(daytona, run, job_group(a.job)):
        raise SystemExit(f"run {run} already has a running {a.job} sandbox. Watch it: attach --run {run} "
                         f"(or pass --force to start another).")
    out_dir = ROOT / "outputs" / "daytona" / run
    if a.job in ("merge", "eval", "tokens"):        # own local folder: never appends to the training console
        out_dir = ROOT / "outputs" / "daytona" / f"{a.job}_{merged_name(run, a.checkpoint)}{'_' + a.tag if a.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tries = 1 + (a.auto_resume if a.job in ("smoke", "train", "dpo") else 0)
    outcome, code = "done", 1
    for n in range(1, tries + 1):
        outcome, code = attempt(daytona, cfg, a, a.job, run, a.resume or n > 1, n, out_dir)
        if outcome != "lost":
            break
        print(f"[daytona] attempt {n}/{tries} lost the sandbox; "
              f"{'resuming from the last S3 checkpoint in a new sandbox' if n < tries else 'no attempts left'}", flush=True)
    if (a.job in ("smoke", "check") or (a.job == "dpo" and a.max_steps)) and outcome == "done" and code == 0:
        print("\n[smoke] what is on S3 for this run:", s3_summary(cfg, run), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
