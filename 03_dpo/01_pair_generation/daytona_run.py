"""Run the QA-pair generation pipeline on a Daytona GPU sandbox: create it, upload the code, install vLLM, run pipeline.py
detached, follow its log, bring the reports back, delete the sandbox. Modelled on SFT_Script/06_daytona/daytona_run.py.

    python daytona_run.py warm                       # get a GPU, install vLLM, pre-download the model, then WAIT (prints the warm name)
    python daytona_run.py smoke --warm NAME          # 20 chunks, whole path end to end, on the warm sandbox (no re-install)
    python daytona_run.py release --run NAME         # delete a warm sandbox you are not going to use
    python daytona_run.py smoke                      # same, but creates and deletes its own sandbox
    python daytona_run.py pilot                      # 200 chunks: measure pass rates before the full run
    python daytona_run.py full                       # every chunk
    python daytona_run.py full --resume --run NAME   # continue a run (progress comes back from S3)
    python daytona_run.py attach --run NAME          # re-watch a running job (e.g. after closing the laptop)
    options: --run NAME  --gpu H200,H100  --on-demand  --keep  --auto-resume N  --force

The job runs DETACHED on the sandbox: closing this script or the laptop does not stop it (Ctrl-C leaves it running; use
`attach`). Progress files go to S3 every few hundred calls; if a spot sandbox is taken away, a new one is created and the
run resumes from S3, up to --auto-resume times. Daytona deletes the sandbox after its lifetime cap whatever happens.
Needs DAYTONA_API_KEY, DAYTONA_API_URL, DAYTONA_TARGET and the AWS keys in .env; the sandbox gets a copy without DAYTONA_*.
"""
import argparse
import io
import os
import shlex
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import config
import s3util

VENV, CODE_DIR = "/root/venv", "/root/qa"
LOG_FILE, EXIT_FILE = "/root/job.log", "/root/job.exit"
CHUNK = 200_000
GONE = {"destroyed", "destroying", "stopped", "stopping", "error", "build_failed", "archived", "archiving"}
CODE_SUFFIXES = {".py", ".txt"}                        # code and prompts; never .env, work/, outputs/, tests
CODE_EXCLUDE_DIRS = {"work", "outputs", "tests", "__pycache__", ".pytest_cache"}
RESULT_FILES = ("report.json", "first_response.json", "server.log", "errors.jsonl")


def code_files(root: Path = config.HERE) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in CODE_SUFFIXES
            and not set(p.relative_to(root).parts[:-1]) & CODE_EXCLUDE_DIRS and p.name != "daytona_run.py"]


def make_zip(root: Path = config.HERE) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in code_files(root):
            z.write(p, p.relative_to(root).as_posix())
    return buf.getvalue()


def filtered_env(text: str) -> str:
    """.env for the sandbox: everything except the DAYTONA_* lines (the sandbox never talks to Daytona)."""
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("DAYTONA_")) + "\n"


def setup_steps() -> list[tuple[str, str]]:
    act = f". {VENV}/bin/activate && "
    return [("GPU", "nvidia-smi"),
            ("system packages", "command -v git >/dev/null && python3 -m venv --help >/dev/null 2>&1 || "
                                "(apt-get update -qq && apt-get install -y -qq git python3-venv python3-pip)"),
            ("python environment", f"test -x {VENV}/bin/python || python3 -m venv {VENV}"),
            ("install requirements (vLLM: a few minutes)", act + "pip install -q -U pip && pip install -q -r requirements.txt")]


def warm_steps() -> list[tuple[str, str]]:
    """setup_steps plus: check vLLM imports and pre-download the model weights (the long pole, ~65 GB) into the HF cache.
    `original/` and `metal/` hold the same weights in other formats and are skipped. Run from the code folder, so config.py
    loads .env (HF_TOKEN) into the environment: no secret is put on a command line."""
    act = f". {VENV}/bin/activate && "
    download = ("import config; from huggingface_hub import snapshot_download as d; "
                f"p = d('{config.MODEL_REPO}', ignore_patterns=['original/*', 'metal/*']); print('model files in', p)")
    return setup_steps() + [
        ("check vLLM", act + "python -c \"import vllm; print('vllm', vllm.__version__)\""),
        ("download the model weights (~65 GB)", act + f"HF_HOME=/root/hf python -c \"{download}\" && du -sh /root/hf")]


def job_command(run: str, stage: str, resume: bool, sheets: str | None = None, variant: str | None = None) -> str:
    if stage == "rewrite":                                  # hard-negative rewrite of a sample (rewrite_run.py)
        return f"python rewrite_run.py run --run {run}"
    if stage in ("calibrate", "numerical"):               # numerical pairs: judged capability by capability (numerical_run.py)
        return f"python numerical_run.py --run {run}" + (f" --sheets {sheets}" if sheets else "") + (" --per-capability 4" if stage == "calibrate" else "") + (" --resume" if resume else "")
    return f"python pipeline.py --run {run} --stage {stage}" + (f" --variant {variant}" if variant else "") + (" --resume" if resume else "")


def detached(cmd: str) -> str:
    """Shell line that starts `cmd` in the background, immune to this script closing; output in LOG_FILE, exit code in EXIT_FILE."""
    inner = (f"export PYTHONUNBUFFERED=1 HF_HOME=/root/hf; . {VENV}/bin/activate && cd {CODE_DIR} && {cmd}; echo $? > {EXIT_FILE}")
    return f"rm -f {EXIT_FILE}; : > {LOG_FILE}; setsid nohup bash -c {shlex.quote(inner)} > {LOG_FILE} 2>&1 < /dev/null &"


def state_of(sb) -> str:
    s = getattr(sb, "state", None)
    return str(getattr(s, "value", s)).lower()


def gone(sb) -> bool:
    return state_of(sb) in GONE or getattr(sb, "spot_evicted_at", None) is not None


def follow(sb, emit, poll: float = 30.0, misses: int = 4, sleep=time.sleep) -> tuple[str, int | None]:
    """Stream a detached job's log until it finishes: ("done", exit code), or ("lost", None) once the sandbox is confirmed gone."""
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
            if code:
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
                except Exception:  # noqa: BLE001 - cannot even ask: keep trying, never assume it is gone
                    pass
        sleep(poll)


def run_step(sb, session: str, title: str, cmd: str, log, poll: float = 5.0) -> int:
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


def create_sandbox(daytona, gpu: str, spot: bool, ttl: int, labels: dict):
    from daytona import CreateSandboxFromImageParams, GpuType, Resources
    print(f"[daytona] requesting {gpu} ({'spot' if spot else 'on-demand'}, lifetime cap {ttl} min) ...", flush=True)
    return daytona.create(CreateSandboxFromImageParams(
        image=config.DAYTONA_IMAGE, spot=spot, resources=Resources(gpu=1, gpu_type=getattr(GpuType, gpu)),
        auto_stop_interval=0, auto_delete_interval=0, ttl_minutes=ttl, labels=labels), timeout=900)


def live_sandboxes(daytona, run: str) -> list:
    from daytona import ListSandboxesQuery
    return [s for s in daytona.list(ListSandboxesQuery(labels={"dpo-qa-run": run}, limit=20)) if not gone(s)]


def finish_sandbox(sb, ttl: int, keep: bool) -> None:
    if keep:
        print(f"[daytona] sandbox {sb.id} left running. Release it when done (daytona_run.py release --run <warm name>); "
              f"otherwise Daytona removes it at its lifetime cap.", flush=True)
        return
    try:
        sb.delete()
        print(f"[daytona] sandbox {sb.id} deleted", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[daytona] COULD NOT DELETE sandbox {sb.id} ({e}). Delete it in the Daytona dashboard; otherwise it is removed after {ttl} min.", flush=True)


def attempt(daytona, a, run: str, resume: bool, n: int, out_dir: Path) -> tuple[str, int]:
    """One sandbox from creation to deletion. Returns (outcome, exit code): "done", "lost" (spot eviction: safe to resume), "detached"."""
    spot, ttl, sb, gpu = config.DAYTONA_SPOT and not a.on_demand, a.ttl or config.TTL_MINUTES[a.job], None, ""
    if a.warm:                                               # reuse a sandbox prepared by `warm`: no creation, no installs
        found = live_sandboxes(daytona, a.warm)
        if not found:
            print(f"[daytona] no running warm sandbox named {a.warm} (it may have hit its lifetime cap): run `warm` again", flush=True)
            return "done", 2
        sb, gpu = found[0], "(warm)"
    else:
        for gpu in [g.strip() for g in (a.gpu or config.DAYTONA_GPU).split(",")]:
            try:
                sb = create_sandbox(daytona, gpu, spot, ttl, {"dpo-qa-run": run, "dpo-qa-job": a.job})
                break
            except Exception as e:  # noqa: BLE001 - e.g. "No available GPU spot runners"
                print(f"[daytona] no {gpu}: {e}", flush=True)
    if sb is None:
        return "done", 2
    (out_dir / "sandbox.txt").write_text(f"sandbox {sb.id}\ngpu {gpu}\nspot {spot}\nattempt {n}\n")
    keep, code, outcome = a.keep or a.job == "warm" or bool(a.warm), 1, "done"      # a warm sandbox is never deleted by a job
    try:
        with open(out_dir / "console.log", "a", encoding="utf-8") as log:
            sb.process.exec(f"mkdir -p {CODE_DIR}")
            sb.fs.upload_file(make_zip(), "/root/qa.zip")
            sb.process.exec(f"cd {CODE_DIR} && python3 -m zipfile -e /root/qa.zip .")
            sb.fs.upload_file(filtered_env((config.HERE / ".env").read_text(encoding="utf-8")).encode(), f"{CODE_DIR}/.env")
            print(f"[daytona] sandbox {sb.id} ready, code uploaded to {CODE_DIR}", flush=True)
            try:
                sb.process.create_session("job")
            except Exception as e:  # noqa: BLE001 - a reused sandbox already has the session
                if "already exists" not in str(e):
                    raise
            steps = warm_steps() if a.job == "warm" else ([] if a.warm else setup_steps())    # a reused sandbox is already set up
            for title, cmd in steps:
                code = run_step(sb, "job", title, f"cd {CODE_DIR} && {cmd}", log)
                if code != 0:
                    print(f"\n[daytona] step failed ({title}), exit code {code}. Stopping.", flush=True)
                    return "done", code
            if a.job == "warm":
                print(f"\n[daytona] WARM AND WAITING: sandbox {sb.id} ({gpu}) is ready. Start with:\n"
                      f"    python daytona_run.py smoke --warm {a.warm or run}\nIt is removed by its {ttl}-minute cap, or release it with:\n"
                      f"    python daytona_run.py release --run {a.warm or run}", flush=True)
                return "done", 0
            cmd = job_command(run, a.job, resume, a.sheets, a.variant)
            sb.process.exec(detached(cmd), timeout=60)
            print(f"\n===== {a.job} started (detached; Ctrl-C leaves it running; reattach: python daytona_run.py attach --run {run}) =====\n$ {cmd}", flush=True)
            emit = lambda text: (print(text, end="", flush=True), log.write(text), log.flush())  # noqa: E731
            outcome, code = follow(sb, emit)
            if outcome == "lost":
                print(f"\n[daytona] sandbox {sb.id} is gone (spot eviction?).", flush=True)
            else:
                for name in RESULT_FILES:
                    try:
                        data = sb.fs.download_file(f"{CODE_DIR}/work/{run}/{name}")
                        if data:
                            (out_dir / name).write_bytes(data)
                            print(f"[daytona] saved {out_dir / name}", flush=True)
                    except Exception:  # noqa: BLE001 - the file may not exist if the job stopped early
                        pass
    except KeyboardInterrupt:
        keep, outcome = True, "detached"
        print(f"\n[daytona] detached: the job keeps running in sandbox {sb.id}. Re-watch it with:\n    python daytona_run.py attach --run {run}", flush=True)
    except Exception as e:  # noqa: BLE001 - e.g. a spot sandbox taken away mid-setup
        print(f"\n[daytona] lost contact with the sandbox ({type(e).__name__}: {e}).", flush=True)
        try:
            sb.refresh_data()
            outcome = "lost" if gone(sb) else "detached"
        except Exception:  # noqa: BLE001 - cannot ask: assume it still runs, never delete it
            outcome = "detached"
        if outcome == "detached":
            print(f"[daytona] sandbox {sb.id} NOT deleted; it may still be running. Re-watch: python daytona_run.py attach --run {run}", flush=True)
    finally:
        s3util.upload_files([out_dir / "console.log"], f"{run}/logs")
        finish_sandbox(sb, ttl, keep or outcome == "detached")
    return outcome, code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["warm", "smoke", "pilot", "full", "calibrate", "numerical", "rewrite", "attach", "release"])
    ap.add_argument("--variant", choices=["documents", "tearsheet", "cim"], help="smoke/pilot/full jobs: tearsheet or cim = that generator")
    ap.add_argument("--sheets", help="numerical/calibrate jobs: only these sheet numbers, e.g. 6,7")
    ap.add_argument("--ttl", type=int, metavar="MINUTES", help="lifetime cap of a NEW sandbox (default config.TTL_MINUTES[job]); a reused one keeps its own")
    ap.add_argument("--warm", metavar="NAME", help="run the job on the warm sandbox NAME instead of creating a new one")
    ap.add_argument("--run", help="run name (default <job>-<timestamp>); attach and --resume need it")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto-resume", type=int, default=3, metavar="N", help="after a spot eviction, create a new sandbox and resume, up to N times")
    ap.add_argument("--gpu", help="GPU or several in order of preference (default config.DAYTONA_GPU)")
    ap.add_argument("--on-demand", action="store_true")
    ap.add_argument("--keep", action="store_true", help="do not delete the sandbox at the end")
    ap.add_argument("--force", action="store_true", help="start even if this run already has a running sandbox")
    a = ap.parse_args()

    for k in ("DAYTONA_API_KEY", "DAYTONA_API_URL"):
        if not os.environ.get(k):
            raise SystemExit(f"{k} missing in .env")
    from daytona import Daytona, DaytonaConfig
    daytona = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"], api_url=os.environ["DAYTONA_API_URL"],
                                    target=os.environ.get("DAYTONA_TARGET") or "us"))
    if a.job == "attach":
        if not a.run:
            raise SystemExit("attach needs --run <run name>")
        found = live_sandboxes(daytona, a.run)
        if not found:
            raise SystemExit(f"no running sandbox for run {a.run}")
        out_dir = config.HERE / "outputs" / a.run
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "console.log", "a", encoding="utf-8") as log:
            outcome, code = follow(found[0], lambda t: (print(t, end="", flush=True), log.write(t), log.flush()))
        if outcome == "done":
            finish_sandbox(found[0], config.TTL_MINUTES["full"], a.keep)
            return code or 0
        print(f"\n[daytona] sandbox is GONE. Resume: python daytona_run.py full --run {a.run} --resume", file=sys.stderr)
        return 3

    if a.job == "release":
        if not a.run:
            raise SystemExit("release needs --run <warm name>")
        found = live_sandboxes(daytona, a.run)
        for sb in found:
            finish_sandbox(sb, 0, False)
        print(f"[daytona] released {len(found)} sandbox(es) for {a.run}")
        return 0

    run = a.run or f"{a.job}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    if not a.warm and live_sandboxes(daytona, run) and not a.force:
        raise SystemExit(f"run {run} already has a running sandbox (use attach, or --force)")
    out_dir = config.HERE / "outputs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    resume, n = a.resume, 1
    while True:
        outcome, code = attempt(daytona, a, run, resume, n, out_dir)
        if outcome != "lost" or n > a.auto_resume:
            return code
        n, resume = n + 1, True
        print(f"[daytona] resuming from S3 in a new sandbox (attempt {n})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
