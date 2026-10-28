"""Job entry point (runs on the GPU sandbox, or anywhere a gpt-oss-120b server is reachable):
corpus -> assignments -> generation -> judging -> selection -> dpo_pairs.jsonl, everything saved to S3 as it goes.

    python pipeline.py --run smoke-1 --stage smoke        # 20 chunks, the whole path end to end
    python pipeline.py --run pilot-1 --stage pilot        # 200 chunks
    python pipeline.py --run full-1  --stage full         # every chunk
    python pipeline.py --run full-1  --stage full --resume    # after a spot eviction: pulls progress from S3 and continues
    options: --server-url URL (use a server that is already running; no vLLM is started)  --limit-chunks N  --local (no S3)

Outputs in s3://<bucket>/<OUT_ROOT>/<run>/: assignments.jsonl pairs_raw.jsonl calls_done.jsonl errors.jsonl judge_in.jsonl
judge_key.json judge_out.jsonl scored_pairs.jsonl dpo_pairs.jsonl pairs_unusable.jsonl report.json first_response.json server.log
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import requests

import assign
import build_dpo_pairs
import capacity
import config
import generate
import judge
import prep_corpus
import s3util
import score_pairs

# What a run saves on S3 (see s3util.STAGE_DIR for the folder of each). judge_in.jsonl and judge_key.json are not saved: they are rebuilt
# identically from pairs_raw.jsonl, and the judge checkpoints carry a fingerprint of that input. Judge verdicts are checkpointed
# as parts under judge/verdicts/ (checkpoints.py) while the judge runs.
# dpo_pairs.jsonl (every pair in trainer format) is still built on the sandbox but NOT saved: scored_pairs.jsonl is the stored dataset,
# and the pairs to train on are chosen from it by make_training_set.py into qa_pairs/training/dpo_train.jsonl.
RUN_FILES = ("assignments.jsonl", "pairs_raw.jsonl", "pairs_unusable.jsonl", "calls_done.jsonl", "errors.jsonl", "judge_out.jsonl",
             "scored_pairs.jsonl", "report.json", "first_response.json", "server.log")
PULL_FILES = ("pairs_raw.jsonl", "pairs_unusable.jsonl", "calls_done.jsonl", "errors.jsonl")     # generation progress, needed to resume
LAYOUT = """Run {run}: what is where
  generation/  pairs_raw.jsonl (every generated pair), pairs_unusable.jsonl (objects the model returned in an unusable shape),
               calls_done.jsonl, errors.jsonl, assignments.jsonl, first_response.json
  judge/       manifest.json (settings, input fingerprint, counts, status), verdicts/part-NNNNN.jsonl (the LLM judge's verdicts,
               written once each, ~500 pairs per part), judge_out.jsonl (all verdicts merged, written when the judge finishes)
  final/       scored_pairs.jsonl (every pair with judge verdicts and flags), report.json
  (the pairs to TRAIN on are chosen from scored_pairs.jsonl into ../training/dpo_train.jsonl by make_training_set.py)
  logs/        server.log, console logs
Nothing is dropped anywhere: pairs carry flags instead. Filter on passes_all_gates or on the judge verdicts.
"""


def count_lines(path: Path) -> int:
    return sum(1 for line in open(path, encoding="utf-8") if line.strip()) if path.exists() else 0


# ---- the model server -----------------------------------------------------------------------------

def start_server(log_path: Path, vllm_args: str) -> subprocess.Popen:
    cmd = ["vllm", "serve", config.MODEL_REPO, "--served-model-name", config.SERVED_NAME, "--port", str(config.SERVER_PORT)]
    cmd += shlex.split(vllm_args)
    print(f"[server] {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT, env={**os.environ, **config.VLLM_ENV})


def server_is_up(url: str) -> bool:
    """True if a model server already answers at `url` (e.g. left running by an interrupted run): it is reused, not started twice."""
    try:
        return requests.get(f"{url}/v1/models", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def wait_ready(url: str, proc: subprocess.Popen | None, timeout: int = config.SERVER_START_TIMEOUT_S) -> None:
    """Block until the server answers; fail fast if the vLLM process dies (the log has the reason)."""
    started = time.time()
    while time.time() - started < timeout:
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"[server] vLLM exited with code {proc.returncode} before it was ready: see server.log")
        try:
            if requests.get(f"{url}/v1/models", timeout=5).status_code == 200:
                print(f"[server] ready after {int(time.time() - started)} s", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(10)
    raise SystemExit(f"[server] not ready after {timeout} s")


# ---- planning: server size, workers and expected run time (capacity.py) --------------------------------

def plan(calls: list[dict], chunks: dict[str, dict], ttl_hours: float | None = None, template_tokens: int | None = None,
         out_tokens: int | None = None) -> dict:
    """Settings for the GPU found (None values without a GPU) and the expected hours for this job, printed."""
    template_tokens = template_tokens or config.TEMPLATE_TOKENS
    out_tokens = out_tokens or config.OUT_TOKENS_PER_CALL
    gpu = capacity.detect_gpu()
    prompt_tokens = sum(template_tokens + len(chunks[c["chunk_id"]]["text"]) / config.CHARS_PER_TOKEN for c in calls)
    pairs = sum(len(c["types"]) for c in calls) * config.JUDGE_PASS_ESTIMATE
    judge_calls = math.ceil(pairs / config.JUDGE_BATCH)
    avg_chunk = prompt_tokens / max(1, len(calls)) - template_tokens
    judge_tokens = pairs * (avg_chunk + 700) + judge_calls * 2600
    out = {"gpu": gpu[0] if gpu else None, "seqs": None, "workers": None, "judge_workers": None}
    if not gpu:
        print("[plan] no GPU found: using the default worker counts from config.py", flush=True)
        return out
    out["seqs"] = capacity.serve_seqs(gpu[0])
    out["workers"] = int(os.environ.get("DPO_WORKERS") or out["seqs"] + 16)
    out["judge_workers"] = max(1, min(out["seqs"], capacity.max_seqs(gpu[0], 24000)))
    est = capacity.estimate(gpu[0], len(calls), prompt_tokens, out_tokens, judge_calls, judge_tokens)
    print(f"[plan] {gpu[0]} ({gpu[1]:.0f} GB): server --max-num-seqs {out['seqs']}, {out['workers']} generation workers, "
          f"{out['judge_workers']} judge workers", flush=True)
    print(f"[plan] this job: {len(calls)} calls, {prompt_tokens / 1e6:.1f}M prompt tokens, ~{len(calls) * out_tokens / 1e6:.0f}M output tokens (assumed "
          f"{out_tokens}/call). Expected {est['gen_tok_s']:.0f} tok/s at MFU {est['gen_mfu']:.1%} "
          f"(decode is memory-bound; MBU assumed 50%): load {est['load_h']:.1f} h + generate {est['gen_h']:.1f} h + judge {est['judge_h']:.1f} h "
          f"= {est['total_h']:.1f} h", flush=True)
    if ttl_hours and est["total_h"] > 0.9 * ttl_hours:
        print(f"[plan] WARNING: {est['total_h']:.1f} h expected but this sandbox is capped at {ttl_hours:.0f} h. On {gpu[0]} this job will not "
              f"finish in one sandbox: use an H200, or run it in pieces with --resume.", flush=True)
    return out


def report_measured(stats, seconds: float, gpu: str | None, workers: int | None, calls_done: int, calls_full: int) -> None:
    """What generation actually achieved, and what that implies for the full run."""
    if not stats["calls"] or not stats["completion_tokens"]:
        return
    tps = stats["completion_tokens"] / seconds
    line = f"[measured] {stats['calls']} calls in {seconds / 60:.1f} min: {tps:.0f} output tok/s, {stats['completion_tokens'] / stats['calls']:.0f} tokens/call"
    if gpu and workers:
        line += (f" | MFU {capacity.mfu(gpu, tps):.1%}, implied MBU {capacity.implied_mbu(gpu, workers, 7000, tps):.0%} "
                 f"(the estimate assumed 50%)")
    print(line, flush=True)
    if calls_done < calls_full:
        print(f"[measured] full run ({calls_full} calls) at this rate: generation about {calls_full * seconds / stats['calls'] / 3600:.1f} h "
              f"(judging not included)", flush=True)


# ---- the run ----------------------------------------------------------------------------------------

def ensure_corpus(s3, local: bool) -> Path:
    """work/corpus/chunks.jsonl: from S3 if a run already built it there, else built now (and uploaded)."""
    corpus = config.HERE / "work" / "corpus"
    chunks = corpus / "chunks.jsonl"
    if chunks.exists():
        return chunks
    key = f"{config.OUT_ROOT}{config.CORPUS_DIR}/chunks.jsonl"
    if not local and s3util.exists(s3, key):
        s3util.download(s3, key, chunks)
        print(f"[corpus] downloaded {key}", flush=True)
        return chunks
    keys = [k for k in s3util.list_keys(s3, config.DOCS_PREFIX) if k.endswith(".md")]
    result = prep_corpus.prepare(prep_corpus.fetch_all(s3, keys))
    prep_corpus.write_jsonl(chunks, result["chunks"])
    prep_corpus.write_jsonl(corpus / "dropped_docs.jsonl", result["dropped"])
    (corpus / "corpus_report.json").write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
    print(json.dumps(result["report"], indent=2), flush=True)
    if not local:
        s3util.upload_files([corpus / n for n in ("chunks.jsonl", "dropped_docs.jsonl", "corpus_report.json")], config.CORPUS_DIR, s3)
    return chunks


def ensure_tearsheet_corpus(s3, local: bool) -> Path:
    """work/corpus_tearsheets/chunks.jsonl: one chunk per sheet of documents/tear sheets/ (built from S3 each run; 60 small files)."""
    chunks = config.HERE / "work" / "corpus_tearsheets" / "chunks.jsonl"
    if chunks.exists() or local:
        return chunks
    prefix = config.DOCS_PREFIX + config.TEARSHEET_FOLDER + "/"
    keys = [k for k in s3util.list_keys(s3, prefix) if k.endswith(".md")]
    texts = {k: s3.get_object(Bucket=config.BUCKET, Key=k)["Body"].read().decode("utf-8") for k in keys}
    import prep_tearsheets
    rows = prep_tearsheets.sheet_chunks(texts)
    prep_corpus.write_jsonl(chunks, rows)
    print(f"[corpus] {len(rows)} tear sheets", flush=True)
    return chunks


def make_calls(chunks: list[dict], variant: str) -> list[dict]:
    """The assignments of a variant: the tear-sheet grid, the CIM 5-per-chunk mix, or the document passes."""
    if variant == "tearsheet":
        return assign.assign_grid(chunks)
    if variant == "cim":
        return assign.assign_balanced(chunks, **config.CIM_ASSIGN)
    return assign.assign(chunks)


def ensure_cim_corpus(s3, local: bool) -> Path:
    """work/corpus_cims/chunks.jsonl: built by prep_cims.py from the CIM PDFs and their structured JSON, and kept on S3 (the chunks carry the
    JSON's numbers as `facts`); a run downloads it. The CIM text itself is in documents/cims/*.md."""
    chunks = config.HERE / "work" / "corpus_cims" / "chunks.jsonl"
    if chunks.exists() or local:
        return chunks
    key = f"{config.OUT_ROOT}corpus_cims/chunks.jsonl"
    if not s3util.exists(s3, key):
        raise SystemExit(f"{key} is not on S3: build it with prep_cims.py and upload it first")
    s3util.download(s3, key, chunks)
    print(f"[corpus] downloaded {key}", flush=True)
    return chunks


def pull_progress(s3, run: str, work: Path) -> None:
    for name in PULL_FILES:
        # the structured location first; a run started before the layout existed has its files at the top of the run folder
        for key in (s3util.stage_key(run, name), f"{config.OUT_ROOT}{run}/{name}"):
            if not (work / name).exists() and s3util.exists(s3, key):
                s3util.download(s3, key, work / name)
                print(f"[resume] pulled {name} from {key[len(config.OUT_ROOT):]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; its files live in OUT_ROOT/<run>/")
    ap.add_argument("--stage", choices=["smoke", "pilot", "full"], required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--server-url", help="use this running server instead of starting vLLM")
    ap.add_argument("--limit-chunks", type=int, help="override the stage's chunk limit")
    ap.add_argument("--local", action="store_true", help="no S3 at all (the corpus must already be in work/corpus)")
    ap.add_argument("--variant", choices=["documents", "tearsheet", "cim"], default="documents",
                    help="tearsheet: one chunk per tear sheet, every compatible type x defect once; cim: CIM sections, up to 5 pairs per chunk; each with its own prompts")
    a = ap.parse_args()
    tear, cim = a.variant == "tearsheet", a.variant == "cim"

    work = config.HERE / "work" / a.run
    work.mkdir(parents=True, exist_ok=True)
    s3 = None if a.local else s3util.client()
    run_prefix = None if a.local else a.run
    limit = a.limit_chunks if a.limit_chunks is not None else (config.STAGE_LIMITS_TEARSHEET if tear else config.STAGE_LIMITS_CIM if cim else config.STAGE_LIMITS)[a.stage]

    chunks_path = ensure_tearsheet_corpus(s3, a.local) if tear else ensure_cim_corpus(s3, a.local) if cim else ensure_corpus(s3, a.local)
    if s3:
        s3util.put_text(s3, f"{config.OUT_ROOT}{a.run}/LAYOUT.md", LAYOUT.format(run=a.run))
    if a.resume and s3:
        pull_progress(s3, a.run, work)
    if not (work / "assignments.jsonl").exists():
        chunks = assign.sample_chunks(assign.read_jsonl(chunks_path), limit)
        calls = make_calls(chunks, a.variant)
        with open(work / "assignments.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(c) + "\n" for c in calls)
        print(assign.summary(calls), flush=True)

    all_chunks = assign.read_jsonl(chunks_path)
    chunk_map = {c["chunk_id"]: c for c in all_chunks}
    calls_run = assign.read_jsonl(work / "assignments.jsonl")
    settings = plan(calls_run, chunk_map, config.TTL_MINUTES[a.stage] / 60 if not a.server_url else None,
                    *((config.CIM_TEMPLATE_TOKENS, config.CIM_OUT_TOKENS_PER_CALL) if cim else ()))
    vllm_args = config.VLLM_ARGS_OVERRIDE or (f"{config.VLLM_BASE_ARGS} --max-num-seqs {settings['seqs']}" if settings["seqs"]
                                              else f"{config.VLLM_BASE_ARGS} --max-num-seqs 64")
    server, url = None, a.server_url or config.SERVER_URL
    try:
        if not a.server_url and not server_is_up(url):
            server = start_server(work / "server.log", vllm_args)
        wait_ready(url, server)

        started = time.time()
        stats = generate.generate(chunks_path, work / "assignments.jsonl", work, workers=settings["workers"] or config.GEN_WORKERS,
                                  url=url, s3_run=run_prefix,
                                  **({"prompt_path": config.GENERATION_PROMPT_TEARSHEET} if tear else {"prompt_path": config.GENERATION_PROMPT_CIM} if cim else {}))
        calls_full = len(make_calls(all_chunks, a.variant)) if limit else len(calls_run)
        report_measured(stats, time.time() - started, settings["gpu"], settings["workers"], len(calls_run), calls_full)
        first = work / "first_response.json"
        if a.stage == "smoke" and first.exists():
            print(f"[smoke] first reply (check the message keys and that the content is the JSON array):\n{first.read_text()}", flush=True)
        if count_lines(work / "pairs_raw.jsonl") == 0:
            print("[pipeline] no pairs were generated: see errors.jsonl and first_response.json", flush=True)
            return 2

        score_pairs.cmd_build(SimpleNamespace(pairs=str(work / "pairs_raw.jsonl"), out=str(work / "judge_in.jsonl"), key=str(work / "judge_key.json")))
        if count_lines(work / "judge_in.jsonl") == 0:
            print("[pipeline] no pair has all its text fields: nothing to judge. See pairs_unusable.jsonl and errors.jsonl.", flush=True)
            if run_prefix:
                s3util.upload_files([work / n for n in RUN_FILES], run_prefix, s3)
            return 2
        try:
            judge.run(work / "judge_in.jsonl", work / "judge_out.jsonl", url=url,
                      workers=int(config.JUDGE_WORKERS_OVERRIDE or settings["judge_workers"] or config.JUDGE_WORKERS), s3_run=run_prefix,
                      **({"prompt_path": config.JUDGE_PROMPT_TEARSHEET} if tear else {"prompt_path": config.JUDGE_PROMPT_CIM} if cim else {}))
        except RuntimeError as e:                                   # checkpoints made for a different judge input
            print(f"[pipeline] {e}", flush=True)
            return 2
    finally:
        if server is not None:
            server.terminate()

    score_pairs.cmd_select(SimpleNamespace(pairs=str(work / "pairs_raw.jsonl"), judge=str(work / "judge_out.jsonl"),
                                           key=str(work / "judge_key.json"), out=str(work / "scored_pairs.jsonl")))
    report = build_dpo_pairs.build(work / "scored_pairs.jsonl", work / "dpo_pairs.jsonl", work / "report.json")
    print(json.dumps(report, indent=2), flush=True)

    if run_prefix:
        failed = s3util.upload_files([work / n for n in RUN_FILES], run_prefix, s3)
        print(f"[pipeline] saved to s3://{config.BUCKET}/{config.OUT_ROOT}{run_prefix}/" + (f" (failed: {failed})" if failed else ""), flush=True)
    print(f"[pipeline] done: {report['pairs']} training pairs in {work / 'dpo_pairs.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
