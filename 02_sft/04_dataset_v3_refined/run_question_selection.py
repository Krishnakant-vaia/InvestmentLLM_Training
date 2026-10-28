"""Question selection pipeline: extract -> classify -> dedup -> select.

  python run_question_selection.py extract  --out work/
  python run_question_selection.py classify --out work/ [--limit 1000]
  python run_question_selection.py dedup    --out work/
  python run_question_selection.py select   --out work/ [--upload s3://bucket/prefix/]

Each stage reads the previous stage's files in --out, so stages can be rerun
independently. classify and dedup are resumable: finished jobs are skipped, and
rerunning retries the ones that failed.

  classify  one question per call -> standalone, quality, depth, reason
  dedup     questions that pass, grouped by source document -> which ones repeat

Environment (from the shell or --env files):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, [AWS_SESSION_TOKEN], [AWS_DEFAULT_REGION]
  [SEL_BASE_URL]  OpenAI-compatible endpoint, default http://localhost:8000/v1 (vLLM in Daytona)
  [SEL_API_KEY]   default EMPTY
  [SEL_MODEL_ID]  default openai/gpt-oss-120b
"""
import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv

import selection_core as core

HERE = Path(__file__).parent
SOURCE_BUCKET = "ft-vaia"
SOURCE_KEY = "Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_kept.jsonl"
CLASSIFY_PROMPT = HERE / "question_selection_prompt.txt"
DEDUP_PROMPT = HERE / "question_dedup_prompt.txt"
SYSTEM_PROMPT = "Respond with only the JSON object specified. No other text, no Markdown fences."

ROWS_FILE = "questions.jsonl"
LABELS_FILE, LABEL_ERRORS = "labels.jsonl", "classify_errors.jsonl"
DUPS_FILE, DUP_ERRORS = "duplicates.jsonl", "dedup_errors.jsonl"
KEPT_FILE, STATS_FILE = "selected_questions.jsonl", "selection_stats.json"


class FatalAPIError(Exception):
    """Auth or permission failure: stop the run instead of retrying every job."""


def s3_client():
    return boto3.session.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN") or None,
        region_name=os.environ.get("AWS_DEFAULT_REGION") or None,
    ).client("s3")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- stage 1: extract ---

def extract(out):
    """Stream the corpus once; keep one row per unique normalized question."""
    body = s3_client().get_object(Bucket=SOURCE_BUCKET, Key=SOURCE_KEY)["Body"]
    seen, n_in, n_bad, n_dup = set(), 0, 0, 0
    with open(out / ROWS_FILE, "w", encoding="utf-8") as f:
        for line in body.iter_lines(chunk_size=1 << 20):
            if not line:
                continue
            n_in += 1
            try:
                row = core.extract_row(json.loads(line))
            except (json.JSONDecodeError, AttributeError):
                row = None
            if row is None:
                n_bad += 1
                continue
            key = core.normalize_question(row["question"])
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n_in % 50_000 == 0:
                print(f"extract: {n_in} read, {len(seen)} kept", flush=True)
    print(f"extract done: {n_in} read, {len(seen)} kept, {n_dup} exact duplicates, {n_bad} unusable")


# --- model calls ---

def call_model(session, prompt, timeout, max_tokens):
    """OpenAI-compatible chat completion, e.g. vLLM serving gpt-oss-120b inside Daytona."""
    base = os.environ.get("SEL_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    resp = session.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('SEL_API_KEY', 'EMPTY')}"},
        json={
            "model": os.environ.get("SEL_MODEL_ID", "openai/gpt-oss-120b"),
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "reasoning_effort": "low",
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    if resp.status_code in (401, 403):
        raise FatalAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    choices = resp.json().get("choices")
    if not choices:
        raise RuntimeError(f"unrecognized response shape: {resp.text[:300]}")
    return choices[0]["message"].get("content") or ""


def ask(session, prompt, parse, retries, timeout, max_tokens):
    """One call -> parsed result. Network errors and invalid replies are both retried."""
    last = None
    for attempt in range(retries + 1):
        try:
            return parse(call_model(session, prompt, timeout, max_tokens))
        except FatalAPIError:
            raise
        except (requests.RequestException, RuntimeError, ValueError) as e:
            last = e
            time.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError(f"gave up after {retries + 1} attempts: {last}")


def run_jobs(jobs, worker, out_path, err_path, workers, label):
    """jobs: [(job_id, payload)]; worker(payload) -> result dict merged into the output line.
    Appends one line per finished job so a rerun can skip it."""
    lock = threading.Lock()
    n_ok = n_err = 0
    with open(out_path, "a", encoding="utf-8") as of, open(err_path, "a", encoding="utf-8") as ef, \
         ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(worker, payload): jid for jid, payload in jobs}
        for fut in as_completed(futures):
            jid = futures[fut]
            try:
                result = fut.result()
            except FatalAPIError:
                pool.shutdown(cancel_futures=True)
                raise
            except Exception as e:
                with lock:
                    ef.write(json.dumps({"job": jid, "error": str(e)[:500]}) + "\n")
                    n_err += 1
                continue
            with lock:
                of.write(json.dumps({"job": jid} | result, ensure_ascii=False) + "\n")
                of.flush()
                n_ok += 1
                if n_ok % 1000 == 0:
                    print(f"{label}: {n_ok} ok, {n_err} failed of {len(jobs)}", flush=True)
    print(f"{label} done: {n_ok} ok, {n_err} failed (see {err_path.name}); rerun to retry failures")


def new_session(workers):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# --- stage 2: classify (one question per call) ---

def load_labels(out, n):
    labels = [None] * n
    for rec in read_jsonl(out / LABELS_FILE):
        labels[rec["job"]] = rec["label"]
    return labels


def classify(out, workers, retries, timeout, limit):
    rows = read_jsonl(out / ROWS_FILE)
    labels = load_labels(out, len(rows))
    todo = [(i, r["question"]) for i, r in enumerate(rows) if labels[i] is None]
    if limit:
        todo = todo[:limit]
    print(f"classify: {len(rows) - labels.count(None)} done, {len(todo)} to run", flush=True)
    template = CLASSIFY_PROMPT.read_text(encoding="utf-8")
    session = new_session(workers)
    worker = lambda q: {"label": ask(session, core.render_prompt(template, q), core.parse_label, retries, timeout, 1500)}
    run_jobs(todo, worker, out / LABELS_FILE, out / LABEL_ERRORS, workers, "classify")


# --- stage 3: dedup (passing questions, per source document) ---

def load_duplicates(out):
    dups = set()
    for rec in read_jsonl(out / DUPS_FILE):
        dups.update(rec["duplicates"])
    return dups


def dedup(out, workers, retries, timeout):
    rows = read_jsonl(out / ROWS_FILE)
    labels = load_labels(out, len(rows))
    done = {rec["job"] for rec in read_jsonl(out / DUPS_FILE)}
    todo = [(bid, idxs) for bid, idxs in core.dedup_batches(rows, labels) if bid not in done]
    print(f"dedup: {len(done)} batches done, {len(todo)} to run", flush=True)
    template = DEDUP_PROMPT.read_text(encoding="utf-8")
    session = new_session(workers)

    def worker(idxs):
        prompt = core.render_dedup_prompt(template, [rows[i]["question"] for i in idxs])
        dup_of = ask(session, prompt, lambda raw: core.parse_dedup(raw, len(idxs)), retries, timeout, 4000)
        return {"rows": idxs, "duplicates": [idxs[k] for k, d in enumerate(dup_of) if d is not None]}

    run_jobs(todo, worker, out / DUPS_FILE, out / DUP_ERRORS, workers, "dedup")


# --- stage 4: select ---

def select(out, target, upload):
    rows = read_jsonl(out / ROWS_FILE)
    kept, stats = core.select(rows, load_labels(out, len(rows)), load_duplicates(out), target=target)
    with open(out / KEPT_FILE, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats["funnel"], indent=2))
    if len(kept) < target:
        print(f"WARNING: only {len(kept)} rows kept, below target {target}")
    if upload:
        bucket, _, prefix = upload.removeprefix("s3://").partition("/")
        s3 = s3_client()
        for name in (KEPT_FILE, STATS_FILE):
            s3.upload_file(str(out / name), bucket, prefix.rstrip("/") + "/" + name)
        print(f"uploaded to {upload}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["extract", "classify", "dedup", "select"])
    p.add_argument("--out", type=Path, default=HERE / "work")
    p.add_argument("--env", action="append", default=[], help=".env file to load (repeatable)")
    p.add_argument("--workers", type=int, default=128, help="concurrent calls; vLLM batches them on the GPU")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--limit", type=int, default=0, help="classify only N questions (pilot)")
    p.add_argument("--target", type=int, default=140_000)
    p.add_argument("--upload", help="s3://bucket/prefix/ for selected output")
    a = p.parse_args()
    for env in a.env:
        load_dotenv(env, override=True)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "extract":
        extract(a.out)
    elif a.stage == "classify":
        classify(a.out, a.workers, a.retries, a.timeout, a.limit)
    elif a.stage == "dedup":
        dedup(a.out, a.workers, a.retries, a.timeout)
    else:
        select(a.out, a.target, a.upload)


if __name__ == "__main__":
    main()
