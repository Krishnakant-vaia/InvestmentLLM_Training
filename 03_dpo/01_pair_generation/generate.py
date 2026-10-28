"""Step 3: assignments -> raw preference pairs, by calling gpt-oss-120b on the local vLLM server.

    python generate.py --chunks work/corpus/chunks.jsonl --assignments work/run/assignments.jsonl --run-dir work/run

Resumable: every finished call is recorded in calls_done.jsonl and skipped on the next start. Failed calls (server errors,
truncation before the first pair, no JSON) go to errors.jsonl and are retried on the next start. Create a file named PAUSE in
the run directory to stop after the calls in flight. Each output line of pairs_raw.jsonl is one pair with the document it was
written from, ready for `score_pairs.py build`.
"""
import argparse
import ast
import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

import config
import s3util

REQUIRED = ("question_type", "defect_type", "question", "preferred_thinking", "preferred", "not_preferred_thinking", "not_preferred", "numeric_check")
OPTIONAL = ("evidence",)                        # kept when present and a list of strings: verbatim quotes score_pairs.code_flags checks against the document
DONE_STATUSES = {"ok", "empty", "invalid"}      # finished; "error" calls are retried


class CallError(Exception):
    """A call that failed and should be retried on the next start."""


# ---- prompt and model ------------------------------------------------------------------------------

def load_prompt(path: Path = config.HERE / "prompts" / "qa_prompt.txt") -> str:
    """GENERATION_PROMPT from qa_prompt.txt, read as a Python literal (nothing in the file is executed)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "GENERATION_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"no GENERATION_PROMPT in {path}")


class LLM:
    """Chat client for an OpenAI-compatible server (vLLM)."""

    def __init__(self, url: str = config.SERVER_URL, model: str = config.SERVED_NAME, workers: int = config.GEN_WORKERS,
                 api_key: str | None = None):
        self.url, self.model = url.rstrip("/") + "/v1/chat/completions", model
        self.session = requests.Session()
        if api_key:                                   # a hosted OpenAI-compatible endpoint (second-opinion judge)
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def chat(self, user: str, system: str = config.GEN_SYSTEM_PROMPT, temperature: float = config.GEN_TEMPERATURE,
             max_tokens: int = config.GEN_MAX_TOKENS, effort: str = config.GEN_REASONING_EFFORT,
             timeout: int = config.GEN_TIMEOUT_S, retries: int = config.GEN_RETRIES) -> dict:
        payload = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   # Top-level field: the only form vLLM 0.31 honours for gpt-oss (measured: low ~600 tokens, default ~1,400, high ~3,800
                   # on the same prompt). chat_template_kwargs and a "Reasoning: low" system line had no effect.
                   "reasoning_effort": effort}
        last = None
        for attempt in range(retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    raise CallError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                choice = data["choices"][0]
                msg = choice["message"]
                return {"content": msg.get("content") or "", "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage") or {}, "message_keys": sorted(msg)}
            except CallError:
                raise
            except Exception as e:  # noqa: BLE001 - connection errors, timeouts, bad JSON: retry with backoff
                last = e
                time.sleep(min(2 ** attempt + random.random(), 30))
        raise CallError(f"gave up after {retries} attempts: {type(last).__name__}: {last}")


# ---- parsing and validation -------------------------------------------------------------------------

def iter_json_objects(text: str):
    """Complete objects of the first JSON array in `text`, tolerating code fences and a reply cut off mid-object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("[")
    if start < 0:
        return
    decoder, i = json.JSONDecoder(), start + 1
    while True:
        while i < len(text) and text[i] in " \n\r\t,":
            i += 1
        if i >= len(text) or text[i] != "{":
            return
        try:
            obj, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return
        yield obj


def wrap_think(text: str) -> str:
    """One <think>...</think> wrapper around a reasoning trace; the model sometimes leaves a tag out (6 of 120 in the smoke test)."""
    inner = re.sub(r"</?think>", "", text).strip()
    return f"<think>{inner}</think>"


KEY_ALIASES = {"chosen": "preferred", "chosen_thinking": "preferred_thinking",              # the old prompt's names, still accepted
               "rejected": "not_preferred", "rejected_thinking": "not_preferred_thinking"}


def normalise_keys(obj):
    """Every prompt now asks for preferred / not_preferred; the old prompt's chosen / rejected keys are renamed on arrival."""
    if not isinstance(obj, dict):
        return obj
    out = {KEY_ALIASES.get(k, k): v for k, v in obj.items()}
    nc = out.get("numeric_check")
    if isinstance(nc, dict) and "rejected_value" in nc:
        out["numeric_check"] = {("not_preferred_value" if k == "rejected_value" else k): v for k, v in nc.items()}
    return out


def validate_pair(obj, assigned: dict[str, str], seen: set[str]) -> str | None:
    """Why this object cannot be used, or None. `assigned` maps each assigned question type to its defect."""
    if not isinstance(obj, dict) or any(k not in obj for k in REQUIRED):
        return "missing keys"
    if obj["question_type"] not in assigned:
        return "question_type was not assigned"
    if obj["question_type"] in seen:
        return "question_type repeated"
    if obj["defect_type"] != assigned[obj["question_type"]]:
        return "defect_type differs from the assignment"
    if not all(isinstance(obj[k], str) and obj[k].strip() for k in ("question", "preferred", "not_preferred", "preferred_thinking", "not_preferred_thinking")):
        return "empty text field"
    if not isinstance(obj["numeric_check"], dict):
        return "numeric_check is not an object"
    return None


# ---- one call ---------------------------------------------------------------------------------------

def run_call(call: dict, chunk: dict, llm: LLM, template: str) -> dict:
    """{"record": calls_done line, "pairs": [...], "debug": {...}}. Raises CallError when the call must be retried."""
    started = time.time()
    prompt = template.format(chunk_text=chunk["text"], question_types=", ".join(call["types"]), defect_types=", ".join(call["defects"]),
                             style=call.get("style") or config.STYLES[0])    # extra keywords are ignored by a template without {style}
    out = llm.chat(prompt)
    text = out["content"]
    if not text.strip():
        raise CallError(f"empty reply (finish_reason={out['finish_reason']}, message keys={out['message_keys']}): "
                        "the answer may have been left in the reasoning field; see VLLM_SERVE_ARGS in config.py")
    objs = list(iter_json_objects(text))
    truncated = out["finish_reason"] == "length"
    if not objs and "[" not in text:
        raise CallError(f"no JSON array in the reply: {text[:200]!r}")
    if not objs and truncated:
        raise CallError("reply cut off before the first pair (raise GEN_MAX_TOKENS or lower pairs per call)")
    assigned, seen, pairs, reasons, unusable = dict(zip(call["types"], call["defects"])), set(), [], Counter(), []
    for o in objs:
        o = normalise_keys(o)
        if isinstance(o, dict):
            for k in ("preferred_thinking", "not_preferred_thinking"):
                if isinstance(o.get(k), str) and o[k].strip():
                    o[k] = wrap_think(o[k])
        why = validate_pair(o, assigned, seen)
        if why:
            reasons[why] += 1
            unusable.append({"call_id": call["call_id"], "chunk_id": chunk["chunk_id"], "reason": why, "object": o})
            continue
        seen.add(o["question_type"])
        pairs.append({**{k: o[k] for k in REQUIRED},
                      **{k: o[k] for k in OPTIONAL if isinstance(o.get(k), list) and all(isinstance(x, str) for x in o[k])},
                      "call_id": call["call_id"], "chunk_id": chunk["chunk_id"],
                      "source_key": chunk["source_key"], "folder": chunk["folder"], "document": chunk["text"],
                      **({"facts": chunk["facts"]} if chunk.get("facts") else {})})
    status = "ok" if pairs else ("empty" if not objs else "invalid")
    usage = out["usage"]
    record = {"call_id": call["call_id"], "status": status, "n_assigned": len(call["types"]), "n_pairs": len(pairs),
              "truncated": truncated, "invalid_reasons": dict(reasons), "finish_reason": out["finish_reason"],
              "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
              "seconds": round(time.time() - started, 1)}
    return {"record": record, "pairs": pairs, "unusable": unusable, "debug": {"message_keys": out["message_keys"], "content_head": text[:600]}}


# ---- the run ----------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def generate(chunks_path: Path, assignments_path: Path, run_dir: Path, workers: int = config.GEN_WORKERS,
             limit_calls: int | None = None, url: str = config.SERVER_URL, s3_run: str | None = None,
             prompt_path: Path = config.HERE / "prompts" / "qa_prompt.txt") -> Counter:
    run_dir.mkdir(parents=True, exist_ok=True)
    chunks = {c["chunk_id"]: c for c in read_jsonl(chunks_path)}
    done = {r["call_id"] for r in read_jsonl(run_dir / "calls_done.jsonl") if r["status"] in DONE_STATUSES}
    pending = [c for c in read_jsonl(assignments_path) if c["call_id"] not in done and c["chunk_id"] in chunks]
    if limit_calls:
        pending = pending[:limit_calls]
    print(f"[generate] {len(pending)} calls to run ({len(done)} already done), {workers} workers", flush=True)
    template, llm, lock, stats = load_prompt(prompt_path), LLM(url, workers=workers), threading.Lock(), Counter()
    pause, files = run_dir / "PAUSE", [run_dir / n for n in ("pairs_raw.jsonl", "pairs_unusable.jsonl", "calls_done.jsonl", "errors.jsonl")]
    started = time.time()

    def work(call):
        try:
            return call, run_call(call, chunks[call["chunk_id"]], llm, template), None
        except CallError as e:
            return call, None, str(e)
        except Exception as e:  # noqa: BLE001 - a bug in one call must not stop the run
            return call, None, f"{type(e).__name__}: {e}"

    def handle(call, result, error):
        with lock:
            stats["calls"] += 1
            if error:
                stats["errors"] += 1
                append_jsonl(run_dir / "errors.jsonl", [{"call_id": call["call_id"], "error": error, "ts": time.time()}])
                return
            rec = result["record"]
            append_jsonl(run_dir / "pairs_raw.jsonl", result["pairs"])
            if result["unusable"]:
                append_jsonl(run_dir / "pairs_unusable.jsonl", result["unusable"])
            append_jsonl(run_dir / "calls_done.jsonl", [rec])
            stats["pairs"] += rec["n_pairs"]
            stats["asked"] += rec["n_assigned"]
            stats["truncated"] += rec["truncated"]
            stats["prompt_tokens"] += rec["prompt_tokens"] or 0
            stats["completion_tokens"] += rec["completion_tokens"] or 0
            if not (run_dir / "first_response.json").exists():
                (run_dir / "first_response.json").write_text(json.dumps(result["debug"], indent=2), encoding="utf-8")
            if stats["calls"] % 25 == 0:
                rate = stats["calls"] / (time.time() - started)
                print(f"[generate] {stats['calls']}/{len(pending)} calls | {stats['pairs']}/{stats['asked']} pairs | "
                      f"errors {stats['errors']} | truncated {stats['truncated']} | {rate:.2f} calls/s | "
                      f"tokens in {stats['prompt_tokens']:,} out {stats['completion_tokens']:,}", flush=True)
            if s3_run and stats["calls"] % config.UPLOAD_EVERY_CALLS == 0:
                s3util.upload_files(files, s3_run)

    with ThreadPoolExecutor(workers) as pool:
        queue, inflight = iter(pending), set()
        while True:
            while len(inflight) < workers * 2 and not pause.exists():
                call = next(queue, None)
                if call is None:
                    break
                inflight.add(pool.submit(work, call))
            if not inflight:
                break
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in finished:
                handle(*f.result())
    if pause.exists():
        print("[generate] PAUSE file found: stopped after the calls in flight. Delete it and rerun to continue.", flush=True)
    if s3_run:
        s3util.upload_files(files, s3_run)
    print(f"[generate] done: {stats['calls']} calls, {stats['pairs']} pairs of {stats['asked']} asked, "
          f"{stats['errors']} errors, {stats['truncated']} truncated", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=config.GEN_WORKERS)
    ap.add_argument("--limit-calls", type=int)
    ap.add_argument("--url", default=config.SERVER_URL)
    ap.add_argument("--s3-run", help="run name: progress files are uploaded to OUT_ROOT/<run>/")
    a = ap.parse_args()
    generate(Path(a.chunks), Path(a.assignments), Path(a.run_dir), a.workers, a.limit_calls, a.url, a.s3_run)


if __name__ == "__main__":
    main()
