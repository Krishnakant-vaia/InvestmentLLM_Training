"""Step 4: judge both answers of every pair with DPO_pair_judge_prompt.txt (same gpt-oss-120b server, temperature 0).

    python score_pairs.py build pairs_raw.jsonl --out judge_in.jsonl --key judge_key.json     # code gates, random a/b sides
    python judge.py --in judge_in.jsonl --out judge_out.jsonl                                   # this script
    python score_pairs.py select pairs_raw.jsonl --judge judge_out.jsonl --key judge_key.json --out scored_pairs.jsonl

Resumable (lines already in the output file are skipped). A batch whose verdicts fail validation is retried, then split in
half, like the corpus filter; a single record that still fails gets no verdict and `select` flags it "no judge verdict" (it is kept).
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import checkpoints
import config
import s3util
from generate import LLM, CallError, read_jsonl

GATES = {f"G{i}" for i in range(1, 9)}
PLACEHOLDER = "{{batch_records_jsonl}}"


def load_template(path: Path = config.JUDGE_PROMPT) -> str:
    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        raise ValueError(f"{path} has no {PLACEHOLDER} placeholder")
    return text


def valid_side(v) -> bool:
    """One answer's verdict: integer score 1-10, a reason, a gate list; score, reason and gates must agree."""
    if not isinstance(v, dict) or not isinstance(v.get("score"), int) or isinstance(v.get("score"), bool):
        return False
    gates = v.get("failed_gates")
    if not 1 <= v["score"] <= 10 or not isinstance(v.get("reason"), str) or not isinstance(gates, list) or set(gates) - GATES:
        return False
    passed = v["score"] >= 7
    return passed == (v["reason"] == "PASS_ALL_GATES") == (not gates)


def valid_verdicts(n: int, scores) -> bool:
    """Exactly one verdict per position 0..n-1, each with two valid sides."""
    if not isinstance(scores, list) or len(scores) != n:
        return False
    ids = []
    for s in scores:
        if not isinstance(s, dict) or not isinstance(s.get("id"), int) or not (valid_side(s.get("a")) and valid_side(s.get("b"))):
            return False
        ids.append(s["id"])
    return sorted(ids) == list(range(n))


def judge_batch(items: list[dict], llm: LLM, template: str, effort: str = config.JUDGE_REASONING_EFFORT,
                same_size_retries: int = 2) -> list[dict]:
    """Verdicts (with the items' own ids) for a batch; [] for records that could not be judged."""
    lines = [json.dumps({**{k: v for k, v in it.items() if k != "id"}, "id": pos}, ensure_ascii=False) for pos, it in enumerate(items)]
    prompt = template.replace(PLACEHOLDER, "\n".join(lines))
    reason = "?"
    for _ in range(same_size_retries):
        try:
            out = llm.chat(prompt, system="Respond with only the JSON object specified. No other text, no Markdown fences.",
                           temperature=config.JUDGE_TEMPERATURE, max_tokens=config.JUDGE_MAX_TOKENS, effort=effort)
            text = out["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            scores = json.loads(text).get("scores")
            if valid_verdicts(len(items), scores):
                by_pos = {s["id"]: s for s in scores}
                return [{"id": it["id"], "a": by_pos[pos]["a"], "b": by_pos[pos]["b"], "effort": effort} for pos, it in enumerate(items)]
            reason = f"invalid verdicts for a batch of {len(items)} (finish_reason={out['finish_reason']})"
        except (CallError, ValueError, AttributeError) as e:
            reason = f"{type(e).__name__}: {e}"
    if len(items) == 1:
        print(f"[judge] giving up on record {items[0]['id']}: {reason}", flush=True)
        return []
    mid = len(items) // 2
    print(f"[judge] splitting a batch of {len(items)}: {reason}", flush=True)
    return judge_batch(items[:mid], llm, template, effort, same_size_retries) + judge_batch(items[mid:], llm, template, effort, same_size_retries)


def run(in_path: Path, out_path: Path, batch: int = config.JUDGE_BATCH, workers: int = config.JUDGE_WORKERS,
        url: str = config.SERVER_URL, model: str = config.SERVED_NAME, api_key: str | None = None,
        effort: str = config.JUDGE_REASONING_EFFORT, s3_run: str | None = None, upload_every: int = 50,
        prompt_path: Path = config.JUDGE_PROMPT) -> int:
    items = read_jsonl(in_path)
    s3, saved_ids, n_parts, meta = (s3util.client() if s3_run else None), set(), 0, {}
    if s3_run:                         # bring back every verdict already saved on S3 (checks that it is the same judge input)
        input_sha = checkpoints.fingerprint(in_path)
        saved_ids, n_parts, earlier = checkpoints.pull(s3, s3_run, out_path, input_sha)
        meta = {"run": s3_run, "model": model, "effort": effort, "batch": batch, "total": len(items), "input_sha256": input_sha,
                "started_at": (earlier or {}).get("started_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")}
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["id"] not in done]
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    print(f"[judge] {len(todo)} records in {len(batches)} batches ({len(done)} already judged)", flush=True)
    template, llm, lock, n, batches_done = load_template(prompt_path), LLM(url, model, workers, api_key), threading.Lock(), [0], [0]
    parts = [n_parts]

    def work(b):
        verdicts = judge_batch(b, llm, template, effort)
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(v) + "\n" for v in verdicts)
            n[0] += len(verdicts)
            batches_done[0] += 1
            print(f"[judge] {n[0]}/{len(todo)} judged", flush=True)
            if s3_run and batches_done[0] % upload_every == 0:        # a spot sandbox can vanish: keep the verdicts safe on S3
                parts[0] = checkpoints.push(s3, s3_run, out_path, saved_ids, parts[0], meta)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, batches))
    if s3_run:
        checkpoints.push(s3, s3_run, out_path, saved_ids, parts[0], meta, complete=True)
        s3util.upload_files([out_path], s3_run)                      # the merged verdicts too, for convenience: judge/judge_out.jsonl
    return n[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=config.JUDGE_BATCH)
    ap.add_argument("--workers", type=int, default=config.JUDGE_WORKERS)
    ap.add_argument("--effort", default=config.JUDGE_REASONING_EFFORT, choices=["low", "medium", "high"], help="reasoning effort of the judge")
    ap.add_argument("--url", default=config.SERVER_URL)
    ap.add_argument("--model", default=config.SERVED_NAME, help="model name at that endpoint (a second-opinion judge)")
    ap.add_argument("--prompt", default=str(config.JUDGE_PROMPT), help="judge prompt file (DPO_pair_judge_prompt_numerical.txt for the numerical set)")
    ap.add_argument("--api-key-env", help="name of the environment variable that holds the endpoint's API key")
    a = ap.parse_args()
    run(Path(a.inp), Path(a.out), a.batch, a.workers, a.url, a.model, os.environ.get(a.api_key_env) if a.api_key_env else None, a.effort,
        prompt_path=Path(a.prompt))


if __name__ == "__main__":
    main()
