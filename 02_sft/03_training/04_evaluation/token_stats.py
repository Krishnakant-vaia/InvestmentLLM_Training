"""How long are a merged model's answers? Output-token statistics on a set of finance questions.

Run (on an eval GPU sandbox: 06_daytona/daytona_run.py tokens --run NAME --questions FILE.jsonl):
    python 04_evaluation/token_stats.py --questions /root/eval_input/token_questions.jsonl --run sft-2026-10-01_23-20

Questions file (JSONL): an optional first line {"system_prompt": "..."}, then {"id", "group", "question"} per line.
Every question is answered 4 ways: temperature 0 and 0.7, each with and without the system prompt, thinking on.
Per answer: reasoning tokens (before </think>), answer tokens (after), total, and whether it hit --max-tokens.

Output: outputs/token_stats/token_stats.jsonl (every answer) and token_stats_summary.json (mean, median, p90, max,
cut-off share, tokens/s per setting and per question group), uploaded to merged_models/<name>/eval/token_stats/.
Barrier: like run_eval_bank.py, it refuses to run on a machine holding training data, checkpoints or adapters.
"""
import argparse
import inspect
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")      # no CUDA compiler on the sandbox image (see run_eval_bank.py)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.chat_format import ChatFormat  # noqa: E402
from sft_common.config import ROOT, load_config, resolve  # noqa: E402
from sft_common.eval_bank import barrier_violations  # noqa: E402
from sft_common.runs import stage_cfg  # noqa: E402
from sft_common.pretuned import fix_generation_config  # noqa: E402
from sft_common.s3_models import parse_uri  # noqa: E402
from sft_common.s3_sync import RunStore, merged_name, merged_store  # noqa: E402

OUT_DIR = ROOT / "outputs" / "token_stats"
THINK_END = "</think>"
SETTINGS = [("t0_system", 0.0, True), ("t0_nosystem", 0.0, False), ("t07_system", 0.7, True), ("t07_nosystem", 0.7, False)]


def ensure_model(cfg: dict, name: str, store) -> Path:
    local = resolve(cfg["merging"]["merged_dir"]) / name
    if not (local / "config.json").exists():
        n = store.download_dir("", local, keep=lambda k: not k.startswith(("logs/", "eval/")))
        if not n or not (local / "config.json").exists():
            raise SystemExit(f"no merged model at {store.uri()}")
    return local


def split_tokens(token_ids: list[int], think_end_id: int) -> tuple[int, int]:
    """(reasoning tokens, answer tokens). The prompt opens the think block, so output starts inside it;
    no </think> at all means the answer never started (it is all reasoning)."""
    if think_end_id in token_ids:
        i = token_ids.index(think_end_id)
        return i, len(token_ids) - i - 1
    return len(token_ids), 0


def describe(xs: list[int]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {"mean": round(st.mean(s), 1), "median": s[len(s) // 2], "p90": s[int(len(s) * 0.9)], "max": s[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", type=int)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--model-uri", help="measure any S3 model folder (e.g. the pre-tuned model) instead of a run's merged model; "
                                        "results go to <storage.s3_root>/baselines/<run>/token_stats/, never into that folder")
    ap.add_argument("--tag", default="", help="name of this question set: outputs get this suffix so earlier runs are kept")
    a = ap.parse_args()

    cfg = stage_cfg(load_config(), a.run)
    bad = barrier_violations(ROOT)
    if bad:
        raise SystemExit(f"[barrier] training material found on this machine, stopping: {bad[:5]}")
    lines = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    system = lines.pop(0)["system_prompt"] if "system_prompt" in lines[0] else None
    qs = lines
    if a.model_uri:      # an arbitrary model folder is only READ; outputs go to their own baselines/ folder
        bucket, prefix = parse_uri(a.model_uri)
        parent, _, leaf = prefix.rstrip("/").rpartition("/")
        name, store = a.run, RunStore(cfg["storage"]["s3_root"], f"baselines/{a.run}")
        source = RunStore(f"s3://{bucket}/{parent}", leaf)
    else:
        name = merged_name(a.run, a.checkpoint)
        store = source = merged_store(cfg["storage"]["s3_root"], a.run, a.checkpoint)
    print(f"[tokens] {len(qs)} questions x {len(SETTINGS)} settings; model {name}; system prompt "
          f"{'loaded' if system else 'MISSING'}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    path = ensure_model(cfg, name, source)
    fix_generation_config(path)               # stop at <|im_end|> too, as the merge step does for every merged model
    tok = AutoTokenizer.from_pretrained(path)
    fmt, think_end_id = ChatFormat(tok), tok.convert_tokens_to_ids(THINK_END)
    extra = {}
    try:
        from vllm.engine.arg_utils import EngineArgs
        if "gdn_prefill_backend" in inspect.signature(EngineArgs).parameters:
            extra["gdn_prefill_backend"] = "triton"
    except Exception:  # noqa: BLE001
        pass
    llm = LLM(model=str(path), dtype="bfloat16", max_model_len=cfg["evaluation"]["max_model_len"], trust_remote_code=True, **extra)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {"model": name, "questions": len(qs), "max_tokens": a.max_tokens, "settings": {}}
    for label, temp, with_system in SETTINGS:
        prompts = [fmt.prompt(([{"role": "system", "content": system}] if with_system and system else [])
                              + [{"role": "user", "content": q["question"]}], thinking=True) for q in qs]
        params = SamplingParams(temperature=temp, top_p=0.95 if temp else 1.0, max_tokens=a.max_tokens, seed=3407)
        t0 = time.time()
        outs = llm.generate(prompts, params)
        secs = time.time() - t0
        per = []
        for q, o in zip(qs, outs):
            g = o.outputs[0]
            r_tok, a_tok = split_tokens(list(g.token_ids), think_end_id)
            row = {"setting": label, "id": q["id"], "group": q["group"], "question": q["question"],
                   "prompt_tokens": len(o.prompt_token_ids), "reasoning_tokens": r_tok, "answer_tokens": a_tok,
                   "output_tokens": len(g.token_ids), "truncated": g.finish_reason == "length", "output": g.text}
            rows.append(row)
            per.append(row)
        groups = sorted({r["group"] for r in per})
        summary["settings"][label] = {
            "temperature": temp, "system_prompt": with_system, "seconds": round(secs, 1),
            "tokens_per_second": round(sum(r["output_tokens"] for r in per) / max(secs, 1e-9)),
            "output_tokens": describe([r["output_tokens"] for r in per]),
            "reasoning_tokens": describe([r["reasoning_tokens"] for r in per]),
            "answer_tokens": describe([r["answer_tokens"] for r in per]),
            "answer_words": describe([len(r["output"].split(THINK_END)[-1].split()) for r in per]),
            "hit_limit": sum(r["truncated"] for r in per), "no_answer_after_thinking": sum(r["answer_tokens"] == 0 for r in per),
            "by_group": {grp: {"n": sum(r["group"] == grp for r in per),
                               "answer_tokens": describe([r["answer_tokens"] for r in per if r["group"] == grp]),
                               "reasoning_tokens": describe([r["reasoning_tokens"] for r in per if r["group"] == grp])}
                         for grp in groups},
        }
        print(f"[tokens] {label}: {json.dumps(summary['settings'][label]['output_tokens'])} in {secs:.0f}s", flush=True)

    suffix = f"_{a.tag}" if a.tag else ""
    out, summ = OUT_DIR / f"token_stats{suffix}.jsonl", OUT_DIR / f"token_stats_summary{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    for p in (out, summ):
        ok = store.upload_file(p, f"eval/token_stats/{p.name}")
        print(f"[tokens] -> {store.uri('eval/token_stats/' + p.name)}: {'uploaded' if ok else 'UPLOAD FAILED'}", flush=True)


if __name__ == "__main__":
    main()
