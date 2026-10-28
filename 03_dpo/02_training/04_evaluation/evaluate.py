"""Step 4: does the fine-tuned model actually work? Compare it with the pre-tuned model.

Run:  python 04_evaluation/evaluate.py              # the last run; both models, then comparison
      python 04_evaluation/evaluate.py --run NAME   # a specific run (merged model pulled from S3 if needed)
      python 04_evaluation/evaluate.py --no-judge   # skip the gpt-oss-120b judge

What it does, per model (pre-tuned, then fine-tuned):
  1. takes evaluation.samples_per_source held-out val rows per source (never trained on);
  2. cuts each row into prompt + reference turn (the first tool call, else the last answer);
  3. generates with vLLM, greedy (temperature 0, repeatable); thinking on for the sources in
     evaluation.thinking_sources (domain answers were trained with reasoning), off otherwise;
  4. scores each answer (sft_common/metrics.py) and, for evaluation.judge.sources, asks the
     judge (sft_common/judge.py) for correctness / reasoning scores;
  5. averages per source.
Then it prints a pre-tuned vs fine-tuned table and uploads everything to <run>/eval/ on S3.

Each model is generated in its own subprocess, so the GPU is fully freed between the two.
Outputs (outputs/eval/<run>/): <model>/generations.jsonl, <model>/metrics.json, comparison.json
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.chat_format import ChatFormat, split_think
from sft_common.config import load_config, resolve
from sft_common.judge import CloudflareJudge, judge_many
from sft_common.metrics import aggregate, pick_target, score
from sft_common.pretuned import ensure_local_model
from sft_common.runs import Run, existing_run
from sft_common.s3_sync import merged_name, merged_store

MODELS = ("pretuned", "finetuned")   # evaluated in this order; the table shows them side by side


# ---- inputs -------------------------------------------------------------------------------

def load_samples(val_file: Path, per_source: int, seed: int) -> list[dict]:
    """Up to `per_source` random val rows from each source (same rows for both models)."""
    by_source: dict[str, list] = {}
    with open(val_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_source.setdefault(row["source"], []).append(row)
    rng, out = random.Random(seed), []
    for src in sorted(by_source):
        rows = by_source[src]
        rng.shuffle(rows)
        out += rows[:per_source]
    return out


def model_dir(cfg: dict, run: Run, which: str) -> Path:
    """Local folder of the model to evaluate, downloading it from S3 if it is not here."""
    if which == "pretuned":
        return ensure_local_model(cfg)
    local = run.merged_dir.parent / merged_name(run.name, None)       # merged_models/<run>_final on S3
    if not (local / "config.json").exists():
        n = merged_store(cfg["storage"]["s3_root"], run.name, None).download_dir("", local)
        if not n:
            raise SystemExit(f"No merged model for run {run.name}: run 03_merging/merge_model.py first.")
    return local


# ---- one model ----------------------------------------------------------------------------

def generate(path: Path, rows: list[dict], cfg: dict) -> list[dict]:
    """Greedy generation for every row's prompt. Returns [{"text", "finish_reason"}]."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    e = cfg["evaluation"]
    fmt = ChatFormat(AutoTokenizer.from_pretrained(path))      # the model's own chat template
    prompts, params = [], []
    for r in rows:
        prompts.append(fmt.prompt(pick_target(r)[0], thinking=r["source"] in e["thinking_sources"]))
        limit = e["max_new_tokens"].get(r["source"], e["max_new_tokens"]["default"])
        params.append(SamplingParams(temperature=0.0, max_tokens=limit))
    # Context is its own setting: it must hold prompt + answer for every val row, including rows
    # longer than the training limit (model.max_seq_length only decides which rows are trained on).
    llm = LLM(model=str(path), dtype="bfloat16", max_model_len=e["max_model_len"], trust_remote_code=True)
    outs = llm.generate(prompts, params)                        # one SamplingParams per prompt
    return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason} for o in outs]


def judge_rows(cfg: dict, rows: list[dict], gens: list[dict]) -> dict[int, dict]:
    """Judge verdicts for the rows whose source is in evaluation.judge.sources, by row index."""
    j = cfg["evaluation"]["judge"]
    idx = [i for i, r in enumerate(rows) if r["source"] in j["sources"]]
    items = []
    for i in idx:
        prompt, ref = pick_target(rows[i])
        items.append({"question": prompt[-1]["content"],        # last user turn (document + question)
                      "reference": ref["content"],
                      "answer": split_think(gens[i]["text"])[1]})   # judged on the answer, not the reasoning
    judge = CloudflareJudge.from_env(j["model"], j["max_tokens"])
    print(f"[judge] scoring {len(items)} answers with {j['model']}")
    return dict(zip(idx, judge_many(judge, items, j["workers"])))


def evaluate_one(cfg: dict, run: Run, which: str, n: int, use_judge: bool) -> None:
    rows = load_samples(resolve(cfg["evaluation"]["val_file"]), n, cfg["data"]["seed"])
    gens = generate(model_dir(cfg, run, which), rows, cfg)
    verdicts = judge_rows(cfg, rows, gens) if use_judge else {}

    per_source: dict[str, list] = {}
    out_dir = run.eval_dir / which
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for i, (row, g) in enumerate(zip(rows, gens)):
            s = score(row, g["text"], g["finish_reason"])
            v = verdicts.get(i)
            if v is not None:                                   # judge scores join the other metrics;
                s["judge_correctness"] = v.get("correctness")   # None (judge error) is skipped by aggregate
                s["judge_reasoning"] = v.get("reasoning")
                s["judge_failed"] = "error" in v
            per_source.setdefault(row["source"], []).append(s)
            f.write(json.dumps({"i": i, "source": row["source"], "reference": pick_target(row)[1]["content"],
                                "output": g["text"], "finish_reason": g["finish_reason"],
                                "scores": s, "judge": v}, ensure_ascii=False) + "\n")
    metrics = {"model": which, **{src: aggregate(v) for src, v in sorted(per_source.items())}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


# ---- comparison ---------------------------------------------------------------------------

def compare(run: Run) -> dict:
    """pre-tuned vs fine-tuned, per source and metric, with the difference."""
    m = {w: json.loads((run.eval_dir / w / "metrics.json").read_text()) for w in MODELS}
    table = {}
    for src in sorted(set(m["pretuned"]) & set(m["finetuned"]) - {"model"}):
        table[src] = {}
        for k in sorted(set(m["pretuned"][src]) | set(m["finetuned"][src])):
            a, b = m["pretuned"][src].get(k), m["finetuned"][src].get(k)
            table[src][k] = {"pretuned": a, "finetuned": b,
                             "delta": round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None}
    return table


def print_table(table: dict) -> None:
    print(f"\n{'source':13} {'metric':18} {'pretuned':>9} {'finetuned':>9} {'delta':>8}")
    for src, metrics in table.items():
        for k, v in metrics.items():
            fmt = lambda x: "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)  # noqa: E731
            print(f"{src:13} {k:18} {fmt(v['pretuned']):>9} {fmt(v['finetuned']):>9} {fmt(v['delta']):>8}")


# ---- main ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run to evaluate (default: the last run started on this machine)")
    ap.add_argument("--n", type=int, help="rows per source (default: evaluation.samples_per_source)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--only", choices=MODELS, help=argparse.SUPPRESS)  # internal: one model per subprocess
    a = ap.parse_args()

    cfg = load_config()
    run = existing_run(cfg, a.run)
    n = a.n or cfg["evaluation"]["samples_per_source"]
    use_judge = cfg["evaluation"]["judge"]["enabled"] and not a.no_judge

    if a.only:                                    # subprocess: evaluate a single model and exit
        evaluate_one(cfg, run, a.only, n, use_judge)
        return

    if use_judge:                                 # fail now, not after an hour of generation
        CloudflareJudge.from_env(cfg["evaluation"]["judge"]["model"])
    for which in MODELS:
        cmd = [sys.executable, __file__, "--only", which, "--run", run.name, "--n", str(n)]
        subprocess.run(cmd + (["--no-judge"] if not use_judge else []), check=True)

    table = compare(run)
    (run.eval_dir / "comparison.json").write_text(json.dumps(table, indent=2))
    print_table(table)
    ok = run.store.upload_dir(run.eval_dir, "eval")
    print(f"\n[eval] {run.eval_dir} -> {run.store.uri('eval')}: {'uploaded' if ok else 'UPLOAD FAILED'}")


if __name__ == "__main__":
    main()
