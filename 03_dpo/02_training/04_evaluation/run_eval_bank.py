"""Answer every question of the VAIA eval bank with a merged model; write the answers as JSONL.

Run (on an eval GPU sandbox: 06_daytona/daytona_run.py eval ...):
    python 04_evaluation/run_eval_bank.py --workbook /root/eval_input/eval_bank.xlsx \
        --run sft-2026-09-29_19-35 --checkpoint 3000 [--overlap]

Model: <storage.s3_root>/merged_models/<run>_checkpoint-N/ (downloaded if not here), served by vLLM.
Output: outputs/eval_bank/eval_bank_responses.jsonl, one line per question:
    QA_ID, category, sub_capability, question_type, answer_format, difficulty, model_name,
    prompt (exactly what the model saw), model_response (final answer), thinking, prompt_tokens,
    output_tokens, truncated, train_overlap (with --overlap)
and, with --overlap, outputs/eval_bank/train_overlap.json. Both are uploaded to
merged_models/<name>/eval/ on S3 (next to the model, never into the training run).

Barrier: only Context, Question, Options and Answer Format reach the model (eval_bank.question_prompt);
gold answers and rubrics stay in the workbook for the judge. The workbook is never copied into
the project, the training data or S3.
"""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

# The sandbox image has no CUDA compiler (nvcc), so vLLM must not pick kernels that are compiled on
# first use (FlashInfer): use its prebuilt / Triton ones instead. Set before vLLM is imported.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.chat_format import ChatFormat, split_think  # noqa: E402
from sft_common.config import ROOT, load_config, resolve  # noqa: E402
from sft_common.eval_bank import barrier_violations, load_bank, question_prompt, train_overlap  # noqa: E402
from sft_common.runs import is_dpo_run, stage_cfg  # noqa: E402
from sft_common.s3_sync import RunStore, merged_name, merged_store  # noqa: E402

OUT_DIR = ROOT / "outputs" / "eval_bank"


def ensure_model(cfg: dict, name: str, store) -> Path:
    local = resolve(cfg["merging"]["merged_dir"]) / name
    if not (local / "config.json").exists():
        n = store.download_dir("", local, keep=lambda k: not k.startswith(("logs/", "eval/")))   # model files only
        if not n or not (local / "config.json").exists():
            raise SystemExit(f"no merged model at {store.uri()}")
        print(f"[eval] downloaded model ({n} files) from {store.uri()}")
    return local


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", required=True, help="eval bank .xlsx")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", type=int, help="merged checkpoint (default: the final merged model)")
    ap.add_argument("--max-new-tokens", type=int, default=8192, help="reasoning + answer budget per question")
    ap.add_argument("--no-thinking", action="store_true", help="answer directly, without reasoning first")
    ap.add_argument("--overlap", action="store_true", help="also check every question against the run's training data (train.jsonl; a DPO run: dpo_pairs.jsonl)")
    ap.add_argument("--limit", type=int, help="only the first N questions (quick test)")
    ap.add_argument("--tag", default="", help="name for this eval bank: outputs are eval_bank_responses_<tag>.jsonl (keeps earlier banks)")
    a = ap.parse_args()

    cfg = stage_cfg(load_config(), a.run)                # a DPO run's models live under dpo.s3_root
    bad = barrier_violations(ROOT)                       # the barrier: nothing from training may be on this machine
    if bad:
        raise SystemExit(f"[barrier] training material found on the eval machine, stopping: {bad[:5]}")
    print("[barrier] no training data, checkpoints, optimizer state or adapters on this machine; "
          + ("the training data is fetched for the overlap check after this point and deleted right after it"
             if a.overlap else "nothing from training will be downloaded"), flush=True)
    name = merged_name(a.run, a.checkpoint)
    store = merged_store(cfg["storage"]["s3_root"], a.run, a.checkpoint)
    rows = load_bank(a.workbook)[: a.limit]
    print(f"[eval] {len(rows)} questions from {Path(a.workbook).name}; model {name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""

    overlap = {}
    if a.overlap:
        data_file = "dpo_pairs.jsonl" if is_dpo_run(a.run) else "train.jsonl"
        train = OUT_DIR / data_file
        if not train.exists():
            RunStore(cfg["storage"]["s3_root"], a.run).download_dir("data", OUT_DIR, keep=lambda f: f == data_file)
        t0 = time.time()
        overlap = train_overlap(rows, train)
        flagged = [q for q, v in overlap.items() if v["flagged"]]
        (OUT_DIR / f"train_overlap{suffix}.json").write_text(json.dumps(
            {"n_questions": len(rows), "flagged": flagged, "per_question": overlap}, indent=1))
        print(f"[overlap] {len(flagged)} of {len(rows)} questions overlap {data_file} "
              f"(>= half their 13-word phrases found) in {time.time() - t0:.0f}s: {flagged[:20]}")
        store.upload_file(OUT_DIR / f"train_overlap{suffix}.json", f"eval/train_overlap{suffix}.json")   # kept even if generation fails
        train.unlink()                                   # 3 GB, not needed any more

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    path = ensure_model(cfg, name, store)
    fmt = ChatFormat(AutoTokenizer.from_pretrained(path))
    prompts = [fmt.prompt([{"role": "user", "content": question_prompt(r)}], thinking=not a.no_thinking)
               for r in rows]
    extra = {}
    try:                                                 # Qwen3.5 linear-attention prefill: Triton, not JIT FlashInfer
        from vllm.engine.arg_utils import EngineArgs
        if "gdn_prefill_backend" in inspect.signature(EngineArgs).parameters:
            extra["gdn_prefill_backend"] = "triton"
    except Exception:  # noqa: BLE001 - older/newer vLLM without this option
        pass
    llm = LLM(model=str(path), dtype="bfloat16", max_model_len=cfg["evaluation"]["max_model_len"],
              trust_remote_code=True, **extra)
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens))  # greedy
    print(f"[eval] generated {len(outs)} answers in {(time.time() - t0) / 60:.1f} min")

    out = OUT_DIR / f"eval_bank_responses{suffix}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r, p, o in zip(rows, prompts, outs):
            g = o.outputs[0]
            thinking, answer = split_think(g.text) if not a.no_thinking else ("", g.text)
            f.write(json.dumps({
                "QA_ID": r["QA_ID"], "category": r["Category"], "sub_capability": r.get("Sub-Capability", ""),
                "question_type": r.get("Question Type", ""), "answer_format": r.get("Answer Format", ""),
                "difficulty": r.get("Difficulty", ""), "domain": r.get("Domain", ""), "model_name": name,
                "prompt": question_prompt(r), "model_response": answer.strip(), "thinking": thinking.strip(),
                "prompt_tokens": len(o.prompt_token_ids), "output_tokens": len(g.token_ids),
                "truncated": g.finish_reason == "length",
                **({"train_overlap": overlap[r["QA_ID"]]["fraction"]} if overlap else {}),
            }, ensure_ascii=False) + "\n")
    cut = sum(o.outputs[0].finish_reason == "length" for o in outs)
    print(f"[eval] wrote {out} ({cut} answers hit the {a.max_new_tokens}-token limit)")

    ok = store.upload_file(out, f"eval/{out.name}")
    print(f"[eval] -> {store.uri('eval/' + out.name)}: {'uploaded' if ok else 'UPLOAD FAILED'}")


if __name__ == "__main__":
    main()
