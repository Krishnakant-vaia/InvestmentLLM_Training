"""Step 3: merge a run's LoRA adapter into its base model and save a full 16-bit model.

Run:  python 03_merging/merge_model.py --run NAME                   # the final adapter (after training)
      python 03_merging/merge_model.py --run NAME --checkpoint 3000   # a mid-run checkpoint, from S3
The base is the pre-tuned model for an SFT run, and the merged SFT model (dpo.base) for a DPO run (dpo-...):
a DPO adapter was trained on top of the SFT weights and only makes sense added to them.

Output (Hugging Face format: BF16 .safetensors shards + config + tokenizer + chat template):
    final adapter:  outputs/merged/<run>_final/          -> <storage.s3_root>/merged_models/<run>_final/
    checkpoint N:   outputs/merged/<run>_checkpoint-N/   -> <storage.s3_root>/merged_models/<run>_checkpoint-N/
Each folder also holds merge_info.json (what was merged, from where) and, on S3, _COMPLETE, written
last: a merged folder without it is an unfinished upload.

Barrier with training: a checkpoint is only READ (adapter files only; optimizer state is skipped),
and uploads are refused anywhere outside merged_models/ (RunStore.upload_model), so nothing is ever
written into the run's folder. The merged model needs no adapter code at inference: it is what
vLLM / Modal serve.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
# Unsloth must be imported before transformers (pulled in by sft_common) to patch it.
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402

from sft_common.adapter import point_adapter_at_base  # noqa: E402
from sft_common.config import load_config, resolve  # noqa: E402
from sft_common.pretuned import ensure_local_model, fix_generation_config  # noqa: E402
from sft_common.runs import existing_run, is_dpo_run, stage_cfg  # noqa: E402
from sft_common.s3_fetch import fetch_model  # noqa: E402
from sft_common.s3_sync import is_adapter_file, merged_name, merged_store  # noqa: E402

SANITY_PROMPT = "In two sentences, what is a stock dividend?"


def fetch_adapter(run, checkpoint: int | None) -> Path:
    """Local folder with the adapter to merge; downloaded from S3 when missing."""
    if checkpoint is None:
        local, rel = run.adapter_dir, "final_adapter"
    else:
        if checkpoint not in run.store.checkpoint_steps():       # only fully uploaded checkpoints
            raise SystemExit(f"checkpoint-{checkpoint} is not complete on {run.store.uri('checkpoints')}")
        local, rel = run.train_dir / f"checkpoint-{checkpoint}", f"checkpoints/checkpoint-{checkpoint}"
    if not (local / "adapter_config.json").exists():
        n = run.store.download_dir(rel, local, keep=is_adapter_file)
        if not n:
            raise SystemExit(f"No adapter for run {run.name} at {run.store.uri(rel)}")
        print(f"[merge] downloaded adapter ({n} files) from {run.store.uri(rel)}")
    return local


def eval_loss_at(adapter_dir: Path, step: int | None) -> float | None:
    """Val loss logged at `step` (from the checkpoint's trainer_state.json), if there is one."""
    f = adapter_dir / "trainer_state.json"
    if step is None or not f.exists():
        return None
    hist = json.loads(f.read_text(encoding="utf-8")).get("log_history", [])
    return next((h["eval_loss"] for h in hist if h.get("step") == step and "eval_loss" in h), None)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sanity_check(model_dir: Path) -> str:
    """Load the SAVED merged model with plain transformers (as vLLM/Modal will) and answer one
    short question. Raises if it cannot load or answers nothing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    msgs = [{"role": "user", "content": SANITY_PROMPT}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:                                  # template without the thinking switch
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=120, do_sample=False)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del model
    if not text:
        raise SystemExit("[merge] sanity check FAILED: the merged model produced no text")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run to merge (default: the last run started on this machine)")
    ap.add_argument("--checkpoint", type=int, help="merge this checkpoint step (from S3) instead of the final adapter")
    ap.add_argument("--skip-sanity", action="store_true", help="do not load + test the merged model before uploading")
    a = ap.parse_args()

    cfg = load_config()
    cfg = stage_cfg(cfg, a.run)                  # a DPO run's folders live under dpo.s3_root
    run = existing_run(cfg, a.run)
    target = merged_store(cfg["storage"]["s3_root"], run.name, a.checkpoint)   # merged_models/<run>_checkpoint-N
    out_dir = run.merged_dir.parent / merged_name(run.name, a.checkpoint)
    started = time.time()

    adapter_dir = fetch_adapter(run, a.checkpoint)
    # The adapter records the local path of its base model; re-point it at this machine's copy
    # of that model (downloaded from S3 if missing) so the merge works anywhere.
    if is_dpo_run(run.name):
        base_uri = cfg["dpo"]["base"]["s3_uri"]
        base_dir = fetch_model(base_uri, resolve(cfg["dpo"]["base"]["local_dir"]))
    else:
        base_uri = cfg["model"]["pretuned"]["s3_prefix"]
        base_dir = ensure_local_model(cfg)
    point_adapter_at_base(adapter_dir, base_dir)

    # Loading an adapter folder loads the base model and attaches the LoRA weights to it.
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(adapter_dir), max_seq_length=cfg["model"]["max_seq_length"],
        dtype=torch.bfloat16, load_in_4bit=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # merged_16bit: W_new = W + B x A for every adapted layer, saved as plain BF16 weights.
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    fix_generation_config(out_dir)  # stop generating at <|im_end|> too
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[merge] merged model saved to {out_dir}")

    answer = None
    if not a.skip_sanity:
        answer = sanity_check(out_dir)
        print(f"[merge] sanity check OK. Q: {SANITY_PROMPT}\n[merge] A: {answer}")

    weights = sorted(out_dir.glob("*.safetensors"))
    info = {
        "run": run.name,
        "source": "final_adapter" if a.checkpoint is None else f"checkpoints/checkpoint-{a.checkpoint}",
        "checkpoint_step": a.checkpoint,
        "adapter_uri": run.store.uri("final_adapter" if a.checkpoint is None else f"checkpoints/checkpoint-{a.checkpoint}"),
        "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
        "base_model_uri": base_uri,
        "val_loss_at_checkpoint": eval_loss_at(adapter_dir, a.checkpoint),
        "dtype": "bfloat16",
        "format": "huggingface safetensors",
        "weight_files": [w.name for w in weights],
        "weights_GB": round(sum(w.stat().st_size for w in weights) / 1e9, 2),
        "merged_uri": target.uri(),
        "sanity_prompt": SANITY_PROMPT if answer else None,
        "sanity_answer": answer,
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "merge_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    ok = target.upload_model(out_dir)                                # _COMPLETE is written last
    print(f"[merge] -> {target.uri()}: {'uploaded' if ok else 'UPLOAD FAILED (rerun to retry)'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
