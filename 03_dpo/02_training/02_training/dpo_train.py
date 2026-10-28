"""Step 6: DPO (LoRA) on the SFT-merged model, from preference pairs {prompt, chosen, rejected}.

Run:
    python 02_training/dpo_train.py --check                  # first, on a new GPU: reference + memory checks, no training
    python 02_training/dpo_train.py --max-steps 10           # smoke test: 10 real steps, 2 checkpoints to S3, then stop
    python 02_training/dpo_train.py                          # new run, named dpo-YYYY-MM-DD_HH-MM
    python 02_training/dpo_train.py --resume --run NAME      # continue a run (checkpoint from local or S3)

What happens, in order:
  1. download the merged SFT model (dpo.base) and the pairs file (dpo.data) from S3;
  2. validate every pair, render it with the model's chat template (as SFT did), drop bad / too-long pairs whole,
     and split train/val by prompt (sft_common/dpo_data.py);
  3. load the SFT model in BF16 and attach a NEW LoRA adapter. The frozen reference is the same model with the
     adapter switched off, so only one copy of the 9B weights is in memory;
  4. group pairs into micro-batches by token budget, prove none is lost, save the grouping; on --resume prove it is unchanged;
  5. before training (always; --check stops here): prove the reference really is the adapter-free model, and run the
     two heaviest micro-batches once to catch out-of-memory now rather than hours in;
  6. train: dpo.grad_accum_steps micro-batches per step; val per source every dpo.eval_steps, first at step 0
     (the baseline); every checkpoint goes to S3 at once; save the final adapter.
Everything goes to dpo.s3_root/<run_name>/. Needs a CUDA GPU with Unsloth installed (requirements/training.txt).
The loss is computed here (sft_common/dpo_loss.py), not by TRL's DPOTrainer: see README_DPO.md for why.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
# Unsloth must be imported before transformers (also pulled in by sft_common) to patch it.
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import Dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

from sft_common.batching import batch_plan, batch_stats, heaviest_batches, make_sampler, optimizer_steps, verify_coverage  # noqa: E402
from sft_common.benchmark import smoke_schedule  # noqa: E402
from sft_common.callbacks import S3SyncCallback  # noqa: E402
from sft_common.config import load_config, resolve  # noqa: E402
from sft_common.dpo_data import collate_pairs, prepare, read_pairs, side_tokens  # noqa: E402
from sft_common.dpo_loss import completion_logps, dpo_loss  # noqa: E402
from sft_common.kernels import require_fast_kernels  # noqa: E402
from sft_common.lora import adapters_per_target, unmatched  # noqa: E402
from sft_common.runs import DPO_PREFIX, Run, dpo_view  # noqa: E402
from sft_common.s3_fetch import fetch_file, fetch_model  # noqa: E402
from pairs.shape import shape  # noqa: E402
from pairs.source import load_pairs  # noqa: E402
from sft_common.tracking import init_wandb, make_run_name, resolve_report_to  # noqa: E402

SFT_TOKENS_PER_S = 6000        # measured SFT speed on one H100; DPO does ~4/3 of the work per token (extra reference pass)
FALL_WARN = 0.15               # warn when the chosen answers' log-prob has fallen this much (share of its start value)


# ---- model --------------------------------------------------------------------------------

def load_policy(cfg: dict, base_dir: Path):
    """The merged SFT model in BF16 + a fresh LoRA adapter (the only trainable weights)."""
    lora = cfg["dpo"]["lora"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_dir), max_seq_length=cfg["model"]["max_seq_length"] + 16,
        dtype=torch.bfloat16, load_in_4bit=False,        # BF16 like SFT: a 4-bit copy would not be the model SFT produced
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], use_rslora=lora["use_rslora"], bias="none",
        use_gradient_checkpointing="unsloth", random_state=cfg["data"]["seed"],
    )
    counts = adapters_per_target((n for n, _ in model.named_modules()), lora["target_modules"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] adapters per target: {counts} | trainable parameters: {trainable / 1e6:.0f}M")
    if unmatched(counts):
        print(f"[lora] WARNING: no layer matched {unmatched(counts)}: those names do not exist in this model")
    return model, tokenizer


# ---- trainer ------------------------------------------------------------------------------

class DPOBudgetTrainer(Trainer):
    """Trainer whose micro-batches are pairs grouped up to a token budget, with the DPO loss.

    Each micro-batch is 2B sequences (B chosen, then B rejected). The policy pass has gradients; the reference
    pass is the same weights with the LoRA adapter switched off, under no_grad. Both only compute log-probs of
    the answer tokens (sft_common/dpo_loss.py), never the full vocabulary scores.
    """

    def __init__(self, *args, d: dict, pad_id: int, **kwargs):
        self.d, self.pad_id = d, pad_id
        self._stats = {"train": defaultdict(list), "eval": defaultdict(list)}   # metrics of the micro-batches since the last log
        self._first_chosen: dict[str, float] = {}                               # per source: chosen log-prob at the first eval
        super().__init__(*args, **kwargs)
        # The loss below is already a mean over the micro-batch's pairs. With this False the Trainer divides it by
        # the accumulation steps itself (True would assume a token-count normalisation we do not do: a 8x too big loss).
        self.model_accepts_loss_kwargs = False

    # -- the loss
    def _logps(self, model, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        inner = self.accelerator.unwrap_model(model)
        inner = inner.get_base_model() if hasattr(inner, "get_base_model") else inner   # the PEFT-wrapped LM
        hidden = inner.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        return completion_logps(hidden, inner.lm_head.weight, inputs["labels"], self.d["loss_chunk"])

    def reference_logps(self, model, inputs: dict) -> torch.Tensor:
        """Log-probs under the frozen reference: the same model with the adapter off, no gradients."""
        with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
            return self._logps(model, inputs)[0]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        n = inputs["input_ids"].shape[0] // 2
        pol, pol_tokens = self._logps(model, inputs)
        ref = self.reference_logps(model, inputs)
        loss, metrics = dpo_loss(pol[:n], pol[n:], ref[:n], ref[n:], self.d["beta"], self.d["label_smoothing"])
        if self.d["sft_alpha"]:                              # RPO: also raise the chosen answers' own likelihood
            loss = loss + self.d["sft_alpha"] * (-(pol[:n] / pol_tokens[:n].clamp(min=1))).mean()
        for k, v in metrics.items():
            self._stats["train" if model.training else "eval"][k].append(v)
        return (loss, {"loss": loss.detach()}) if return_outputs else loss

    # -- batching: pairs grouped by token budget, never cut
    def sampler_for(self, ds, shuffle: bool):
        lengths = [len(p) + max(len(c), len(r)) for p, c, r in zip(ds["prompt_ids"], ds["chosen_ids"], ds["rejected_ids"])]
        return make_sampler("padded", lengths, max_tokens=self.d["tokens_per_micro_batch"] // 2,   # per side: both count
                            max_rows=self.d["max_rows_per_micro_batch"], seed=self.args.seed, shuffle=shuffle)

    def _loader(self, ds, shuffle: bool):
        loader = DataLoader(ds, batch_sampler=self.sampler_for(ds, shuffle), collate_fn=partial(collate_pairs, pad_id=self.pad_id),
                            num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)
        return self.accelerator.prepare(loader)

    def get_train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def get_eval_dataloader(self, eval_dataset=None):
        if isinstance(eval_dataset, str):                       # the Trainer evaluates a dict of sources one name at a time
            eval_dataset = self.eval_dataset[eval_dataset]
        return self._loader(eval_dataset if eval_dataset is not None else self.eval_dataset, False)

    # -- logging: average the stored per-micro-batch metrics into each log
    def log(self, logs, *args, **kwargs):
        evals = [k for k in logs if k.startswith("eval_") and k.endswith("_loss")]    # eval_<source>_loss
        mode, prefix = ("eval", evals[0][:-4]) if evals else ("train", "")
        for k, v in self._stats[mode].items():
            logs[prefix + k] = sum(v) / len(v)
        self._stats[mode].clear()
        if mode == "eval":
            self._print_eval(prefix[5:-1], prefix, logs)
        super().log(logs, *args, **kwargs)

    def _print_eval(self, src: str, prefix: str, logs: dict) -> None:
        g = lambda k: logs.get(prefix + k, float("nan"))  # noqa: E731
        start = self._first_chosen.setdefault(src, g("logps/chosen"))
        print(f"[dpo eval] step {self.state.global_step} {src}: loss {logs[prefix + 'loss']:.4f} | reward accuracy {g('rewards/accuracies'):.1%} | "
              f"margin {g('rewards/margins'):.3f} | chosen logp {g('logps/chosen'):.1f} (start {start:.1f}) | "
              f"reference already prefers chosen on {g('ref/prefers_chosen'):.1%}", flush=True)
        fallen = (start - g("logps/chosen")) / max(abs(start), 1.0)
        if fallen > FALL_WARN and not self.d["sft_alpha"]:
            print(f"[dpo eval] WARNING {src}: the chosen answers' log-prob fell {fallen:.0%} from the start: the model is unlearning "
                  f"the good answers while pushing the bad ones down. Set dpo.sft_alpha (e.g. 0.1) or lower the learning rate.", flush=True)


# ---- checks -------------------------------------------------------------------------------

def to_device(batch: dict, model) -> dict:
    return {k: v.to(model.device) for k, v in batch.items()}


def reference_check(trainer: DPOBudgetTrainer) -> bool:
    """Prove the reference is the adapter-free model: nudging the adapter must change the policy's log-probs but not
    the reference's. Without adapter switching working, DPO would compare the model with itself (loss stuck at 0.693)."""
    ds = trainer.train_dataset
    batches = trainer.sampler_for(ds, True).batches()
    idx = next((b for b in batches if len(b) >= 2), batches[0])
    batch = to_device(collate_pairs([ds[i] for i in idx], trainer.pad_id), trainer.model)
    model = trainer.model
    params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    saved = [p.detach().clone() for p in params]
    model.train()
    with torch.no_grad():
        before = trainer.reference_logps(model, batch)
        for p in params:
            p.add_(torch.randn_like(p) * 0.05)
        policy = trainer._logps(model, batch)[0]
        after = trainer.reference_logps(model, batch)
        for p, s in zip(params, saved):
            p.copy_(s)
    moved, stable = (policy - before).abs().max().item(), (after - before).abs().max().item()
    ok = stable <= 0.05 and moved > 20 * max(stable, 1e-3)
    print(f"[reference check] {len(idx)} real pairs: adapter nudged -> policy log-probs moved by up to {moved:.2f}, "
          f"reference log-probs by {stable:.4f} -> {'OK: the reference is the frozen SFT model' if ok else 'FAILED'}", flush=True)
    return ok


def memory_check(trainer: DPOBudgetTrainer) -> None:
    """One forward + backward (policy and reference) on the two heaviest micro-batches; print peak GPU memory."""
    ds, model = trainer.train_dataset, trainer.model
    lengths = [len(p) + max(len(c), len(r)) for p, c, r in zip(ds["prompt_ids"], ds["chosen_ids"], ds["rejected_ids"])]
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    model.train()
    for name, idx in heaviest_batches(lengths, trainer.sampler_for(ds, True).batches(), "padded").items():
        batch = to_device(collate_pairs([ds[i] for i in idx], trainer.pad_id), model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trainer.compute_loss(model, batch).backward()
        model.zero_grad(set_to_none=True)
        print(f"[memory] {name}: {len(idx)} pairs, 2 x {max(lengths[i] for i in idx)} tokens -> "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} of {total:.0f} GB", flush=True)


def plan_and_check(trainer: DPOBudgetTrainer, plan_file: Path, previous: str | None) -> list[str]:
    """Prove no pair is lost or repeated, save the grouping, and (on resume) prove it equals the original one."""
    d, ds = trainer.d, trainer.train_dataset
    lengths = [len(p) + max(len(c), len(r)) for p, c, r in zip(ds["prompt_ids"], ds["chosen_ids"], ds["rejected_ids"])]
    batches = trainer.sampler_for(ds, True).batches()
    try:
        verify_coverage(len(lengths), batches)
    except ValueError as e:
        raise SystemExit(f"[integrity] {e}")
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_file, "w", encoding="utf-8") as f:
        for rec in batch_plan(batches, lengths, ds["line"], ds["source"]):
            f.write(json.dumps(rec) + "\n")
    if previous and previous != hashlib.sha256(plan_file.read_bytes()).hexdigest():
        raise SystemExit("[integrity] the grouping differs from this run's original batch plan (pairs file or config changed): "
                         "resuming would skip the wrong pairs. Restore the original data/config, or start a new run.")
    s = batch_stats(lengths, batches, "padded", d["tokens_per_micro_batch"] // 2)
    steps = optimizer_steps(s["micro_batches"], d["grad_accum_steps"], d["epochs"])
    processed = sum(len(b) * max(lengths[i] for i in b) for b in batches) * 2          # both sides, padding included
    hours = processed * d["epochs"] * 4 / 3 / SFT_TOKENS_PER_S / 3600
    return [f"[integrity] {len(lengths)} pairs, each in exactly one micro-batch, none cut; plan: {plan_file}",
            f"[batching] budget {d['tokens_per_micro_batch']} tokens per micro-batch (both sides), max {d['max_rows_per_micro_batch']} pairs",
            f"    micro-batches={s['micro_batches']} avg pairs={s['avg_rows_per_micro_batch']} fill={s['avg_fill_pct']}% "
            f"padding={s['padding_pct']}% single-pair micro-batches={s['single_row_micro_batches']}",
            f"    optimizer steps={steps} ({d['grad_accum_steps']} micro-batches per step, {d['epochs']} epoch(s)); "
            f"rough ETA {hours:.1f} h (from SFT's measured speed, +33% for the reference pass; not measured for DPO)"]


# ---- S3: run inputs and resume ------------------------------------------------------------

def save_run_inputs(cfg: dict, run: Run, pairs_file: Path, prepared) -> None:
    """Upload what this run was trained with: config, pip freeze, the pairs file, and exactly which lines went where."""
    cfg_dir = run.train_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))     # config only; .env is never copied
    try:
        (cfg_dir / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    except OSError:
        pass
    (cfg_dir / "split.json").write_text(json.dumps({"train_lines": sorted(p["line"] for p in prepared.train),
                                                    "val_lines": {s: sorted(p["line"] for p in v) for s, v in prepared.val.items()}}))
    run.store.upload_dir(cfg_dir, "config")
    run.store.upload_file(pairs_file, "data/dpo_pairs.jsonl")
    run.store.upload_dir(run.logs_dir, "logs")
    print(f"[s3] run inputs saved to {run.store.uri()}")


def find_checkpoint(run: Run) -> str | None:
    """Newest checkpoint of this run, local or on S3 (downloaded if S3 is newer). None if none."""
    local = {int(p.name[11:]): p for p in run.train_dir.glob("checkpoint-*") if p.name[11:].isdigit() and (p / "trainer_state.json").exists()}
    remote = run.store.checkpoint_steps()
    best_local, best_remote = max(local, default=None), max(remote, default=None)
    if best_remote is not None and (best_local is None or best_remote > best_local):
        dest = run.train_dir / f"checkpoint-{best_remote}"
        print(f"[resume] downloading checkpoint-{best_remote} from S3")
        run.store.download_dir(f"checkpoints/checkpoint-{best_remote}", dest)
        if not (run.logs_dir / "batch_plan.jsonl").exists():
            run.store.download_dir("logs", run.logs_dir)
        return str(dest)
    return str(local[best_local]) if best_local is not None else None


# ---- main ---------------------------------------------------------------------------------

def to_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_list(pairs)


def build_args(cfg: dict, run: Run, max_steps: int | None, has_val: bool) -> TrainingArguments:
    d = cfg["dpo"]
    kwargs = dict(
        output_dir=str(run.train_dir), run_name=run.name,
        num_train_epochs=d["epochs"],
        per_device_train_batch_size=1, per_device_eval_batch_size=1,    # placeholders: the batch sampler decides micro-batches
        gradient_accumulation_steps=d["grad_accum_steps"],
        learning_rate=d["learning_rate"], warmup_ratio=d["warmup_ratio"], weight_decay=d["weight_decay"],
        lr_scheduler_type=d["lr_scheduler"], max_grad_norm=d["max_grad_norm"],
        bf16=True, optim=d["optimizer"],
        logging_steps=d["logging_steps"],
        eval_strategy="steps" if has_val else "no", eval_steps=d["eval_steps"], eval_on_start=has_val,
        prediction_loss_only=True,                                      # evals only need the loss; do not collect logits
        save_strategy="steps", save_steps=d["save_steps"], save_total_limit=d["save_total_limit"],
        seed=cfg["data"]["seed"], report_to=resolve_report_to(dpo_view(cfg)),
        remove_unused_columns=False,                                    # our columns are not model inputs; the collator uses them
        dataloader_num_workers=2, dataloader_pin_memory=True,
    )
    if max_steps:
        kwargs.update(smoke_schedule(max_steps))
    return TrainingArguments(**kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help=f"run name ({DPO_PREFIX}...): required with --resume")
    ap.add_argument("--resume", action="store_true", help="continue --run from its newest checkpoint")
    ap.add_argument("--check", action="store_true", help="reference + memory checks on the real data, then exit (nothing saved to S3)")
    ap.add_argument("--max-steps", type=int, metavar="N", help="smoke test: N optimizer steps (saving and evaluating twice), then stop")
    a = ap.parse_args()

    cfg = load_config()
    d, view = cfg["dpo"], dpo_view(cfg)
    if a.resume and not a.run:
        raise SystemExit("--resume needs --run <run name>")
    if a.run and not a.run.startswith(DPO_PREFIX):
        raise SystemExit(f"DPO run names start with {DPO_PREFIX!r} (the merge and Daytona steps rely on it)")
    if d["max_pair_tokens"] > d["tokens_per_micro_batch"] // 2:
        raise SystemExit("dpo.max_pair_tokens must fit twice into dpo.tokens_per_micro_batch (one pair = chosen + rejected)")
    pairs_file = resolve(d["data"]["local_path"])
    d["data"]["local_path"] = str(pairs_file)
    rows, source_report = load_pairs(d["data"])          # downloads the file and refuses it if it does not match its manifest
    rows, shape_report = shape(rows, d["data"].get("shape") or {})
    print("\n".join(source_report + shape_report))
    require_fast_kernels()

    run = Run(view, "checks" if a.check else (a.run or make_run_name("dpo")))
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run.name} | local {run.train_dir} | S3 {run.store.uri()}")

    # Data first (CPU, seconds): bad pairs should fail here, not after the 18 GB model has loaded.
    base_dir = fetch_model(d["base"]["s3_uri"], resolve(d["base"]["local_dir"]))
    prepared = prepare(rows, AutoTokenizer.from_pretrained(base_dir), d["data"] | {"max_pair_tokens": d["max_pair_tokens"]},
                       cfg["data"]["seed"])
    report = source_report + shape_report + list(prepared.report)
    print("\n".join(report))
    if not prepared.train:
        raise SystemExit("[dpo data] no usable training pairs")

    model, tokenizer = load_policy(cfg, base_dir)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    train_ds, val = to_dataset(prepared.train), {s: to_dataset(v) for s, v in prepared.val.items()}
    s3_sync = S3SyncCallback(run)
    trainer = DPOBudgetTrainer(
        model=model, args=build_args(cfg, run, a.max_steps, bool(val)), train_dataset=train_ds, eval_dataset=val or None,
        processing_class=tokenizer, callbacks=[] if a.check else [s3_sync],
        d=d, pad_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )

    plan_file, previous = run.logs_dir / "batch_plan.jsonl", None
    if a.resume:
        if not plan_file.exists():
            run.store.download_dir("logs", run.logs_dir)
        previous = hashlib.sha256(plan_file.read_bytes()).hexdigest() if plan_file.exists() else None
    report += plan_and_check(trainer, plan_file, previous)
    print("\n".join(report[len(prepared.report):]))
    (run.logs_dir / "startup_report.txt").write_text("\n".join(report) + "\n")

    if not reference_check(trainer):
        raise SystemExit("The reference is not the frozen model (adapter switching has no effect): DPO would be meaningless.")
    memory_check(trainer)                        # the heaviest micro-batches now: an out-of-memory error should stop the run here, not hours in
    if a.check:
        return

    resume_from = find_checkpoint(run) if a.resume else None
    if a.resume and resume_from is None:
        print("[resume] no checkpoint found for this run; starting from the beginning")
    if not resume_from:
        save_run_inputs(cfg, run, pairs_file, prepared)
    init_wandb(view, run.name)
    trainer.train(resume_from_checkpoint=resume_from)

    model.save_pretrained(run.adapter_dir)
    tokenizer.save_pretrained(run.adapter_dir)
    s3_sync.sync_all(trainer.state)
    ok = run.store.upload_dir(run.adapter_dir, "final_adapter")
    print(f"[done] adapter {run.adapter_dir} -> {run.store.uri('final_adapter')}: {'uploaded' if ok else 'UPLOAD FAILED'}")
    print(f"[next] python 03_merging/merge_model.py --run {run.name}")


if __name__ == "__main__":
    main()
