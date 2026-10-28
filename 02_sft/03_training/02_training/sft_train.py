"""Step 2: SFT (LoRA) on top of the pre-tuned Qwen3.5-9B, using the step 1 train/val files.

Run:
    python 02_training/sft_train.py --memory-check        # first, on a new GPU: peak memory, no training
    python 02_training/sft_train.py --check-packing       # packed mode only: do rows stay apart? no training
    python 02_training/sft_train.py --benchmark 1,2,4,8   # speed + peak memory per micro-batch size, ETA; no training
    python 02_training/sft_train.py --max-steps 20        # smoke test: 20 real steps, 2 checkpoints to S3, then stop
    python 02_training/sft_train.py                       # new run, named sft-YYYY-MM-DD_HH-MM
    python 02_training/sft_train.py --resume              # continue the last run (checkpoint from local or S3)
    python 02_training/sft_train.py --resume --run NAME   # continue a specific run, e.g. on another machine

What happens, in order:
  1. load the pre-tuned model (downloaded from S3 on first use) and attach fresh LoRA adapters;
  2. load train/val from step 1; rows over model.max_seq_length (12,000 tokens) are dropped whole,
     never cut: ~255 tool-calling rows, listed per source in the startup report;
  3. mask the loss to assistant tokens only (system/user turns are context, not targets);
  4. group rows into micro-batches of up to training.tokens_per_micro_batch tokens, in
     training.batching_mode "packed" or "padded" (sft_common/batching.py);
  5. prove no QA pair is lost, duplicated or cut, and save the exact grouping (batch plan);
     on --resume, also prove the grouping is identical to the run's original one;
  6. packed mode only: check packed rows can't see each other (else stop before training);
  7. save the run's inputs to S3, then train: training.grad_accum_steps micro-batches per
     optimizer step; every checkpoint is uploaded to S3 as soon as it is saved;
  8. save the final LoRA adapter locally and to S3.
Everything goes to storage.s3_root/<run_name>/ (sft_common/s3_sync.py). Loss curves go to wandb
when WANDB_API_KEY is set in .env (sft_common/tracking.py).
Needs a CUDA GPU with Unsloth installed (requirements/training.txt).
"""
import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
# Unsloth must be imported before transformers/trl (also pulled in by sft_common) to patch them.
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from sft_common.batching import (batch_plan, batch_stats, heaviest_batches, length_report,  # noqa: E402
                                 make_sampler, optimizer_steps, verify_coverage)
from sft_common.benchmark import (eta_hours, format_table, pick_batches, recommend,  # noqa: E402
                                  scaled_budget, smoke_schedule, suggest_accum)
from sft_common.callbacks import FitAnalysisCallback, S3SyncCallback  # noqa: E402
from sft_common.chat_format import ASSISTANT_PART, USER_PART  # noqa: E402
from sft_common.config import load_config, resolve  # noqa: E402
from sft_common.kernels import require_fast_kernels  # noqa: E402
from sft_common.lean_loss import lean_loss  # noqa: E402
from sft_common.lora import adapters_per_target, unmatched  # noqa: E402
from sft_common.pretuned import ensure_local_model  # noqa: E402
from sft_common.runs import Run, existing_run, new_run  # noqa: E402
from sft_common.tracking import init_wandb, resolve_report_to  # noqa: E402

# TRL may append an end-of-text token when it tokenizes, so the trainer's (and Unsloth's) length limit
# gets a little room above model.max_seq_length. Without it a row of exactly max_seq_length tokens
# (12,000 is now near the tail of the tool-calling rows) would silently lose its last token.
TRAINER_SLACK = 16


# ---- model --------------------------------------------------------------------------------

def load_model(cfg: dict, max_seq_length: int | None = None):
    """Pre-tuned full weights (downloaded from S3 on first use) + fresh LoRA adapters.

    Only the LoRA matrices are trained; the 9B base weights stay frozen (bias="none": no biases).
    Qwen3.5 is multimodal; we train the text stack only.
    `max_seq_length` overrides model.max_seq_length for this load only: the benchmark packs up to
    256k tokens into one sequence, longer than the per-row limit used when dropping rows.
    """
    m, lora = cfg["model"], cfg["training"]["lora"]
    start = ensure_local_model(cfg)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(start),
        max_seq_length=max_seq_length or m["max_seq_length"] + TRAINER_SLACK,
        dtype=torch.bfloat16,
        load_in_4bit=m["load_in_4bit"],
        # Force PyTorch's own memory-efficient attention kernel. Without an explicit choice,
        # Unsloth falls back to eager attention when flash-attn-2 is not installed, which
        # materializes a full seq x seq score matrix per layer - tens of GB for one long row.
        # sdpa needs no extra install and is efficient on Hopper/Ampere GPUs.
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],                                  # rank of each LoRA matrix pair
        lora_alpha=lora["alpha"],                     # scale of the LoRA update
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],        # which projections get adapters
        use_rslora=lora["use_rslora"],                # rank-stabilised scaling
        bias="none",
        use_gradient_checkpointing="unsloth",         # recompute activations to save memory
        random_state=cfg["data"]["seed"],
    )
    # A target name that matches nothing is skipped silently by PEFT; print what really got adapters.
    counts = adapters_per_target((n for n, _ in model.named_modules()), lora["target_modules"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] adapters per target: {counts} | trainable parameters: {trainable / 1e6:.0f}M")
    if unmatched(counts):
        print(f"[lora] WARNING: no layer matched {unmatched(counts)}: those names do not exist in this model")
    return model, tokenizer


# ---- data ---------------------------------------------------------------------------------

def load_data(cfg: dict, tokenizer) -> tuple:
    """Load train/val text from step 1, report token lengths per source, drop over-long rows.

    Returns (datasets, report_lines, train_rows). Rows longer than model.max_seq_length are
    dropped whole, never truncated (a truncated answer would teach the model to stop mid-sentence).
    `train_rows` maps each kept training row to its (line number in train.jsonl, source), so the
    batch plan can say exactly which QA pairs trained together.
    """
    d = resolve(cfg["data"]["out_dir"])
    ds = load_dataset("json", data_files={"train": str(d / "train.jsonl"), "val": str(d / "val.jsonl")})
    tok = getattr(tokenizer, "tokenizer", tokenizer)       # processors wrap the real tokenizer
    limit = cfg["model"]["max_seq_length"]

    def n_tokens(batch):
        return {"n_tokens": [len(ids) for ids in tok(batch["text"], add_special_tokens=False)["input_ids"]]}

    workers = cfg["training"].get("cpu_workers", 1)          # parallel CPU processes for tokenizing
    ds = ds.map(n_tokens, batched=True, num_proc=workers)
    # Kept training rows in dataset order (the filter below keeps order): (train.jsonl line, source).
    train_rows = [(line, src) for line, (n, src) in enumerate(zip(ds["train"]["n_tokens"], ds["train"]["source"]))
                  if n <= limit]
    lines = []
    for split in ds:
        lines.append(f"[{split}] rows per source (dropped = longer than {limit} tokens):")
        for src, r in sorted(length_report(ds[split]["source"], ds[split]["n_tokens"], limit).items()):
            lines.append(f"    {src:13} rows={r['rows']:>7} dropped={r['dropped']:>4} longest={r['max_tokens']}")
        # Keep only the rendered text; the trainer tokenizes it again itself.
        ds[split] = ds[split].filter(lambda r: r["n_tokens"] <= limit, num_proc=workers).select_columns(["text"])
    # The periodic val loss uses a fixed random subset of the val rows, so evals stay quick.
    n_eval = min(cfg["training"]["eval_rows"], len(ds["val"]))
    ds["val"] = ds["val"].shuffle(seed=cfg["data"]["seed"]).select(range(n_eval))
    return ds, lines, train_rows


# ---- trainer ------------------------------------------------------------------------------

def build_args(cfg: dict, run: Run, max_steps: int | None = None) -> SFTConfig:
    t = cfg["training"]
    kwargs = dict(
        output_dir=str(run.train_dir),               # checkpoints land in outputs/sft/<run>/checkpoint-N
        run_name=run.name,                           # also the wandb run name
        num_train_epochs=t["epochs"],
        # Micro-batches are built by the batching sampler, not by a fixed row count; the Trainer
        # still wants a number here (used only in its own logging), so 1 is a placeholder.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["grad_accum_steps"],   # micro-batches per optimizer step
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],              # share of steps spent ramping the LR up
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler"],         # cosine: smooth decay after warmup
        max_grad_norm=t["max_grad_norm"],            # clip update size: guards against loss spikes
        bf16=True,
        optim=t["optimizer"],                        # adamw_8bit: 8-bit optimizer state, less memory
        logging_steps=t["logging_steps"],            # how often train loss is logged
        eval_strategy="steps",
        eval_steps=t["eval_steps"],                  # how often val loss is computed
        save_strategy="steps",
        save_steps=t["save_steps"],                  # how often a checkpoint is saved (and sent to S3)
        save_total_limit=t["save_total_limit"],      # local copies kept; S3 keeps every checkpoint
        seed=cfg["data"]["seed"],
        report_to=resolve_report_to(cfg),            # "wandb" if configured, else "none"
        dataset_text_field="text",
        dataset_num_proc=t.get("cpu_workers", 1),   # TRL's own tokenization pass runs in parallel too
        dataloader_num_workers=t.get("cpu_workers", 1),   # batches are prepared in background processes
        dataloader_pin_memory=True,
        packing=False,                               # TRL's own packing off: our batching (packed/padded) groups rows
    )
    # TRL renamed max_seq_length -> max_length in newer releases.
    name = "max_length" if "max_length" in inspect.signature(SFTConfig).parameters else "max_seq_length"
    kwargs[name] = cfg["model"]["max_seq_length"] + TRAINER_SLACK
    if t.get("keep_best"):                         # keep the best val-loss checkpoint among the saved ones
        if t["eval_steps"] != t["save_steps"]:
            raise SystemExit("[config] keep_best needs training.eval_steps == training.save_steps")
        kwargs.update(load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False)
    if max_steps:                                  # smoke run: a handful of steps, saving/evaluating twice
        kwargs.update(smoke_schedule(max_steps))
    return SFTConfig(**kwargs)


class BudgetSFTTrainer(SFTTrainer):
    """SFTTrainer whose train/eval loaders fill each micro-batch up to a token budget.

    padded mode: the usual collator pads the rows of a micro-batch to the longest one.
    packed mode: DataCollatorWithFlattening lays the rows end to end in one sequence, restarts
    position ids at 0 for every row, and masks the first label of each row so no row is trained
    to predict the next row's first token.
    """

    def __init__(self, *args, batching_mode: str, tokens_per_micro_batch: int, max_rows_per_micro_batch: int,
                 lean_loss: bool = False, **kwargs):
        self.lean_loss = lean_loss
        self.batching_mode = batching_mode
        self.tokens_per_micro_batch = tokens_per_micro_batch
        self.max_rows_per_micro_batch = max_rows_per_micro_batch
        self._packed_collator = None
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """The training/eval loss. With training.lean_loss the vocabulary layer only sees the answer tokens
        (sft_common/lean_loss.py); otherwise the standard loss. Normalisation follows the Trainer's own rule:
        divide by the whole accumulation window's target count when the Trainer passes it, else average
        over this micro-batch (the Trainer then divides by the accumulation steps)."""
        if not self.lean_loss:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        inner = self.accelerator.unwrap_model(model)
        inner = inner.get_base_model() if hasattr(inner, "get_base_model") else inner   # the PEFT-wrapped LM
        hidden = inner.model(**{k: v for k, v in inputs.items() if k != "labels"}).last_hidden_state
        items = num_items_in_batch if getattr(self, "model_accepts_loss_kwargs", False) else None
        loss, _ = lean_loss(hidden, inner.lm_head.weight, inputs["labels"], num_items=items)
        return (loss, {"loss": loss.detach()}) if return_outputs else loss

    def sampler_for(self, ds, shuffle: bool):
        # Lengths come from the tokenized dataset itself, so indices always line up.
        return make_sampler(self.batching_mode, [len(ids) for ids in ds["input_ids"]],
                            max_tokens=self.tokens_per_micro_batch, max_rows=self.max_rows_per_micro_batch,
                            seed=self.args.seed, shuffle=shuffle)

    def collator(self):
        """The collator for the current mode (padded: the trainer's own; packed: flattening)."""
        if self.batching_mode != "packed":
            return self.data_collator
        if self._packed_collator is None:
            from transformers import DataCollatorWithFlattening
            # separator_id=-100: the first label of every row is masked (no cross-row target);
            # flash-attn kwargs carry the row boundaries to attention kernels that use them.
            self._packed_collator = DataCollatorWithFlattening(
                return_position_ids=True, separator_id=-100, return_flash_attn_kwargs=True)
        return self._packed_collator

    def _budget_loader(self, ds, shuffle: bool, description: str):
        if self.batching_mode == "packed" and "labels" not in ds.column_names:
            # The flattening collator needs the assistant-only labels already in the dataset
            # (train_on_responses_only adds them); without them it would train on every token.
            raise SystemExit("packed mode needs a 'labels' column; use training.batching_mode: padded")
        sampler = self.sampler_for(ds, shuffle)
        ds = self._remove_unused_columns(ds, description=description)   # keep only model inputs
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=self.collator(),
                            num_workers=self.args.dataloader_num_workers,
                            pin_memory=self.args.dataloader_pin_memory)
        return self.accelerator.prepare(loader)

    def get_train_dataloader(self):
        return self._budget_loader(self.train_dataset, True, "training")

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        return self._budget_loader(ds, False, "evaluation")                # fixed order for eval


def _digest(path: Path) -> str | None:
    """Fingerprint of a batch plan file (None if it does not exist)."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def check_and_plan(trainer: BudgetSFTTrainer, cfg: dict, train_rows: list[tuple[int, str]],
                   plan_file: Path, previous_plan: str | None = None) -> list[str]:
    """Before training: prove no QA pair is lost, duplicated or cut; save the batch plan; summarise.

    Stops the run (SystemExit) if
      * the trainer holds a different number of rows than we kept (a row was lost/added),
      * any row reaches the trainer's length limit (max_seq_length + TRAINER_SLACK) after its own
        tokenization (it would have been cut),
      * the micro-batches miss or repeat any row,
      * resuming (`previous_plan` = the run's original plan fingerprint) and the grouping differs:
        the Trainer skips the micro-batches already trained by position, so a different grouping
        (e.g. train.jsonl rebuilt, or config changed) would skip the wrong QA pairs.
    The batch plan (one line per micro-batch: train.jsonl line numbers, sources, token counts)
    goes to logs/batch_plan.jsonl, uploaded to S3 with the other logs.
    """
    lengths = [len(ids) for ids in trainer.train_dataset["input_ids"]]
    t, limit = cfg["training"], cfg["model"]["max_seq_length"] + TRAINER_SLACK
    if len(lengths) != len(train_rows):
        raise SystemExit(f"[integrity] trainer has {len(lengths)} rows but {len(train_rows)} were kept")
    cut = sum(n >= limit for n in lengths)
    if cut:
        raise SystemExit(f"[integrity] {cut} rows reach the trainer's length limit ({limit}) after tokenization: they would be cut")
    batches = trainer.sampler_for(trainer.train_dataset, True).batches()
    try:
        verify_coverage(len(lengths), batches)
    except ValueError as e:
        raise SystemExit(f"[integrity] {e}")

    plan_file.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_file, "w", encoding="utf-8") as f:
        for rec in batch_plan(batches, lengths, [r for r, _ in train_rows], [s for _, s in train_rows]):
            f.write(json.dumps(rec) + "\n")
    if previous_plan and previous_plan != _digest(plan_file):
        raise SystemExit("[integrity] the grouping differs from this run's original batch plan "
                         "(train data or batching config changed): resuming would skip the wrong rows. "
                         "Restore the original data/config, or start a new run.")

    s = batch_stats(lengths, batches, trainer.batching_mode, t["tokens_per_micro_batch"])
    steps = optimizer_steps(s["micro_batches"], t["grad_accum_steps"], t["epochs"])
    return [
        f"[integrity] {len(lengths)} rows, each in exactly one micro-batch, none cut; plan: {plan_file}",
        f"[batching] mode={trainer.batching_mode}, budget {t['tokens_per_micro_batch']} tokens "
        f"and max {t['max_rows_per_micro_batch']} rows per micro-batch",
        f"    micro-batches={s['micro_batches']} avg rows={s['avg_rows_per_micro_batch']} "
        f"max rows={s['max_rows_in_a_micro_batch']} max tokens={s['max_tokens_in_a_micro_batch']}",
        f"    avg fill={s['avg_fill_pct']}% of budget, padding={s['padding_pct']}% of GPU work, "
        f"single-row micro-batches={s['single_row_micro_batches']}",
        f"    optimizer steps={steps} ({t['grad_accum_steps']} micro-batches per step, {t['epochs']} epoch(s))",
    ]


def _loss_sum(model, batch: dict) -> tuple[float, int]:
    """(summed token loss, number of target tokens) for one batch. The model's loss is a mean
    over the target tokens; a causal LM predicts labels[1:], so those are the ones counted."""
    batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
    n = int((batch["labels"][:, 1:] != -100).sum())
    return float(model(**batch).loss) * n, n


def check_packing(trainer: BudgetSFTTrainer, tolerance: float) -> bool:
    """Packed mode is only correct if rows packed together cannot see each other.

    Takes the packed micro-batch with the longest row (a long row plus the short rows filling
    it), computes the loss of each row on its own, then of the whole packed micro-batch, and
    compares the totals. Equal (within `tolerance`, bf16 noise) = rows stay apart; different =
    information leaks between rows (e.g. a linear-attention state carried across the boundary).
    """
    ds = trainer._remove_unused_columns(trainer.train_dataset, description="packing check")
    lengths = [len(ids) for ids in ds["input_ids"]]
    bins = [b for b in trainer.sampler_for(trainer.train_dataset, True).batches() if len(b) >= 2]
    if not bins:
        print("[packing] no micro-batch holds two rows; nothing to check")
        return True
    rows = max(bins, key=lambda b: max(lengths[i] for i in b))
    model = trainer.model
    model.eval()
    with torch.no_grad():
        one_by_one = [_loss_sum(model, trainer.data_collator([ds[i]])) for i in rows]   # padded collator, 1 row
        packed = _loss_sum(model, trainer.collator()([ds[i] for i in rows]))            # flattening collator
    model.train()
    separate = sum(s for s, _ in one_by_one)
    rel = abs(packed[0] - separate) / max(abs(separate), 1e-9)
    print(f"[packing] {len(rows)} rows ({', '.join(str(lengths[i]) for i in rows[:6])}{'...' if len(rows) > 6 else ''} tokens)")
    print(f"    target tokens: one-by-one={sum(n for _, n in one_by_one)} packed={packed[1]}")
    print(f"    summed loss:   one-by-one={separate:.2f} packed={packed[0]:.2f} difference={rel:.2%} "
          f"(tolerance {tolerance:.2%}) -> {'OK: rows stay apart' if rel <= tolerance else 'LEAK: rows see each other'}")
    return rel <= tolerance


def loss_agreement_check(trainer: BudgetSFTTrainer, tolerance: float) -> bool:
    """The lean loss must equal the standard loss on a real micro-batch. If it picked the wrong hidden
    states (say, before the final normalisation) it would train on a wrong loss without any error."""
    ds = trainer._remove_unused_columns(trainer.train_dataset, description="loss check")
    batches = trainer.sampler_for(trainer.train_dataset, True).batches()
    idx = next((b for b in batches if len(b) >= 2), batches[0])               # a real batch with several rows
    collated = trainer.collator()([ds[i] for i in idx])
    batch = {k: v.to(trainer.model.device) if hasattr(v, "to") else v for k, v in collated.items()}
    with torch.no_grad():
        standard = float(trainer.model(**batch).loss)
        lean = float(trainer.compute_loss(trainer.model, batch))
    rel = abs(lean - standard) / max(abs(standard), 1e-9)
    print(f"[loss check] real batch of {len(idx)} rows: standard loss {standard:.4f}, lean loss {lean:.4f}, "
          f"difference {rel:.3%} (tolerance {tolerance:.1%}) -> {'OK' if rel <= tolerance else 'MISMATCH'}", flush=True)
    return rel <= tolerance


def memory_check(trainer: BudgetSFTTrainer) -> None:
    """One forward+backward on the two heaviest micro-batches; print peak GPU memory. No training.

    Excludes optimizer state, which is small for LoRA (well under 1 GB with 8-bit Adam).
    """
    ds = trainer._remove_unused_columns(trainer.train_dataset, description="memory check")
    lengths = [len(ids) for ids in ds["input_ids"]]
    batches = trainer.sampler_for(trainer.train_dataset, True).batches()
    model, total = trainer.model, torch.cuda.get_device_properties(0).total_memory / 1e9
    model.train()
    for name, idx in heaviest_batches(lengths, batches, trainer.batching_mode).items():
        collated = trainer.collator()([ds[i] for i in idx])
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in collated.items()}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model(**batch).loss.backward()
        model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[memory] {name}: {len(idx)} rows, {sum(lengths[i] for i in idx)} tokens "
              f"(longest {max(lengths[i] for i in idx)}) -> peak {peak:.1f} of {total:.0f} GB")


def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def benchmark(trainer: BudgetSFTTrainer, cfg: dict, sizes: list[int], warmup: int = 3, timed: int = 10) -> list[dict]:
    """Speed, peak GPU memory and ETA for micro-batch sizes N = 1, 2, 4, 8 ... (N x the token budget).

    For each size: regroup the real training rows with the bigger budget, then run real forward +
    backward passes (and optimizer steps, with learning rate 0 so the weights do not change) on
    `warmup` untimed micro-batches, the hardest first so peak memory is hit early, then `timed`
    timed ones. Tokens/s counts real tokens only (padding is wasted work, so it lowers speed).
    Out of memory, or any other failure, is reported and stops the larger sizes (they would fail
    too). Nothing is trained or saved.
    """
    import gc

    import bitsandbytes as bnb

    t = cfg["training"]
    ds = trainer._remove_unused_columns(trainer.train_dataset, description="benchmark")
    lengths = [len(ids) for ids in ds["input_ids"]]
    total_tokens = sum(lengths) * t["epochs"]
    model, gpu_gb = trainer.model, torch.cuda.get_device_properties(0).total_memory / 1e9
    base = (trainer.tokens_per_micro_batch, trainer.max_rows_per_micro_batch)
    print(f"[benchmark] {torch.cuda.get_device_name(0)}, {gpu_gb:.0f} GB; {total_tokens:,} tokens to train "
          f"({t['epochs']} epoch(s)); sizes {sizes}; loss: {'lean' if trainer.lean_loss else 'standard'}")
    print(f"[benchmark] GPU memory before the first size: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated "
          f"(weights + LoRA)", flush=True)

    results, blocked = [], None
    for n in sorted(sizes):
        budget, max_rows = scaled_budget(*base, n)
        accum = suggest_accum(t["grad_accum_steps"], n)
        res = {"size": n, "budget": budget, "max_rows": max_rows, "accum": accum}
        if blocked:
            results.append({**res, "status": f"skipped ({blocked})"})
            continue
        trainer.tokens_per_micro_batch, trainer.max_rows_per_micro_batch = budget, max_rows
        optimizer = batch = collated = None
        k, idx, picks = -1, [], []                  # which micro-batch was running when/if it fails
        try:
            batches = trainer.sampler_for(trainer.train_dataset, True).batches()
            picks = pick_batches(lengths, batches, warmup + timed, trainer.batching_mode, cfg["data"]["seed"])
            n_warm = min(warmup, max(len(picks) - 1, 0))
            # lr=0: real optimizer memory and time, but the weights stay exactly as they were.
            optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=0.0)
            model.train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            seconds, tokens = 0.0, 0
            for k, idx in enumerate(picks):
                collated = trainer.collator()([ds[i] for i in idx])
                batch = {kk: v.to(model.device) if hasattr(v, "to") else v for kk, v in collated.items()}
                torch.cuda.synchronize()
                started = time.perf_counter()
                (trainer.compute_loss(model, batch) / accum).backward()      # the loss training really uses
                if (k + 1) % accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if k >= n_warm:
                    seconds += time.perf_counter() - started
                    tokens += sum(lengths[i] for i in idx)
                print(f"    size {n}: micro-batch {k + 1}/{len(picks)} ({len(idx)} rows, "
                      f"{sum(lengths[i] for i in idx)} tokens){' [warm-up]' if k < n_warm else ''}", flush=True)
            peak = torch.cuda.max_memory_reserved() / 1e9          # what the allocator held: the number that OOMs
            tps = tokens / seconds if seconds > 0 else None
            res.update(status="ok", micro_batches=len(batches), peak_gb=round(peak, 1),
                       peak_pct=round(100 * peak / gpu_gb, 1), tokens_per_s=round(tps) if tps else None,
                       eta_hours=round(eta_hours(total_tokens, tps), 2) if tps else None)
        except Exception as e:  # noqa: BLE001 - OOM and Unsloth/attention errors are results, not crashes
            oom = _is_oom(e)
            alloc, reserved = torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9
            detail = (f"OOM at micro-batch {k + 1}/{len(picks)} "
                      f"({len(idx)} rows, {sum(lengths[i] for i in idx) if idx else 0} tokens): "
                      f"allocated={alloc:.1f} GB reserved={reserved:.1f} GB of {gpu_gb:.0f} GB" if oom else
                      f"error: {type(e).__name__}: {str(e)[:300]}")
            print(f"    {detail}\n    raw: {str(e)[:500]}", flush=True)
            res["status"] = "OOM" if oom else f"error: {type(e).__name__}: {str(e)[:200]}"
            res["detail"] = detail
            blocked = f"size {n} " + ("ran out of memory" if oom else "failed")
        finally:                                    # leave the GPU clean for the next size
            del optimizer, batch, collated
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        results.append(res)
        print(f"[benchmark] size {n} ({budget} tokens): {res['status']}", flush=True)
    trainer.tokens_per_micro_batch, trainer.max_rows_per_micro_batch = base   # restore the configured budget
    return results


# ---- S3: run inputs and resume ------------------------------------------------------------

def save_run_inputs(cfg: dict, run: Run) -> None:
    """Upload what this run was trained with: config snapshot, pip freeze, and the data files."""
    cfg_dir = run.train_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))  # config only; .env is never copied
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (cfg_dir / "pip_freeze.txt").write_text(freeze)                        # exact library versions
    except OSError:
        pass
    run.store.upload_dir(cfg_dir, "config")
    data = resolve(cfg["data"]["out_dir"])
    files = ["stats.json", "val.jsonl"] + (["train.jsonl"] if cfg["storage"]["upload_train_jsonl"] else [])
    for f in files:
        if (data / f).exists():
            run.store.upload_file(data / f, f"data/{f}")
    run.store.upload_dir(run.logs_dir, "logs")        # startup report + batch plan, before training starts
    print(f"[s3] run inputs saved to {run.store.uri()}")


def find_resume_checkpoint(run: Run) -> str | None:
    """Newest checkpoint of this run, local or on S3 (downloaded if S3 is newer). None if none."""
    local = {int(p.name[11:]): p for p in run.train_dir.glob("checkpoint-*")
             if p.name[11:].isdigit() and (p / "trainer_state.json").exists()}
    remote = run.store.checkpoint_steps()
    best_local, best_remote = max(local, default=None), max(remote, default=None)
    if best_remote is not None and (best_local is None or best_remote > best_local):
        dest = run.train_dir / f"checkpoint-{best_remote}"
        print(f"[resume] downloading checkpoint-{best_remote} from S3")
        run.store.download_dir(f"checkpoints/checkpoint-{best_remote}", dest)
        if not (run.logs_dir / "fit_analysis.json").exists():
            run.store.download_dir("logs", run.logs_dir)   # keep the fit-analysis history continuous
        return str(dest)
    return str(local[best_local]) if best_local is not None else None


# ---- main ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true", help="continue a run from its newest checkpoint")
    ap.add_argument("--run", help="run name to resume (default: the last run started on this machine)")
    ap.add_argument("--memory-check", action="store_true",
                    help="run the two heaviest micro-batches once, print peak GPU memory, and exit")
    ap.add_argument("--check-packing", action="store_true",
                    help="packed mode: check that packed rows stay apart, print the result, and exit")
    ap.add_argument("--benchmark", metavar="SIZES",
                    help="e.g. 1,2,4,8: speed, peak GPU memory and ETA for micro-batches of N x the token "
                         "budget (12k, 24k, 48k, 96k tokens with the default budget), then exit. "
                         "No training, nothing saved to S3")
    ap.add_argument("--max-steps", type=int, metavar="N",
                    help="smoke test: train only N optimizer steps (saving and evaluating twice), then stop")
    ap.add_argument("--skip-packing-check", action="store_true",
                    help="packed mode: train without running the packing check first (not recommended)")
    a = ap.parse_args()

    require_fast_kernels()                       # stop now, not after an OOM, if Qwen3.5's fast kernels are missing
    cfg = load_config()
    t = cfg["training"]
    check_only = a.memory_check or a.check_packing or bool(a.benchmark)
    sizes = [int(x) for x in a.benchmark.split(",")] if a.benchmark else []
    if a.check_packing and t["batching_mode"] != "packed":
        raise SystemExit("--check-packing needs training.batching_mode: packed in config.yaml")
    if check_only:
        run = Run(cfg, "checks")                     # throwaway local folder; nothing goes to S3
    elif a.resume:
        run = existing_run(cfg, a.run)
    else:
        run = new_run(cfg, a.run)
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run.name} | local {run.train_dir} | S3 {run.store.uri()}")

    # Only PACKED mode puts up to max(sizes) x the budget into ONE sequence, so only then must the model accept
    # more than max_seq_length. Padded rows never exceed it. (The benchmark used to load with 96,000 here even in
    # padded mode: that run ran out of memory on a single 12k-token row, while the probe, which loads at the
    # normal 12,016, needs 25 GB for the same row.)
    longest = (min(max(sizes) * t["tokens_per_micro_batch"], 262144)
               if sizes and t["batching_mode"] == "packed" else None)
    model, tokenizer = load_model(cfg, max_seq_length=longest)
    ds, report, train_rows = load_data(cfg, tokenizer)
    args = build_args(cfg, run, a.max_steps)

    s3_sync = S3SyncCallback(run)
    trainer = BudgetSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=args,
        callbacks=[FitAnalysisCallback(run.logs_dir / "fit_analysis.json")]
                  + ([] if check_only else [s3_sync]),
        batching_mode=t["batching_mode"],
        tokens_per_micro_batch=t["tokens_per_micro_batch"],
        max_rows_per_micro_batch=t["max_rows_per_micro_batch"],
        lean_loss=t["lean_loss"],
    )
    # Mask everything except assistant turns: loss only on what the model should learn to write.
    trainer = train_on_responses_only(trainer, instruction_part=USER_PART, response_part=ASSISTANT_PART)

    plan_file = run.logs_dir / "batch_plan.jsonl"
    previous_plan = None
    if a.resume:
        if not plan_file.exists():                   # resuming on another machine: fetch the run's logs
            run.store.download_dir("logs", run.logs_dir)
        previous_plan = _digest(plan_file)           # the grouping this run started with
    report += check_and_plan(trainer, cfg, train_rows, plan_file, previous_plan)
    print("\n".join(report))
    (run.logs_dir / "startup_report.txt").write_text("\n".join(report) + "\n")

    if t["lean_loss"] and not loss_agreement_check(trainer, t["loss_check_tolerance"]):
        raise SystemExit("The lean loss differs from the standard loss on a real batch: it would train the wrong "
                         "thing. Set training.lean_loss: false in config.yaml (standard loss) and start again.")

    if a.memory_check:
        memory_check(trainer)
    if a.check_packing:
        check_packing(trainer, t["packing_check_tolerance"])
    if sizes:
        results = benchmark(trainer, cfg, sizes)
        print("\n" + format_table(results))
        best = recommend(results)
        if best:
            print(f"\n[benchmark] recommended: size {best['size']} -> in config.yaml set "
                  f"tokens_per_micro_batch: {best['budget']}, max_rows_per_micro_batch: {best['max_rows']}, "
                  f"grad_accum_steps: {best['accum']}  (peak {best['peak_gb']} GB = {best['peak_pct']:.0f}% of the GPU, "
                  f"ETA {best['eta_hours']} h)")
        else:
            print("\n[benchmark] no size fits with headroom (<= 90% of GPU memory)")
        (run.logs_dir / "benchmark.json").write_text(json.dumps(results, indent=2))
        print(f"[benchmark] saved {run.logs_dir / 'benchmark.json'}")
    if check_only:
        return

    # Packed mode trains only if packed rows are proven to stay apart (checked before any
    # upload, so a failed check leaves nothing on S3).
    if t["batching_mode"] == "packed" and not a.skip_packing_check:
        if not check_packing(trainer, t["packing_check_tolerance"]):
            raise SystemExit("Packed rows leak into each other on this model/setup. "
                             "Set training.batching_mode: padded in config.yaml and start again.")

    resume_from = find_resume_checkpoint(run) if a.resume else None
    if a.resume and resume_from is None:
        print("[resume] no checkpoint found for this run; starting from the beginning")
    if not resume_from:
        save_run_inputs(cfg, run)                    # first start of this run: record its inputs

    # wandb starts here (after the checks, so failed checks leave no empty runs); the Trainer's
    # wandb integration finds this run already open and logs into it.
    init_wandb(cfg, run.name)
    trainer.train(resume_from_checkpoint=resume_from)

    # After training: save the final adapter, then make sure S3 has everything.
    model.save_pretrained(run.adapter_dir)
    tokenizer.save_pretrained(run.adapter_dir)
    s3_sync.sync_all(trainer.state)                  # any checkpoint whose upload failed + logs
    ok = run.store.upload_dir(run.adapter_dir, "final_adapter")
    print(f"[done] adapter {run.adapter_dir} -> {run.store.uri('final_adapter')}: {'uploaded' if ok else 'UPLOAD FAILED'}")
    print(f"[next] python 03_merging/merge_model.py --run {run.name}")


if __name__ == "__main__":
    main()
