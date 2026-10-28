"""Trains Qwen/Qwen3.5-9B-Base on pretune_dump.py's raw-text output via
plain transformers+peft+bitsandbytes LoRA (no Unsloth), tracked in W&B.
Evaluates via perplexity on the held-out val split — completion-style
training, not Q&A/chat-template SFT.

Dropped Unsloth (2026-09-02) after two real crashes smoke-testing it on the
pod — torchao needing a torch.int1 dtype that didn't exist in the installed
torch, then transformers' bitsandbytes integration needing a newer
nn.Module.set_submodule API. Both were transformers/torch version-skew
issues, not Unsloth-specific bugs, but Unsloth's patching layer (its own log:
"Unsloth: Fast Qwen3_5 patching") adds extra surface area on top of that
skew. Verified before dropping it: Qwen3.5's hybrid gated-deltanet
architecture is natively merged into mainline transformers — AutoModelForCausalLM
loads it directly, no Unsloth-specific patch required.

Deliberately independent of pretune_dump.py and wandb_tracking/wandb_logger.py
— no imports from either. Duplicates the one shared constant it needs
(PACK_SEQ_LEN, which must match pretune_dump.py's packing length since this
script trusts its input is already packed and disables re-packing).
"""

import argparse
import math

# Qwen's own official repo, not the unsloth/ mirror -- no reason to depend on
# a third-party mirror once nothing depends on that mirror's own loading code.
BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
PROJECT_NAME = "vaia-financial-pretuning-training"
PACK_SEQ_LEN = 8192  # must match pretune_dump.py's PACK_SEQ_LEN
MAX_SEQ_LENGTH = PACK_SEQ_LEN
EPOCHS = 1
LEARNING_RATE = 1e-5
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
SEED = 3407

# A fixed step count, not warmup_ratio -- confirmed on the real pod 2026-09-03
# that the installed trl==1.12.0/transformers==5.16.1's SFTConfig no longer
# accepts warmup_ratio at all (TypeError: unexpected keyword argument),
# genuine version skew, not a guess. warmup_steps=59 is the same 3% figure
# warmup_ratio=0.03 was computing anyway, tied to the real corpus's known
# ~1,964 total steps for one epoch (31,423 packed sequences / effective
# batch 16) -- since it's already tied to a specific known corpus size, a
# fixed step count loses nothing versus the ratio.
WARMUP_STEPS = 59

# Where checkpoints/the final model are uploaded. RunPod pods are typically
# ephemeral -- a checkpoint that only exists on local disk is lost if the pod
# is terminated/reclaimed before it's copied out, so every checkpoint
# (not just the final model) gets uploaded here as training progresses.
S3_CHECKPOINT_PREFIX = "s3://ft-vaia/Corpus/Invest_dataset/09022026_pretune_results/"

# Real CFA/finance MCQ benchmark (19,433 questions, 30+ financial subdomains,
# Easy/Medium/Hard tiers, India-focused -- lines up with this project's own
# SEBI/RBI/NSE grounding). Gated on Hugging Face: the account running this
# script needs HF_TOKEN set to an account that has accepted the dataset's
# access terms on huggingface.co -- set by whoever runs this, never by Claude.
MCQ_DATASET_NAME = "bharatgenai/BhashaBench-Finance"
# Confirmed 2026-09-03 on the real pod: this dataset requires a config name
# (its two configs are "English"/"Hindi", 13,451/5,982 questions) -- "English"
# to match this project's other English-language training/eval data.
MCQ_DATASET_CONFIG = "English"
MCQ_DATASET_SPLIT = "test"  # confirmed 2026-09-03 against the real dataset

# A subsample, not the full ~12,100-question MCQ set -- reversed from the
# original "full set every time" design after real timing on the pod showed
# log-likelihood scoring (one model forward pass per option) costs ~1.4s per
# question, meaning the full set would cost ~4.7 HOURS per evaluation x 4
# evaluations =~19 extra hours on top of actual training time -- assumed
# "cheap" during design, before any real per-question timing existed. 300 is
# ~7 minutes per evaluation x 4 =~28 minutes total, small enough not to
# meaningfully extend the run, large enough to track a real trend across
# checkpoints rather than a couple dozen questions' worth of noise.
MCQ_EVAL_SAMPLE_SIZE = 300


def build_lora_config(**overrides) -> dict:
    """Returns the default LoRA hyperparameters, with any keyword overrides
    applied on top — same override mechanism as wandb_logger.build_wandb_config."""
    config = {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        # "all-linear", not an enumerated list: Qwen3.5's hybrid architecture
        # has two structurally different layer types with different
        # projection names (full-attention: q_proj/k_proj/v_proj/o_proj;
        # Gated DeltaNet linear attention, 75% of layers: in_proj_qkv/
        # in_proj_z/in_proj_b/in_proj_a/out_proj) -- confirmed against the
        # real modeling_qwen3_5.py source, not assumed. An enumerated list
        # silently under-covers whichever type isn't listed: PEFT doesn't
        # error, it just skips attaching LoRA there, badly undertraining
        # 75% of the model's sequence-mixing layers with no warning at all.
        # "all-linear" targets every nn.Linear automatically (excluding the
        # output head) -- doesn't require enumerating every projection name
        # across structurally different layer types, and isn't broken by a
        # third layer type or a renamed projection showing up later. Can't
        # adapt dt_bias/A_log (raw gating parameters, not nn.Linear) -- an
        # accepted, normal LoRA limitation, not something to work around.
        "target_modules": "all-linear",
        "bias": "none",
        "random_state": SEED,
    }
    config.update(overrides)
    return config


def build_training_config(output_dir: str, **overrides) -> dict:
    """Returns the default SFTConfig kwargs as a plain dict. packing=False is
    deliberate: pretune_dump.py already packs chunks to PACK_SEQ_LEN, so
    SFTTrainer must not re-pack (which would silently mangle sequence
    boundaries and the <|endoftext|> separators already in the data)."""
    config = {
        "output_dir": output_dir,
        "num_train_epochs": EPOCHS,
        # 4/4, not 2/8 -- raised 2026-09-03 after the real pod smoke test
        # showed only ~36.5GB/80GB GPU memory used at batch_size=2, and the
        # full run's real per-step timing projected to ~54 hours. Same
        # effective batch (4x4=16, same as 2x8) so this changes wall-clock
        # speed via fewer sequential microbatches per step, not what the
        # model actually sees per optimizer update.
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 10,
        # epoch-based checkpointing/eval with EPOCHS=1 meant exactly one save
        # and one eval, at the very end of the whole run (~1,964 total steps)
        # -- a pod interruption near the end would lose everything already
        # trained. Step-based at 500 gives ~4 checkpoints spaced through the
        # run, each with a matching eval_loss/eval_perplexity/eval_accuracy_mcq
        # data point, without paying checkpoint/upload overhead too often
        # (100 steps, ~20 checkpoints, was considered and rejected as too
        # frequent for this run's length).
        "save_strategy": "steps",
        "save_steps": 500,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "report_to": "wandb",
        "max_length": MAX_SEQ_LENGTH,
        "seed": SEED,
        "dataset_text_field": "text",
        "packing": False,
        # 8, not "as many CPUs as possible" -- each dataloader worker is a
        # full forked OS process (not a thread), so this doesn't scale the
        # way S3_FETCH_WORKERS did; going much higher adds process-spawn and
        # memory overhead without a real throughput benefit here, since the
        # data is already-packed plain text, cheap to batch.
        "dataloader_num_workers": 8,
        # prepare_model_for_kbit_training's use_gradient_checkpointing=True
        # alone doesn't make the Trainer actually engage it -- this flag is
        # also required, or checkpointing never activates during training.
        "gradient_checkpointing": True,
        # No warmup at all previously meant the freshly-initialized LoRA
        # adapter got hit with the full learning rate from step 1. cosine
        # decay + a short warmup (WARMUP_STEPS -- see its own comment for
        # why this is a fixed step count, not warmup_ratio) is the more
        # stable default.
        "lr_scheduler_type": "cosine",
        "warmup_steps": WARMUP_STEPS,
    }
    config.update(overrides)
    return config


def build_wandb_run_config(dataset_name: str, train_path: str, val_path: str) -> dict:
    """Returns the dict passed to wandb.init(config=...) for a training run."""
    return {
        "base_model": BASE_MODEL,
        "dataset_name": dataset_name,
        "train_path": train_path,
        "val_path": val_path,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    }


def perplexity_from_loss(mean_loss: float) -> float:
    """Perplexity is e^(average cross-entropy loss) — the standard way to
    turn a raw eval loss into an interpretable held-out-data metric for
    completion/raw-text training. Lower is better; 1.0 is a perfect model."""
    return math.exp(mean_loss)


def log_epoch_perplexity(wandb_module, eval_loss: float | None, step: int) -> None:
    """Converts one epoch's eval_loss into perplexity and logs it to wandb,
    tagged with the training step it came from — this is what gives a real
    per-epoch perplexity curve on the dashboard instead of a single final
    number logged only after training completes. eval_loss can be None if
    an evaluation call fires without that key in its metrics; nothing to
    convert in that case.

    step is logged as a plain data field ("trainer_step"), NOT passed as
    wandb.log's own step= kwarg. Confirmed on the real pod 2026-09-03: doing
    that silently DROPS the whole log call whenever HF Trainer's own
    concurrent wandb logging has already advanced wandb's internal monotonic
    step counter past this value by the time this callback fires (a real,
    reproducible collision, not just a smoke-test artifact of aggressive
    step intervals -- it can recur at production cadence too, since 500 is a
    multiple of logging_steps=10). Letting wandb use its own auto-increment
    trades exact x-axis step alignment for the metric never silently
    vanishing, which is the right tradeoff here."""
    if eval_loss is None:
        return
    wandb_module.log({"eval_perplexity": perplexity_from_loss(eval_loss), "trainer_step": step})


def build_perplexity_callback(wandb_module):
    """Thin wrapper (untested — needs the real transformers TrainerCallback
    base class) around log_epoch_perplexity: returns a TrainerCallback that
    fires on_evaluate every eval_steps=500 and logs that evaluation's
    perplexity."""
    from transformers import TrainerCallback

    class PerplexityLoggingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            log_epoch_perplexity(wandb_module, (metrics or {}).get("eval_loss"), state.global_step)
            return control

    return PerplexityLoggingCallback()


def score_mcq_question(log_probs_per_option: list[float]) -> int:
    """Returns the index of the highest-scoring option -- the model's
    "answer" under log-likelihood scoring. The real per-option log-probability
    computation is a GPU forward pass and lives separately in
    compute_option_log_likelihood; this is just the decision logic around it,
    kept separate specifically so it can be tested without a GPU."""
    return max(range(len(log_probs_per_option)), key=lambda i: log_probs_per_option[i])


def mcq_accuracy(predictions: list[int], gold_labels: list[int]) -> float:
    """Plain fraction-correct. Returns 0.0 on empty input rather than raising
    -- an eval run over zero questions has no accuracy to report, not an error."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold_labels) if p == g)
    return correct / len(predictions)


# Candidate field names, in priority order, for each of a MCQ dataset's needed
# columns. Confirmed against the REAL bharatgenai/BhashaBench-Finance schema
# (config="English") on the first live smoke test 2026-09-03 -- it was NOT
# what was guessed while writing this (a single "options" list field): the
# real columns are 'question', 'correct_answer' (a letter, e.g. "C"), and four
# SEPARATE columns 'option_a'/'option_b'/'option_c'/'option_d', plus
# 'question_type' showing the split mixes MCQ with other formats (fill-in-
# blank, match-the-column, etc. -- ~93% MCQ per the dataset's own stated
# breakdown) that don't have a clean log-likelihood-scorable option set and
# must be filtered out, not force-fit. Both the single-list-field shape
# (kept for flare-cfa-style datasets) and this real separate-columns shape
# are supported; a genuinely unrecognized schema still raises a clear error
# naming what it found, rather than guessing wrong silently -- the exact
# failure mode that cost 5,542 skipped documents in the SFT pipeline's
# capability-code mapping earlier in this project.
_MCQ_QUESTION_FIELD_CANDIDATES = ["question", "prompt", "query", "text"]
_MCQ_OPTIONS_FIELD_CANDIDATES = ["options", "choices", "answers"]
_MCQ_OPTION_COLUMN_CANDIDATES = ["option_a", "option_b", "option_c", "option_d", "option_e", "option_f"]
_MCQ_ANSWER_FIELD_CANDIDATES = ["answer", "gold", "correct_answer", "label"]
_MCQ_QUESTION_TYPE_FIELD = "question_type"  # confirmed real field name; only "MCQ" rows are scorable


def _resolve_answer_index(answer_value, options: list) -> int:
    """The answer field might already be an integer index (like flare-cfa's
    gold), a value found literally in the options list, or a bare letter
    (A/B/C/D) that needs resolving to a position -- BhashaBench-Finance's real
    correct_answer field is this last form (e.g. "C" -> index 2)."""
    if isinstance(answer_value, int):
        return answer_value
    if isinstance(answer_value, str):
        stripped = answer_value.strip()
        if stripped in options:
            return options.index(stripped)
        if len(stripped) == 1 and stripped.isalpha():
            return ord(stripped.upper()) - ord("A")
    raise ValueError(f"load_mcq_eval_set: could not resolve answer {answer_value!r} against options {options!r}")


def load_mcq_eval_set(raw_rows: list[dict]) -> list[dict]:
    """Normalizes rows from whichever of the candidate schemas is actually
    present into a consistent {"question": ..., "options": [...], "answer": int}
    shape. Raises ValueError naming the first row's actual keys if none of the
    candidate schemas match -- see the candidate-list comment above for why."""
    if not raw_rows:
        return []

    # Rows whose question_type isn't "MCQ" (fill-in-blank, match-the-column,
    # etc.) don't have a clean option set to score -- skipped, not errored.
    if _MCQ_QUESTION_TYPE_FIELD in raw_rows[0]:
        raw_rows = [r for r in raw_rows if str(r.get(_MCQ_QUESTION_TYPE_FIELD, "")).strip().upper() == "MCQ"]
    if not raw_rows:
        return []

    first = raw_rows[0]
    q_field = next((f for f in _MCQ_QUESTION_FIELD_CANDIDATES if f in first), None)
    o_field = next((f for f in _MCQ_OPTIONS_FIELD_CANDIDATES if f in first), None)
    option_columns = [c for c in _MCQ_OPTION_COLUMN_CANDIDATES if c in first]
    a_field = next((f for f in _MCQ_ANSWER_FIELD_CANDIDATES if f in first), None)
    if not (q_field and a_field and (o_field or option_columns)):
        raise ValueError(
            f"load_mcq_eval_set: none of the known field-name candidates matched this "
            f"dataset's actual columns {sorted(first.keys())}. Question candidates tried: "
            f"{_MCQ_QUESTION_FIELD_CANDIDATES}, options-as-list: {_MCQ_OPTIONS_FIELD_CANDIDATES}, "
            f"options-as-columns: {_MCQ_OPTION_COLUMN_CANDIDATES}, answer: {_MCQ_ANSWER_FIELD_CANDIDATES}. "
            f"Inspect a real row and add the actual field name to the matching candidate list."
        )

    normalized = []
    for row in raw_rows:
        if o_field:
            options = row[o_field]
        else:
            options = [row[c] for c in option_columns if row.get(c) not in (None, "")]
        normalized.append({
            "question": row[q_field],
            "options": options,
            "answer": _resolve_answer_index(row[a_field], options),
        })
    return normalized


def sample_mcq_eval_set(mcq_dataset: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Deterministic subsample (same seed -> same questions every run, so
    accuracy across checkpoints is comparable, not drawn from a different
    random slice each time). Returns the whole set unchanged if it's already
    at or below sample_size -- nothing to subsample."""
    import random

    if len(mcq_dataset) <= sample_size:
        return mcq_dataset
    return random.Random(seed).sample(mcq_dataset, sample_size)


def compute_option_log_likelihood(model, tokenizer, question: str, option: str) -> float:
    """Sum of log-probabilities the model assigns to option's tokens,
    conditioned on question as a prefix -- the log-likelihood scoring
    approach: measures which continuation the model's own probabilities
    favor, without requiring any instruction-following ability. Chosen over
    free-text generation + answer parsing specifically because this is a
    continued-pretraining checkpoint, not instruction-tuned -- it can't be
    trusted to reliably output a clean, parseable "the answer is X" format,
    especially early in training.

    A real GPU forward pass -- untestable locally, unlike everything else in
    this module. Verified on the real pod's first smoke test."""
    import torch

    prompt_ids = tokenizer(question, return_tensors="pt").input_ids
    full_ids = tokenizer(question + option, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)

    with torch.no_grad():
        logits = model(full_ids).logits

    log_probs = torch.log_softmax(logits, dim=-1)
    option_start = prompt_ids.shape[1]
    total = 0.0
    for i in range(option_start, full_ids.shape[1]):
        token_id = full_ids[0, i]
        total += log_probs[0, i - 1, token_id].item()
    return total


def evaluate_mcq_accuracy(model, tokenizer, mcq_dataset: list[dict]) -> float:
    """Runs the full MCQ set (all of it, every time -- not a subsample; at
    only 4 evaluations total across the run, matching eval_steps=500, the
    full set's cost is paid just 4 times, worth it for a statistically
    reliable number instead of a noisier subsample) and returns overall
    accuracy. Temporarily switches the model to eval mode and back, since
    this fires mid-training via a callback."""
    was_training = model.training
    model.eval()
    predictions = []
    gold_labels = []
    for item in mcq_dataset:
        scores = [compute_option_log_likelihood(model, tokenizer, item["question"], opt)
                  for opt in item["options"]]
        predictions.append(score_mcq_question(scores))
        gold_labels.append(item["answer"])
    if was_training:
        model.train()
    return mcq_accuracy(predictions, gold_labels)


def build_mcq_accuracy_callback(wandb_module, model, tokenizer, mcq_dataset: list[dict]):
    """Thin wrapper (untested — needs the real transformers TrainerCallback
    base class) around evaluate_mcq_accuracy: hooks the same on_evaluate
    trigger build_perplexity_callback already uses (fires at eval_steps=500),
    so every checkpoint has a matching eval_accuracy_mcq data point alongside
    eval_perplexity from that same evaluation event.

    step logged as a plain data field, not wandb.log's step= kwarg -- see
    log_epoch_perplexity's docstring for why (a real, confirmed-on-the-real-
    pod bug where an explicit step= silently drops the whole log call)."""
    from transformers import TrainerCallback

    class MCQAccuracyCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            accuracy = evaluate_mcq_accuracy(model, tokenizer, mcq_dataset)
            wandb_module.log({"eval_accuracy_mcq": accuracy, "trainer_step": state.global_step})
            return control

    return MCQAccuracyCallback()


def upload_checkpoint_to_s3(local_dir: str, s3_prefix: str, s3_client) -> list[str]:
    """Uploads every file under local_dir to s3_prefix, preserving the
    directory's relative structure. Takes an injected s3_client for
    testability, same pattern as pretune_dump.py's upload_outputs_to_s3."""
    import os

    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    bucket, prefix = s3_prefix.replace("s3://", "").split("/", 1)
    uploaded = []
    for root, _dirs, files in os.walk(local_dir):
        for filename in sorted(files):
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            key = prefix + rel_path
            with open(local_path, "rb") as f:
                s3_client.put_object(Bucket=bucket, Key=key, Body=f.read())
            uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def build_checkpoint_upload_callback(s3_client, s3_prefix: str):
    """Thin wrapper (untested — needs the real transformers TrainerCallback
    base class) around upload_checkpoint_to_s3: returns a TrainerCallback
    whose on_save hook fires right after the Trainer saves a checkpoint
    (every save_steps=500) and uploads that checkpoint
    directory to S3 immediately — so training progress survives a pod
    interruption instead of only existing on local, possibly ephemeral,
    disk."""
    import os

    from transformers import TrainerCallback

    class CheckpointUploadCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(checkpoint_dir):
                upload_checkpoint_to_s3(
                    checkpoint_dir, f"{s3_prefix}checkpoint-{state.global_step}/", s3_client,
                )
            return control

    return CheckpointUploadCallback()


def load_pretune_dataset(train_path: str, val_path: str):
    """Loads pretune_dump.py's own output — already {"text": ...} JSONL,
    already packed to MAX_SEQ_LENGTH, no reformatting needed."""
    from datasets import load_dataset

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset = load_dataset("json", data_files=val_path, split="train")
    return train_dataset, val_dataset


def build_s3_client(boto3_module):
    """S3 client with real connect/read timeouts and retries -- a stuck
    connection (data queued for upload but never delivered/ACKed) was
    confirmed twice on real pods during this project's speedup work, and
    boto3's default client has NO read timeout at all, so a stuck socket
    would otherwise hang the whole training run indefinitely rather than
    fail and retry. connect_timeout/read_timeout force a stuck socket to
    actually error out instead of hanging forever; the retry config then
    retries that error automatically. Same pattern already established in
    pretune_dump.py's read client, with explicit timeouts added since those
    weren't part of that original pattern but are directly what this
    specific failure mode needs."""
    from botocore.config import Config as BotoConfig

    return boto3_module.client(
        "s3",
        config=BotoConfig(
            connect_timeout=30,
            read_timeout=120,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def load_fresh_model(use_gradient_checkpointing: bool = True):
    """Loads an unmodified base checkpoint with a fresh LoRA adapter applied
    — called once per training run so results are never contaminated by a
    previous run's learned weights. Plain transformers+peft+bitsandbytes, no
    Unsloth (see module docstring for why).

    use_gradient_checkpointing must match build_training_config's own
    "gradient_checkpointing" value -- prepare_model_for_kbit_training's flag
    here and the Trainer's own flag are two separate switches that both have
    to agree, or checkpointing either doesn't engage or engages
    inconsistently. Parameterized (default True, matching the production
    default) specifically to let this be tested against the real VRAM
    headroom before committing a hyperparameter change to the real run."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    torch.manual_seed(SEED)  # LoraConfig has no random_state kwarg (that was Unsloth-specific)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        # {"": 0}, not "auto" -- "auto" is designed for inference/multi-device
        # placement and can behave unpredictably for single-GPU training;
        # this pins everything to the one GPU the pod actually has.
        device_map={"": 0},
        # Governs the non-quantized layers (embeddings, layer norms) --
        # bnb_4bit_compute_dtype above only governs the quantized layers.
        # dtype, not torch_dtype -- confirmed on the real pod that the
        # installed transformers==5.16.1 deprecated torch_dtype in favor of
        # this (a warning, not a hard break, but real version skew worth
        # fixing rather than leaving a deprecation warning on every run).
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        # Not guaranteed to exist on Qwen/Llama-family tokenizers -- without
        # it, SFTTrainer's data collator can't batch variable-length
        # sequences. <|endoftext|> is this tokenizer's real eos_token,
        # verified against the actual tokenizer earlier in this project.
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    lora_kwargs = {k: v for k, v in build_lora_config().items() if k != "random_state"}
    model = get_peft_model(model, LoraConfig(**lora_kwargs, task_type="CAUSAL_LM"))
    return model, tokenizer


def evaluate_perplexity(trainer) -> float:
    """Runs the trainer's built-in evaluation loop against its eval_dataset
    (the held-out val split) and converts the resulting loss into
    perplexity."""
    metrics = trainer.evaluate()
    return perplexity_from_loss(metrics["eval_loss"])


# Training and validation token-level accuracy: NO custom code needed. A
# custom SFTTrainer subclass overriding compute_loss was built and tested
# for this, then deleted after the real pod smoke test 2026-09-03 showed
# trl==1.12.0/transformers==5.16.1 already compute and log both natively --
# `mean_token_accuracy` (training) and `eval_mean_token_accuracy`
# (validation) appear in W&B automatically via report_to="wandb", same as
# train/loss and eval/loss. The custom mechanism was silently accumulating
# nothing (compute_loss's returned logits were None, likely a memory-saving
# fused loss path in these versions) -- moot either way once the built-in
# metric was confirmed to already cover the exact same need, with less code
# and no unverified custom hook. Only eval_accuracy_mcq (below) remains
# custom, since nothing in the library computes that.


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5-9B-Base on pretune_dump.py's raw-text output.",
    )
    parser.add_argument("--train-jsonl", required=True, help="Path to a local copy of pretune_dump.py's train.jsonl")
    parser.add_argument("--val-jsonl", required=True, help="Path to a local copy of pretune_dump.py's val.jsonl")
    parser.add_argument("--dataset-name", default="pretune-v1", help="Label for this dataset variant, used in the W&B run name and config")
    parser.add_argument("--wandb-project", default=PROJECT_NAME)
    parser.add_argument("--output-dir", default="./outputs/pretune-training")
    parser.add_argument(
        "--s3-checkpoint-prefix", default=S3_CHECKPOINT_PREFIX,
        help="S3 prefix checkpoints and the final model are uploaded to as training progresses",
    )
    parser.add_argument("--mcq-dataset", default=MCQ_DATASET_NAME, help="Gated HF dataset for validation MCQ accuracy")
    parser.add_argument("--mcq-config", default=MCQ_DATASET_CONFIG, help="HF dataset config name (e.g. \"English\"/\"Hindi\" for BhashaBench-Finance)")
    parser.add_argument("--mcq-split", default=MCQ_DATASET_SPLIT, help="Override if wrong for a different --mcq-dataset")
    parser.add_argument("--mcq-sample-size", type=int, default=MCQ_EVAL_SAMPLE_SIZE, help="Questions per MCQ evaluation (subsampled -- full set costs ~4.7hr each, confirmed on the real pod)")
    parser.add_argument("--save-steps", type=int, default=None, help="Override build_training_config's default (500) -- e.g. for a fast smoke test on a tiny data subset")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override build_training_config's default (500) -- e.g. for a fast smoke test on a tiny data subset")
    parser.add_argument("--logging-steps", type=int, default=None, help="Override build_training_config's default (10) -- e.g. for a fast smoke test with fewer total steps than the default")
    parser.add_argument(
        "--disable-gradient-checkpointing", action="store_true",
        help="Trade memory for speed -- disables activation recomputation. Only safe if VRAM headroom is confirmed (checked empirically, not assumed).",
    )
    parser.add_argument(
        "--skip-mcq-eval", action="store_true",
        help="Skip MCQ validation accuracy entirely (e.g. HF_TOKEN/gated access not set up yet, or a quick debug run)",
    )
    args = parser.parse_args()

    train_dataset, val_dataset = load_pretune_dataset(args.train_jsonl, args.val_jsonl)

    import boto3
    import wandb
    from trl import SFTConfig, SFTTrainer

    s3_client = build_s3_client(boto3)
    model, tokenizer = load_fresh_model(use_gradient_checkpointing=not args.disable_gradient_checkpointing)

    mcq_dataset = []
    if not args.skip_mcq_eval:
        from datasets import load_dataset as load_hf_dataset

        # Gated dataset -- requires HF_TOKEN in the environment for an account
        # that has accepted its access terms on huggingface.co (set by
        # whoever runs this, never by Claude). load_mcq_eval_set raises a
        # clear error naming the actual schema if the field-name guesses in
        # this module are wrong -- see its docstring.
        mcq_raw = load_hf_dataset(args.mcq_dataset, args.mcq_config, split=args.mcq_split)
        mcq_dataset = load_mcq_eval_set(list(mcq_raw))
        mcq_dataset = sample_mcq_eval_set(mcq_dataset, args.mcq_sample_size, SEED)

    wandb.init(
        project=args.wandb_project,
        name=f"pretune-{args.dataset_name}",
        config=build_wandb_run_config(args.dataset_name, args.train_jsonl, args.val_jsonl),
    )
    # Track the best (lowest/highest) value seen across the run in the run
    # table, not whichever eval happened to run last. train/eval
    # mean_token_accuracy are trl's own built-in metrics (confirmed on the
    # real pod 2026-09-03) -- no define_metric needed for those, only for
    # this project's own custom-logged ones.
    wandb.define_metric("eval_perplexity", summary="min")
    wandb.define_metric("eval_accuracy_mcq", summary="max")

    callbacks = [
        build_perplexity_callback(wandb),
        build_checkpoint_upload_callback(s3_client, args.s3_checkpoint_prefix),
    ]
    if mcq_dataset:
        callbacks.append(build_mcq_accuracy_callback(wandb, model, tokenizer, mcq_dataset))

    config_overrides = {}
    if args.save_steps is not None:
        config_overrides["save_steps"] = args.save_steps
    if args.eval_steps is not None:
        config_overrides["eval_steps"] = args.eval_steps
    if args.logging_steps is not None:
        config_overrides["logging_steps"] = args.logging_steps
    if args.disable_gradient_checkpointing:
        config_overrides["gradient_checkpointing"] = False
    training_args = SFTConfig(**build_training_config(args.output_dir, **config_overrides))
    trainer = SFTTrainer(
        model=model,
        # processing_class, not tokenizer= -- trl removed the tokenizer=
        # kwarg entirely in v0.16 (TypeError: unexpected keyword argument),
        # confirmed against the currently-installed trl on the pod before
        # this was ever run for real.
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=callbacks,
    )

    trainer.train()
    # No separate post-training evaluate_perplexity() call here: with
    # eval_strategy="steps" (eval_steps=500), the callback above already
    # logged perplexity for every evaluation including the last one at the
    # final step — a second call would just re-run the same evaluation pass
    # over val_dataset for a duplicate number.

    # The last epoch's checkpoint (already uploaded above via the on_save
    # callback) is the trained state, but save_model() to a clean "final"
    # location makes the finished adapter easy to find without knowing which
    # global_step the last epoch landed on.
    import os
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    upload_checkpoint_to_s3(final_dir, f"{args.s3_checkpoint_prefix}final/", s3_client)

    wandb.finish()


if __name__ == "__main__":
    main()
