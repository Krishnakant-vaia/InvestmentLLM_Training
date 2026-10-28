# DPO on the SFT model (step 6)

> **Status (2026-10).** This folder is the SFT training rebuild (`02_sft/03_training`) adapted to DPO. The original notebook is
> kept as `DPO_Workflow.ipynb`. Sections below that say "notebook" describe the original review; this box says what is new.
>
> **Data source:** `qa_pairs/training/dpo_train.jsonl` (+ `manifest.json`) on S3, built by `../01_pair_generation/make_training_set.py`: 13,029 pairs
> (documents 3,780; numerical sets 5,704 + 3,147 + 398). Rows: `{prompt: [system, user], chosen, rejected, source, origin, pair_id, scores...}`.
> A local copy for checks is in `data/raw/` (not for upload).
>
> **New data layer, `pairs/`:**
> - `source.py`: downloads the file and refuses it unless it matches the manifest (count, per-origin counts, no repeated `pair_id`, no missing fields).
> - `shape.py`: evaluation groups = domains (not ~60 capabilities); the train/val split keeps a source document or a question template together
>   (the numerical questions are templates that differ only in names and numbers, so a split by exact prompt leaks); optional
>   `think_format: workings_in_think` (working into `<think>`, final "Answer:" outside, both sides or neither) and `max_preferred_longer_share`
>   (length guard, off by default).
> - `clean.py`: removes bracketed remarks and "; ... ignored." notes from BOTH answers. Cleaning only one side created the opposite tell once.
> - `audit.py`: CPU surface-tell test (`python pairs/audit.py --data data/raw/dpo_pairs.jsonl`).
> - `sft_common/dpo_data.py` (shared package at the repo root) now carries a `group` per pair and splits by whole groups (val is filled in whole groups, never one group that swamps it).
>
> **Finding that decides the next step:** on a 5,000-pair sample a text-only model ranks the preferred answer above the not-preferred one in 93.9% of
> pairs (picking the longer answer wins 77.6%; length-matched pairs 88.1%; numerical domains 96-100%, document domains 76-82%). DPO would learn
> that form, not the mistake. Remedy being built: rewrite the not-preferred answers into hard negatives (same mistake, same form); see
> `../01_pair_generation/README.md` (82 of 100 sample rewrites accepted). Also measure `ref/prefers_chosen` with `--check` on a GPU before a full run.
>
> **Not done yet:** wiring `pairs/` into `02_training/dpo_train.py` and `config.yaml` (`dpo.data.shape`, `manifest_s3_uri`, `s3_uri`); a DPO-only
> launcher (this copy of `06_daytona/daytona_run.py` still has the SFT and eval jobs, and `tests/test_daytona.py::test_zip_of_the_real_project_excludes_secrets`
> fails here for that reason); the loss length-normalisation option; the `<think>` format decision (71% of the pairs have no think block);
> the benchmark overlap check; a GPU `--max-steps 10` smoke run (nothing here has run on a GPU). Tests: 82 pass, 9 skipped (they need the tokenizer/GPU).

Rewrite of `DPO_Training/DPO_Workflow.ipynb`. The notebook is left untouched; everything below replaces it and runs on
the same Daytona / S3 / barrier machinery as SFT.

## If the GPU is taken away while the laptop is off (resume from the last checkpoint)
Training runs detached on the Daytona sandbox and saves a checkpoint to S3 every 100 steps, so the laptop is only needed for the launcher's
auto-resume and cleanup. If the sandbox vanished (spot eviction) while nobody was watching:
1. Check the newest checkpoint: `s3://ft-vaia/Corpus/Invest_dataset/100626_Dataset_DPO/DPO_Run_1/<run>/checkpoints/checkpoint-N/` (steps 100, 200, ...).
2. Resume with the SAME run name: `python 06_daytona/daytona_run.py dpo --resume --run <run>`. It creates a new sandbox, pulls the newest checkpoint from S3
   and continues; at most 100 steps are redone. Up to 3 automatic resumes follow while the launcher is running.
3. Do NOT change `config.yaml` (the `dpo:` section), the pairs file on S3 or the code before resuming: the trainer compares the micro-batch plan with the one
   saved at the start and refuses to resume if it differs (it would skip the wrong pairs).
4. When a run has finished, delete nothing by hand: the launcher removes the sandbox (or Daytona does at the lifetime cap, `dpo.ttl_minutes`, 12 h).
`python 06_daytona/daytona_run.py attach --run <run>` re-watches a run that is still going.

## Before the first run: what you must supply

| Item | Where | Status |
|---|---|---|
| The preference pairs (JSONL, `{prompt, chosen, rejected, source}`) | `dpo.data.s3_uri` in `config.yaml` | **missing**: the notebook points at `10K_dpo.jsonl`, which is not on this machine |
| Base model = merged Run 2 SFT model | `dpo.base.s3_uri` | set to `Fine_Tune_Run_2/merged_models/sft-2026-10-01_23-20_final/` |
| Where DPO runs are saved | `dpo.s3_root` | set to `.../092826_SFT_FineTune_Model/DPO_Run_1/` (change if you prefer another folder) |

Pair rules (the code enforces the first four, you must supply the rest):
`prompt` is the user question (text) or a list of chat messages; `chosen`/`rejected` are the two answers; a pair is dropped
whole if empty, identical, only one answer has `<think>`, the answers differ more than 2x in length, or a side exceeds
6,000 tokens. **The answers must be in the format the SFT model writes** (the notebook's example used `<answer>` tags; SFT never saw them).

## Run

```bash
python 06_daytona/daytona_run.py dpo --max-steps 10      # first: 10 real steps, 2 checkpoints + 2 evals, proves the pipeline on the GPU
python 06_daytona/daytona_run.py dpo                     # the full run (detached, spot-eviction safe, auto-resume)
python 06_daytona/daytona_run.py dpo --resume --run dpo-YYYY-MM-DD_HH-MM
python 06_daytona/daytona_run.py attach --run dpo-YYYY-MM-DD_HH-MM
python 06_daytona/daytona_run.py merge --run dpo-YYYY-MM-DD_HH-MM [--checkpoint N]   # adapter + the SFT model -> merged_models/dpo-..._final
python 06_daytona/daytona_run.py eval  --run dpo-YYYY-MM-DD_HH-MM --workbook X.xlsx [--overlap]
```

Before every training start the script proves (and stops if not): the data is clean, every pair is in exactly one
micro-batch, **the reference really is the adapter-free model**, and the two heaviest micro-batches fit in GPU memory.
Watch in the log / wandb: `eval_<source>_rewards/accuracies` (should rise above 50%), `logps/chosen` (must not fall much),
`ref/prefers_chosen` (share of pairs the SFT model already ranks right; near 100% = nothing to learn, near 0% = the
pairs disagree with the model's own taste).

## Issues found in the notebook, and what replaced them

### It would not run
| # | Notebook | Why it fails | Now |
|---|---|---|---|
| 1 | Cell 6 uses `DPO_PAIRS_JSONL` | never defined: `NameError` | cell removed; pairs path comes from `config.yaml` |
| 2 | `TrainingArguments(evaluation_strategy=...)` | removed in transformers 5 (checked: `TypeError`) | `eval_strategy` |
| 3 | `DPOTrainer(tokenizer=, beta=, max_length=, max_prompt_length=)` with plain `TrainingArguments` | these moved to `DPOConfig` / `processing_class` in TRL; the project pins TRL 0.24 for Unsloth | no TRL `DPOTrainer`: own trainer (`02_training/dpo_train.py`), see "Why not TRL" |
| 4 | Callback does `kwargs.get('trainer')` | HF callbacks are not given the trainer (checked: only model, tokenizer, optimizer, dataloaders): it is `None`, so the per-source eval never ran; had it run, `trainer.evaluate` inside `on_evaluate` would recurse | per-source eval via the Trainer's own dict-of-datasets support |
| 5 | Callback filters on `source_type` | the schema check only requires prompt/chosen/rejected: `KeyError` on a file without it | `source` (or `source_type`) read once, missing = `unknown` |
| 6 | `import torch`, `datasets` before `unsloth` | Unsloth must be imported first to patch transformers | order fixed |

### It would run and give wrong results (the dangerous ones)
| # | Notebook | Consequence | Now |
|---|---|---|---|
| 7 | "ESG" and "General" both read `10K_dpo.jsonl` | the "70/30 mix" is the same 10K twice: 19,000 training rows that are duplicates, and **every validation row is also in training** (eval accuracy is meaningless) | one file, split **by prompt**; test asserts no prompt is on both sides |
| 8 | Prompts/answers fed as raw text | the model was trained (SFT) on the chat template (`<|im_start|>user ... <|im_end|>`); DPO on raw strings optimises a format never used at inference | rendered with the model's own chat template, byte-identical to SFT text (test), answers end on `<|im_end|>` |
| 9 | `load_in_4bit=True` | the reference is a 4-bit copy of the SFT model, not the model SFT produced; policy and reference both drift from BF16 | BF16, like SFT |
| 10 | LoRA on q/k/v/o/gate/up/down only | Qwen3.5's 24 linear-attention layers (`in_proj_qkv`, `in_proj_z`, `out_proj`) stay frozen: the SFT config already fixed this once | same 10 targets as SFT |
| 11 | `max_prompt_length=1024`, `max_length=8192` | TRL truncates silently: a cut answer becomes a "chosen" or "rejected" one | pairs over the limit are dropped whole and counted in the report |
| 12 | Length ratio allowed up to 2x, measured in characters, on the first 2,000 rows, printed not enforced | DPO learns "longer wins" instead of quality | measured in tokens on every pair, enforced, plus a warning when chosen is the longer one in >70% (or <30%) of pairs |
| 13 | Cell 10: `if "<think>" not in tokenizer.additional_special_tokens` then adds `<think>` as a special token and resizes the embeddings | **crashes**: this tokenizer has no `additional_special_tokens` in transformers 5 (checked). The idea is also wrong: `<think>` / `</think>` already exist (ids 248068 / 248069) as ordinary tokens, so decoding keeps them; making them special would strip them from decoded text and touch the embedding matrix | removed; think blocks go through the chat template |
| 14 | ESG / carbon-credit prompts, smoke tests, and "General" data | the model is VaiaGPT (finance) | needs finance pairs (see top) |
| 15 | No monitoring of the chosen answers' likelihood | DPO can push both answers down and still "win" (likelihood displacement) | logged every eval, warning at a 15% fall, `dpo.sft_alpha` (RPO term) as the remedy |
| 16 | No check that the reference equals the frozen model | if adapter switching silently does nothing, DPO compares the model with itself | `reference_check` nudges the adapter: policy log-probs must move, reference's must not |

### It would run out of memory or time
| # | Notebook | Now |
|---|---|---|
| 17 | TRL's DPO builds fp32 scores over the full 248,320-word vocabulary for every token of both answers (about 1 MB per token before softmax copies): a 6k-token pair is ~12 GB for scores alone, several times that with the backward copies | only answer tokens are scored, 2,048 at a time, recomputed in backward (`sft_common/dpo_loss.py`, same idea as SFT's lean loss; tested equal to the naive computation, values and gradients) |
| 18 | Batch size 1, accumulation 8: one pair at a time | token-budget batching (same sampler as SFT), about 30 to 50 pairs per optimizer step; budget halved because a pair is two sequences and the reference needs a pass too |
| 19 | OOM would appear hours in, on whichever long pair is shuffled late | the two heaviest micro-batches run once before training starts |

### It would lose the work
| # | Notebook | Now |
|---|---|---|
| 20 | Checkpoints only on the local disk, `save_total_limit=2` | every checkpoint to S3 at once (`DPO_Run_1/<run>/checkpoints/`), spot eviction resumes automatically, batch-plan fingerprint stops a resume on changed data |
| 21 | Hard-coded `./merged_model_400/`, no merge step | `merge_model.py` merges a `dpo-...` adapter into the **SFT merged model** (not the pre-tuned one), to `merged_models/dpo-..._final` |
| 22 | Interactive notebook on a billed GPU | detached Daytona job with a lifetime cap (`dpo.ttl_minutes`) and the same sandbox cleanup as SFT |
| 23 | No eval barrier | DPO sandboxes get no eval code or judge keys (test: the bundle passes the barrier scan); `eval --overlap` on a DPO run checks the eval questions against `dpo_pairs.jsonl` |

## Why not TRL's `DPOTrainer`
The project pins `trl==0.24.0` / `transformers==5.5.0` for Unsloth. `DPOTrainer` computes full-vocabulary scores
(issue 17), truncates silently (11), applies its own chat-template logic that differs from SFT's (8), and its internals
change between releases. The loss is ten lines; owning it lets the checks above (coverage, frozen reference, memory) exist.
The Trainer, optimizer, scheduler, checkpointing and wandb are still the stock transformers ones.

## Settings that are judgement calls (not measured), all in `config.yaml -> dpo`
* `learning_rate 1e-5`, `beta 0.1`, LoRA scale 2 (no rsLoRA): gentle on purpose, because SFT already drifted once (Run 1).
  If `rewards/accuracies` stays near 50% after the first third of the run, raise the learning rate before changing anything else.
* `grad_accum_steps 8`, `eval/save every 100 steps` (set by the user for this project; one eval of the 9 groups takes ~5 min): about 30 to 50 pairs per step; the 13,029-pair file is 687 steps, so about 6 checkpoints. Outputs go to `100626_Dataset_DPO/DPO_Run_1/`; the AWS check writes its probe there and never deletes it.
* The ETA printed at startup is derived from SFT's measured speed, not measured for DPO.

## Tested
CPU unit tests (maths equal to a naive reference, template parity with SFT, split leakage, collator, barrier, Daytona job)
and an end-to-end CPU run of the real trainer on a tiny random Qwen3 + LoRA (`tests/test_dpo_trainer.py`: first eval loss is ln 2,
only the adapter changes, per-source evals logged, checkpoints saved). **Not yet run on a GPU with Unsloth**: do the
`--max-steps 10` run first.
