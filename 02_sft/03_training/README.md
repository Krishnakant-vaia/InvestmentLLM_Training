# SFT pipeline: Qwen3.5-9B for finance & investment

Fine-tunes a **pre-tuned Qwen3.5-9B** on ~208k QA pairs (finance/investment, general chat, tool
calling) with LoRA, checks the result against the model it started from, and serves it on Modal.

- **All settings:** [`config.yaml`](config.yaml). Every key is commented.
- **All secrets:** `.env`. Nothing is hardcoded.
- **Everything a run produces:** saved to S3 under one folder per run.

---

## 1. The pipeline at a glance

```
 S3 pool (208k QA pairs)          pre-tuned model (S3, 18 GB)
          │                                  │
  [1] data prep ──► train.jsonl / val.jsonl  │
          │                                  │
  [2] training (LoRA) ◄──────────────────────┘ ──► checkpoints + adapter ──► S3
          │
  [3] merge adapter into the model ──► merged model ──► S3
          │
  [4] evaluate: pre-tuned vs fine-tuned ──► scores ──► S3
          │
  [5] batch inference / [8] live API on Modal (up to 262k tokens per request)
```

| Step | Folder | Script | What it does |
|---|---|---|---|
| 1 | `01_data_prep/` | `check_aws.py`, `download_s3.py`, `build_mixture.py` | Checks access, downloads the pool, builds train/val |
| 2 | `02_training/` | `sft_train.py` (or the notebook) | LoRA fine-tuning; saves every checkpoint to S3 |
| 3 | `03_merging/` | `merge_model.py` | Bakes the LoRA adapter into a normal 16-bit model |
| 4 | `04_evaluation/` | `evaluate.py` | Scores pre-tuned vs fine-tuned on held-out QA pairs |
| 5 | `05_production_deployment/` | `batch_inference.py` | Offline: file of questions in, file of answers out |
| 8 | `08_model_deployment/` | `serve.py` | Live OpenAI-compatible streaming API |

---

## 2. Quick start

### Setup
```bash
pip install -r requirements/data_prep.txt    # this machine
pip install -r requirements/training.txt     # GPU machine (Unsloth, TRL, ...)
pip install -r requirements/evaluation.txt   # GPU machine (vLLM)
```

### Credentials
Copy `.env.example` to `.env` and fill it in. `.env` is git-ignored and never uploaded anywhere.

| Variable | Needed for |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Data, pre-tuned model, saving runs to S3 |
| `WANDB_API_KEY` | Loss charts in wandb project `vaia-sft-qwen35-9b` (optional) |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | The eval judge (optional: `evaluate.py --no-judge`) |
| `DAYTONA_API_KEY`, `DAYTONA_API_URL`, `DAYTONA_TARGET` | Running on a Daytona GPU sandbox (never copied into the sandbox) |

### Run
```bash
python 01_data_prep/check_aws.py             # credentials OK, inputs readable, S3 run folder writable
python 01_data_prep/download_s3.py           # the 1.4 GB pool
python 01_data_prep/build_mixture.py         # -> data/processed/train.jsonl, val.jsonl

python 02_training/sft_train.py --memory-check    # peak GPU memory of the heaviest micro-batches (no training)
python 02_training/sft_train.py --check-packing   # do packed QA pairs stay apart? (no training)
python 02_training/sft_train.py --benchmark 1,2,4,8   # speed, peak GPU memory and ETA for micro-batches of
                                                       # 12k / 24k / 48k / 96k tokens (no training, nothing saved)
python 02_training/sft_train.py                   # train: new run "sft-YYYY-MM-DD_HH-MM"

python 03_merging/merge_model.py             # merge the last run
python 04_evaluation/evaluate.py             # compare pre-tuned vs fine-tuned

modal volume put sft-models outputs/merged/<run_name> merged
modal deploy 08_model_deployment/serve.py
```
**On a Daytona GPU sandbox** (no GPU of your own), three jobs, each in a fresh sandbox that is always deleted at the end:

| Command | What it does | Time |
|---|---|---|
| `python 06_daytona/daytona_run.py benchmark` | The 1/2/4/8 micro-batch size test: peak memory, tokens/s, ETA | ~35 min |
| `python 06_daytona/daytona_run.py smoke` | 20 real training steps with the real Trainer: 2 checkpoints and 2 evals saved to S3, then a check of what landed on S3 | ~45 min |
| `python 06_daytona/daytona_run.py train` | The full fine-tuning run | many hours |

Run `benchmark` and `smoke` first, and start `train` only when both are clean. How it protects a long run:
- **Detached:** training runs in the background on the sandbox, so closing the laptop does not stop it.
  `python 06_daytona/daytona_run.py attach --run <run name>` re-watches it. Ctrl-C also leaves it running.
- **Spot preemption:** every checkpoint is on S3 within seconds of being saved. If Daytona takes the spot sandbox
  away, the script creates a new one and resumes from the last S3 checkpoint (up to `--auto-resume N` times, default 3).
  Continue a run by hand with `train --resume --run <run name>`.
- **One sandbox per run:** a second sandbox for the same run is refused (`--force` overrides).
- **Lifetime cap:** Daytona deletes the sandbox after `daytona.ttl_minutes` (benchmark, smoke) or `train_ttl_minutes`
  (train) even if the script dies, so a forgotten sandbox cannot run up costs.
The code and a `.env` without the Daytona key are uploaded; the console log goes to `outputs/daytona/<name>/` and to S3.
Settings: `config.yaml` -> `daytona`. Options: `--gpu H100,H200`, `--on-demand`, `--keep`.

Steps 3 and 4 use the last run trained on this machine. Pass `--run <run_name>` to pick another; its files are fetched from S3 if they aren't here.

---

## 3. Step 1: data preparation

- **Input:** one S3 file, used **exactly as provided**. It's three sources one after another:

  | Rows | Source | Content |
  |---|---|---|
  | 1 – 178,000 | `domain` | Finance/investment QA: a source document + question, answer with reasoning in `<think>` |
  | 178,001 – 190,000 | `general` | General chat (much of it Glaive-style) |
  | 190,001 – 208,018 | `tool_calling` | Function calling, calls written as `<functioncall> {...}` or `<tool_call>[...]` |

- **Rendering:** each row is turned into the exact text the model trains on, using the pre-tuned model's own chat template. Training text therefore matches what the model sees when served.
- **Kept as-is:** tool-call text is not reformatted.
- **The only change:** a trailing user turn with no answer after it is removed (438 rows).
- **Validation split:** 1% of each source is held out as `val.jsonl` and never trained on.

---

## 4. Step 2: training — how it works

### What is trained
- **LoRA:** the 9B model's own weights stay **frozen**. Small trainable matrices (rank 64) are added to the projections listed in `training.lora.target_modules`, and only these are trained, a small share of the parameters. Biases aren't trained. Qwen3.5 has two attention kinds, so the list covers both: `q/k/v/o_proj` (8 full-attention layers), `in_proj_qkv`, `in_proj_z`, `out_proj` (24 linear-attention layers) and `gate/up/down_proj` (the MLP in all 32). Training prints `[lora] adapters per target: {...}` so you can see how many layers each name really reached; a name that matches nothing is skipped silently by PEFT and is flagged with a warning.
- **Assistant tokens only:** loss is computed only on what the model should write. System prompts, questions, documents and tool results are context, not targets.
- **Loss:** cross-entropy on predicting the next token of the reference answer, averaged per token.

### How QA pairs are grouped (`training.batching_mode`)
GPU memory depends on the tokens in one **micro-batch** (measured: ~2.6 MB per token, mostly the output layer over Qwen3.5's 248k-word vocabulary), so each holds up to **12,000 tokens** of whole QA pairs. No QA pair is ever cut or split.

**Row length limit: 12,000 tokens** (`model.max_seq_length`). A QA pair longer than that is dropped whole, never cut. That affects about **255 of 208,018 rows** (0.12% of rows, 0.6% of the tokens), all tool-calling. Domain (longest 8,517 tokens) and general (longest 5,255) lose nothing. The startup report lists the exact numbers per source. A 25,006-token row alone filled an 80 GB H100, which is why the limit is 12k and the micro-batch budget matches it.

- **`packed` (default):** QA pairs are placed end to end with no padding.
  1. The longest remaining pair starts a micro-batch.
  2. The gap is filled **best fit**: the largest remaining pair that still fits, again and again until nothing fits.
  3. Small pairs are saved for the last small gaps, so micro-batches end up almost exactly full and no pair trains alone.

  Example: `8.5k + 2.7k + 0.6k ≈ 12k`.
- **`padded` (safe fallback):** pairs in a micro-batch are padded to the longest one. Pairs of similar length are grouped so little is wasted; very long pairs go alone.

Either way:
- **Order:** micro-batches are shuffled with a fixed seed, so long and short, domain and tool pairs mix across the whole run, and a rerun gives the same order.
- **Per step:** 4 micro-batches make one optimizer step (~128k tokens).

### Safety checks before training starts
| Check | Stops the run if |
|---|---|
| **Integrity** | Any QA pair would be lost, duplicated or cut at the length limit |
| **Packing** (packed mode) | Packed pairs can "see" each other. It compares each pair's loss alone vs packed; they must match within 1%. If this fails, switch to `batching_mode: padded`. |
| **Resume** | On `--resume`, the grouping differs from the run's original one (e.g. data rebuilt), which would skip the wrong pairs |

The exact grouping is saved as `logs/batch_plan.jsonl`: one line per micro-batch, with the `train.jsonl` line numbers of the QA pairs trained together.

### During training
- **Losses:** train loss every 10 steps; val loss every 500 steps on 1,000 fixed val pairs.
- **Fit monitor:** reports **healthy / overfitting / stalled** from the train vs val loss trends.
- **Checkpoints:** saved every 500 steps and **uploaded to S3 immediately**. Locally only the newest 3 are kept; S3 keeps all of them.
- **wandb** (if the key is set): `train/loss`, `eval/loss`, learning rate, gradient norm and the fit-monitor charts. A resumed run continues the same charts.

### Choosing the micro-batch size (`--benchmark`)
`--benchmark 1,2,4,8` tests micro-batches of N × 12k tokens (12k, 24k, 48k, 96k) on your GPU. For each size it runs real forward and
backward passes on real QA pairs (the hardest micro-batches first) and prints peak GPU memory, tokens/s and the ETA
for the whole run, e.g. `size 2 | 24000 | 61 GB | 77% | 11,000 tokens/s | 15 h`. Out of memory is reported, and
larger sizes are skipped. It then recommends the fastest size using at most 90% of GPU memory, with the
`config.yaml` values to set. Bigger micro-batches need fewer per step (`grad_accum_steps` = 10 / N, minimum 1) so a step stays ~120k tokens.

### Resume
`python 02_training/sft_train.py --resume [--run NAME]` continues from the newest checkpoint, **local or on S3**, so a run can continue on a different machine.

---

## 5. Step 3: merging
`merge_model.py`:
- loads the pre-tuned model plus the run's adapter (downloaded from S3 if needed);
- computes `W + B×A` for every adapted layer;
- saves a normal 16-bit model to `outputs/merged/<run>/` and uploads it to S3 `merged/`.

It also makes sure generation stops at the end of each turn (`<|im_end|>`).

---

## 6. Step 4: evaluation — is the fine-tuned model better?
Runs **both** the pre-tuned and the fine-tuned model on the same **200 held-out QA pairs per source** (never trained on), then prints a side-by-side table.

| Metric | Meaning |
|---|---|
| `non_empty`, `complete`, `clean` | Answered; stopped on its own (not cut at the limit); no leaked template tokens |
| `call_made`, `call_parses`, `name_match`, `args_match` | Tool rows: called a tool when it should; the call is valid; right function; right arguments |
| `false_call` | Called a tool when the reference answer didn't |
| `f1` | Word overlap with the reference. A rough trend only: correct paraphrases score low |
| `judge_correctness`, `judge_reasoning` | Domain rows: 1–5 scores from **gpt-oss-120b** (Cloudflare) comparing the answer with the reference |

- **Settings:** domain questions are answered with reasoning on (up to 4,096 tokens); the others with reasoning off.
- **Judge caveat:** the domain references were written by the same gpt-oss-120b, so its scores are for comparing the two models, not absolute truth.
- **Output:** per-answer details and scores in `outputs/eval/<run>/`, uploaded to S3 `eval/`.

---

## 7. Where everything is saved

| What | Local | S3 (`storage.s3_root/<run_name>/`) |
|---|---|---|
| Config snapshot, library versions | `outputs/sft/<run>/config/` | `config/` |
| Train/val data, stats | `data/processed/` | `data/` |
| Startup report, batch plan, fit analysis, loss history | `outputs/sft/<run>/logs/` | `logs/` |
| Every checkpoint | `outputs/sft/<run>/checkpoint-N/` (newest 3) | `checkpoints/` (all) |
| Final LoRA adapter | `outputs/sft/<run>/final/` | `final_adapter/` |
| Merged model | `outputs/merged/<run>/` | `merged/` |
| Eval results | `outputs/eval/<run>/` | `eval/` |

`outputs/last_run.txt` remembers the latest run, so merge and eval find it without `--run`.

---

## 8. Settings you are most likely to change (`config.yaml`)

| Setting | Default | Change it when |
|---|---|---|
| `training.batching_mode` | `padded` | `packed` needs flash-attention-2 / varlen kernels verified first |
| `training.tokens_per_micro_batch` | `12000` | `--benchmark` shows room for more (raise it) or too little (lower it) |
| `training.grad_accum_steps` | `10` | You want more or fewer tokens per optimizer step |
| `training.learning_rate` | `1e-4` | Loss is unstable (lower) or stalls (higher) |
| `training.epochs` | `1` | You want more passes over the data |
| `training.save_steps` / `eval_steps` | `500` | You want checkpoints or val loss more or less often |
| `evaluation.samples_per_source` | `200` | You want a bigger or faster eval |
| `storage.s3_root` | `.../092826_SFT_FineTune_Model/Fine_Tune_Run_2/` (round 1 is archived in `Fine_Tune_Run_1/`) | You save runs somewhere else |

---

## 9. Facts verified on the real model and data
- **Pre-tuned model:** `s3://ft-vaia/Corpus/Invest_dataset/09022026_pretune_results/merged_model/`, a single 17.9 GB file.
  - Architecture `Qwen3_5ForCausalLM`: 32 layers, of which 8 are full attention and 24 linear attention. Vocabulary 248,320.
  - `<think>` / `</think>` are already single tokens, so no vocabulary change is needed.
- **Pool:** 208,018 rows, all usable, ~603M tokens (average ~2,900 per row: domain ~3,250, general ~540, tool-calling ~1,010). The longest QA pair is 25,006 tokens; at the 12,000 limit ~255 tool-calling rows are dropped.
- **Required package:** 24 of the 32 layers are linear-attention layers that need `flash-linear-attention` (pinned in `requirements/training.txt`). Without it transformers silently uses a slow PyTorch fallback needing tens of GB more memory; `sft_train.py` and the Daytona job now stop immediately if it is missing. (The log line `The fast path is not available…` still appears because the optional `causal-conv1d` is not installed; that only affects a small convolution.)
- **Serving:** only 8 layers keep a cache (32 KB per token), so a full 262k-token request needs ~8.6 GB. That fits on an A100-80GB.

---

## 10. Still to confirm on the first GPU run
- **`--check-packing`:** whether Qwen3.5's linear-attention layers keep packed QA pairs apart. This decides packed vs padded.
- **`--benchmark 1,2,4,8`:** which micro-batch size (12k / 24k / 48k / 96k tokens) fits the GPU, and the ETA.
- **Unsloth support:** that Unsloth loads and trains Qwen3.5. If not, try `FastModel` instead of `FastLanguageModel` in `sft_train.py`.
- **Tool-call position:** the chat template writes tool calls in the middle of a conversation without the empty `<think></think>` block they get when served. The eval's tool metrics show whether that matters.
- **Boilerplate:** domain documents contain scraped web boilerplate (menus, banners). Worth cleaning if answers look noisy.

---

## 11. Code layout

```
config.yaml             all settings          .env / .env.example   secrets (never uploaded)
01_data_prep/           step 1                02_training/          step 2 (+ notebooks)
03_merging/             step 3                04_evaluation/        step 4
                        (step 6, DPO: lives in ../../03_dpo/02_training)
05_production_deployment/  step 5             08_model_deployment/  step 8
06_daytona/             run a job on a Daytona GPU sandbox
../../sft_common/       shared code (below; repo root, loaded by step_root.py)   tests/   CPU-only tests
```

| `sft_common/` module | Responsibility |
|---|---|
| `config.py` | Loads `config.yaml` and `.env` |
| `sources.py` | Pool row → messages (text kept verbatim) |
| `chat_format.py` | Renders text with the model's own chat template |
| `batching.py` | Groups QA pairs into micro-batches (packed / padded), coverage proof, batch plan |
| `runs.py` | A run's name and folders; training, merge and eval share one run |
| `s3_sync.py` | Every S3 upload/download of a run; retries, never uploads `.env` |
| `callbacks.py` | Fit monitor; uploads each checkpoint as it is saved |
| `pretuned.py`, `s3_models.py`, `adapter.py` | Fetching the pre-tuned model; making adapters portable |
| `tracking.py` | wandb (optional) |
| `s3_fetch.py`, `dpo_data.py`, `dpo_loss.py` | Fetching models and files from S3; DPO pair handling and loss (used by `03_dpo/02_training`) |
| `benchmark.py` | Micro-batch size benchmark: which batches to run, ETA maths, results table, recommendation |
| `tool_calls.py`, `metrics.py`, `judge.py` | Step 4: parsing tool calls, scoring, the Cloudflare judge |

### Tests (no GPU needed)
```bash
pip install -r requirements/dev.txt && python -m pytest tests -q
```
They cover data prep, chat rendering on the real template, both batching modes, the eval metrics and judge (fake server), and S3 storage (fake S3).
