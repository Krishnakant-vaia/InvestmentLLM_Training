# Investment LLM — Training Scripts

The scripts used to build the VAIA investment LLM (base model `Qwen3.5-9B`): continued pre-training on an investment corpus,
supervised fine-tuning (SFT) on generated Q&A, and preference tuning (DPO). Only the scripts that were actually run are kept.
Datasets, checkpoints and run outputs live in S3 (`s3://ft-vaia/...`) and are not in this repository.

## Pipeline

```
01 Pre-tuning ─────► 02 SFT ─────────────────────────────────► 03 DPO
corpus → raw-text    1 generate Q&A → 2 filter + score →        1 build preference pairs →
LoRA                 3 train / merge / evaluate                 2 train / merge / evaluate
                     4 v3 dataset rebuild (refined questions)
```

Each stage consumes the previous stage's output from S3: pre-tuning produces the pre-tuned base model, SFT trains on top of it,
and DPO starts from the merged SFT model.

## Where to find things

```
InvestmentLLM_trainingscripts/
├── 01_pretuning/                 continued pre-training
│   ├── data_prep/                list corpus keys, tokenise and dump train/val to S3
│   ├── training/                 LoRA pre-training + perplexity/MCQ eval
│   ├── wandb_tracking/           W&B logging for dataset dumps
│   └── deploy/                   pod setup script
├── 02_sft/
│   ├── 01_qa_generation/         generate SFT Q&A from the corpus (all 8 capabilities, one run)
│   ├── 02_qa_filtration/         LLM-judge filtering, quality scoring, fine-tuning pool
│   ├── 03_training/              SFT training, merging, evaluation, Daytona runner, serving
│   └── 04_dataset_v3_refined/    question selection for the v3 dataset rebuild
├── 03_dpo/
│   ├── 01_pair_generation/       preference pairs from CIMs, tear sheets, numerical sets; judging, rewrites
│   └── 02_training/              DPO training, merging, evaluation, Daytona runner
└── sft_common/                   shared library for 02_sft/03_training and 03_dpo/02_training
```

| Folder | Start here | Main entry points |
|---|---|---|
| `01_pretuning/` | `README.md` | `data_prep/pretune_dump.py`, `training/train_pretune.py` |
| `02_sft/01_qa_generation/` | `README.md` | `run_sft_generation.py` |
| `02_sft/02_qa_filtration/` | `README.md` (run order table) | `scripts/run_pipeline.py`, `scripts/run_quality_scoring.py`, `scripts/build_finetune_pool.py` |
| `02_sft/03_training/` | `README.md` | `02_training/sft_train.py`, `03_merging/merge_model.py`, `04_evaluation/`, `06_daytona/daytona_run.py` |
| `02_sft/04_dataset_v3_refined/` | `README.md` | `run_question_selection.py` |
| `03_dpo/01_pair_generation/` | `README.md` | `pipeline.py`, `daytona_run.py`, `score_pairs.py`, `make_training_set.py` |
| `03_dpo/02_training/` | `README_DPO.md` | `02_training/preflight.py`, `02_training/dpo_train.py`, `06_daytona/daytona_run.py` |

## How to use

1. **Pick a stage** from the table and open its README. Every stage is self-contained: its own `requirements*.txt`,
   `.env.example`, tests and (where needed) `config.yaml`.
2. **Install** that stage's requirements (Python 3.11+):
   ```bash
   pip install -r requirements.txt               # or requirements/<step>.txt in the two training stages
   ```
3. **Add credentials**: copy `.env.example` to `.env` in the same folder and fill it in. `.env` is git-ignored; never commit it.
4. **Run from the stage folder.** Scripts resolve config, prompts and outputs relative to it, e.g.:
   ```bash
   cd 02_sft/03_training
   python 01_data_prep/check_aws.py
   python 06_daytona/daytona_run.py smoke
   ```
5. **Test** a stage from its folder:
   ```bash
   python -m pytest -q
   ```

## Conventions

- **Prompts** are plain files next to the code that uses them (`prompts/` in `02_qa_filtration` and `01_pair_generation`).
- **Shared library**: `02_sft/03_training` and `03_dpo/02_training` both import `sft_common/` from the repo root. Their
  `step_root.py` (imported first by every script) puts it on the path and points it at the stage's own `config.yaml`, `.env`
  and `outputs/`. The Daytona runners pack `sft_common/` into the sandbox upload.
- **Long runs** are resumable: rerun the same command after an interruption. Most runners support a pause file; see each README.
- **Never committed**: `.env`, keys, data, model weights, logs, run state and outputs (see `.gitignore`).

## History

- `02_sft/01_qa_generation` merges the former `first_4_capabilities/` and `last_4_capabilities/` copies, which existed only to
  split the run across two PCs.
- `02_sft/02_qa_filtration` tests need its `requirements.txt` installed (`datasketch`, `moto`, `responses`).
