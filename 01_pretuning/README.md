# 01 · Pre-tuning

Continued pre-training of `Qwen/Qwen3.5-9B-Base` on the filtered investment corpus (raw text, completion-style, LoRA).

```
data_prep/        list_doc_keys.py   -> doc_keys.txt (S3 keys of the corpus documents)
                  pretune_dump.py    -> tokenised train/val JSONL on S3 (+ optional W&B dataset artifact)
training/         train_pretune.py   -> LoRA training, perplexity eval, optional MCQ eval, checkpoints to S3
wandb_tracking/   wandb_logger.py    -> standalone W&B run/artifact logging for a finished dump
deploy/           setup_and_run.sh   -> pod setup (RunPod): installs deps, checks credentials
conftest.py       puts the three code folders on the import path for pytest
```

## Run

```bash
pip install -r requirements.txt
cp deploy/.env.example deploy/.env      # fill in AWS (+ W&B) credentials
python data_prep/list_doc_keys.py --prefix Corpus/Invest_dataset/082126_fliteredextractionrun2/ --out doc_keys.txt
python data_prep/pretune_dump.py --doc-keys-from doc_keys.txt --s3-prefix <output prefix> --resume-log resume.log
python training/train_pretune.py --train-jsonl <train.jsonl> --val-jsonl <val.jsonl> --output-dir <dir>
```

`pretune_dump.py --dry-run --local-docs-dir <dir>` runs locally without S3. See `deploy/README.md` for running on a pod.

## Test

```bash
python -m pytest -q
```
