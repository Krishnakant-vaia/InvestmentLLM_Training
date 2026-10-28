# 02.2 · SFT Q&A filtration and scoring

Filters the generated Q&A pool with an LLM judge, scores the survivors 1-10, and builds the fine-tuning pool.
Run every command from this folder (scripts import `qa_filter`/`qa_scoring` and read `prompts/` relative to it).

```
qa_filter/      filtration pipeline: prefilter, batching, judge client with retries, checkpoint, pause
qa_scoring/     quality-scoring pipeline: prompt loading, scoring client, records
prompts/        judge and scoring prompts, plus systemprompt_SFT.txt (the system prompt every SFT row carries)
scripts/        runnable stages (below); scripts/experiments/ holds one-off prompt trials
docs/           architecture notes for the quality analysis
```

## Run order

| Stage | Command |
|---|---|
| 1. Smoke test the judge | `python scripts/run_smoke_test.py` |
| 2. Filter the pool | `python scripts/run_pipeline.py` (monitor: `check_progress.py`, pause: `pause_run.py`) |
| 3. Re-filter + merge on S3 | `refilter_measurement.py`, `merge_and_restructure_s3.py` |
| 4. Quality scoring | `python -u scripts/run_quality_scoring.py --input-file merged_output/sft_data_kept.jsonl --state-dir scoring_state/full` (then `rescore_unscored.py`, `upload_scoring_shards.py`) |
| 5. Build the pool | `python -u scripts/build_finetune_pool.py` |
| 6. Clean document references | `refill_stage1.py` -> `refill_stage2.py` -> `refill_stage3.py` |
| 7. Add function-calling / chat data | `python -u scripts/merge_pool_sources.py` |

## Setup and tests

```bash
pip install -r requirements.txt
cp .env.example .env        # Cloudflare judge credentials
python -m pytest -q tests
```
