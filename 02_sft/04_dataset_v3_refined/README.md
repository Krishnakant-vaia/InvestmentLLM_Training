# 02.4 · SFT dataset v3 (refined)

Rebuilds the SFT dataset by selecting the best questions from the existing pool, then generating fresh responses.
The plan is in `docs/SFT_v3_Dataset_Rebuild_Plan.md`; the two stages are specified in `question_selection_spec.md`
and `response_generation_spec.md`.

```
run_question_selection.py   CLI: extract -> classify -> dedup -> select
selection_core.py           pure logic (parsing, scoring, selection), unit-tested
*_prompt.txt                classify, dedup, response and response-judge prompts
daytona_setup.sh            sandbox setup for running the selection on Daytona
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # SEL_BASE_URL / SEL_API_KEY / SEL_MODEL_ID, AWS
python run_question_selection.py extract  --out work/
python run_question_selection.py classify --out work/ --limit 1000   # drop --limit for the full run
python run_question_selection.py dedup    --out work/
python run_question_selection.py select   --out work/ --upload s3://bucket/prefix/
```

Each stage reads the previous one's files in `--out`; `classify` and `dedup` are resumable.

## Test

```bash
python -m pytest -q tests
```
