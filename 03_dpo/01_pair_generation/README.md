# DPO QA-pair generation

Builds `{prompt, chosen, rejected, source}` preference pairs from the documents in
`s3://ft-vaia/Corpus/Invest_dataset/100626_Dataset_DPO/documents/`, using gpt-oss-120b served by vLLM on a Daytona GPU sandbox.
Prompts (generation, judge, rewrite, system prompt) live in `prompts/`; file names below refer to that folder.
Output (every run): `s3://ft-vaia/Corpus/Invest_dataset/100626_Dataset_DPO/qa_pairs/<run>/` (JSONL).

## Setup
1. Paste credentials into `.env` (AWS, Daytona, optional HF token). `.env.example` lists every variable.
2. `pip install daytona boto3 requests numpy` locally (the sandbox installs vLLM itself).

## Run (always in this order)
```
python daytona_run.py smoke     # 20 chunks, whole path: proves vLLM, the reply format, S3 writes
python daytona_run.py pilot     # 200 chunks: read first_response.json and ~10 pairs by hand; check pass rates in the log
python daytona_run.py full      # every chunk (~1,270)
python daytona_run.py full --run NAME --resume      # continue after an interruption (progress is on S3)
python daytona_run.py attach --run NAME             # re-watch a running job
```

## What each step does
| Script | Step |
|---|---|
| `prep_corpus.py` | strip front matter, filter off-topic/short/duplicate pages, chunk (6,000 chars, 400 overlap) -> `corpus/chunks.jsonl` |
| `assign.py` | code assigns each chunk its question types and defects (balanced shares) -> `assignments.jsonl` |
| `generate.py` | one gpt-oss-120b call per assignment, using `qa_prompt.txt` -> `pairs_raw.jsonl` (resumable) |
| `score_pairs.py build` | random a/b sides -> `judge_in.jsonl`; code checks become flags (nothing is dropped) |
| `judge.py` | judge both answers with `DPO_pair_judge_prompt.txt` -> `judge_out.jsonl` |
| `score_pairs.py select` | attach the judge's scores and flags to EVERY pair -> `scored_pairs.jsonl` (nothing removed, no combined score or rank) |
| `build_dpo_pairs.py` | trainer format (system + document + question as the prompt) and the bias report -> `dpo_pairs.jsonl` (every pair, with `passes_all_gates`, `flags`, judge scores), `report.json` |
| `make_training_set.py` | labels every scored pair SEND / HOLD / EXCLUDE and appends the SEND pairs to `qa_pairs/training/dpo_train.jsonl` (the file the trainer reads) and its `manifest.json`; a run is added once |
| `pipeline.py` | runs all of the above on the sandbox; sizes the vLLM server and workers from the GPU it finds, prints the expected run time, then the measured MFU/MBU; `daytona_run.py` creates the sandbox and starts it |
| `capacity.py` | tokens/s, MFU and run-time model for gpt-oss-120b on H100/H200 (`python capacity.py`) |
| `agreement.py` | second-opinion judge: `judge.py --url/--model/--api-key-env` on another endpoint, then `python agreement.py a.jsonl b.jsonl` |

## Numerical pairs (the Excel sets)
The question/preferred/not-preferred pairs in `DPO_Training/Finance_Numerical_QA_Dataset_Merged.xlsx` (7 sheets: Domain Knowledge,
Enterprise Knowledge, Financial Reasoning, Workflow Understanding, Content Generation, Agentic Reasoning, Continuous Learning) go through the
same filter-and-score path, capability by capability, with a numerical judge prompt. There is no document: the question's own figures are the
evidence and the golden answer is passed to the judge as `reference_solution`.

| Script / file | Step |
|---|---|
| `append_new_sheets.py` | appends the Workflow, Content, Agentic and Continuous Learning workbooks to the merged Excel (backup copy first; generated Pair IDs `WU-4.1-0001`, `CG-`, `AR-`, `CL-`; Priority and Task Type from the Index sheet; Question Type "Not labelled") and cleans the not-preferred answers (see below) |
| `prep_numerical.py` | Excel -> one JSONL per capability under `work/numerical/`, removes duplicate questions (first copy kept; the removed rows are saved), writes `capabilities.json` |
| `DPO_pair_judge_prompt_numerical.txt` | the numerical judge: scores each answer against the question and the reference solution; same gates, 1-10 score, reason codes and JSON as the document judge; flags a golden answer that is itself wrong (G1 fails for both) |
| `numerical_run.py` | on the GPU box: per capability, judge both answers (`judge.py --prompt`), checkpoint to S3, write `judged_pairs.jsonl`; `--sheets 4,5` runs only some sheets, `--per-capability 4` is the calibration |
| `make_training_set.py --numerical --run numerical-N` | labels SEND / HOLD / EXCLUDE and appends the SEND pairs to the training set (no `<think>` block, no document) |

Launch (same warm-sandbox pattern as above; use `PYTHONIOENCODING=utf-8` on Windows):
```
python daytona_run.py warm --gpu H200 --run NAME-warm                       # ~20-25 min, then it waits
python daytona_run.py calibrate --warm NAME-warm --run numerical-calibN --sheets 6,7    # 4 pairs per capability
python daytona_run.py numerical --warm NAME-warm --run numerical-N --sheets 6,7
python daytona_run.py release --run NAME-warm                               # when done: stop the billing
```
Server settings for these short pairs: 128 concurrent sequences, 100 workers. Measured 130-260 pairs/min on an H200; 10 pairs per judge call
(a batch that returns bad JSON is split and retried by `judge.py`).

Cleaning at append time (Workflow file: "Working (flawed):" label, "Next step in workflow" line; both files: bracketed notes that explain the
mistake). Sheets 1-3 were NOT cleaned at the source; the DPO trainer's data layer (`../02_training/pairs/clean.py`) removes remarks from both
sides when it loads the file.

### Results so far
| Run | Sheets | Pairs judged | SEND | HOLD | EXCLUDE |
|---|---|---|---|---|---|
| `full-1` (documents) | | 7,757 | 3,780 | | |
| `numerical-1` | 1-3 | 8,504 | 5,704 | 2,694 | 106 |
| `numerical-2` | 4-5 | 6,200 | 3,147 | 2,992 | 61 |
| `numerical-3` | 6-7 | 3,750 | 398 | 3,283 | 69 |

Training set `qa_pairs/training/dpo_train.jsonl`: **13,029 pairs** (3,780 + 5,704 + 3,147 + 398), `manifest.json` beside it. HOLD is mostly "the
answers differ by more than 2x in length" (the trainer drops those), EXCLUDE is a preferred answer the judge fails (mostly wrong golden answers).
9,249 of the 13,029 have no `<think>` block; the preferred answer is the longer one in 79%.

## One vocabulary: preferred / not_preferred (since 2026-10-21)
Every pair file now uses the same names: `preferred`, `not_preferred`, `preferred_thinking`, `not_preferred_thinking`, `judge_preferred`,
`judge_not_preferred`, `preferred_side`. Before, the document runs and the training file used `chosen` / `rejected` (+ `chosen_judge`...) and the
numerical Excel sets used `accepted` / `rejected`. `schema.py` holds the mapping (and the flag texts that used the old words, such as
"judge: chosen answer fails" -> "judge: preferred answer fails"); every reader runs old files through `normalize_row`, every writer emits the new names.
`migrate_names.py` renamed the 203 pair files on S3 IN PLACE (training file, document `pairs_raw` x2 and `scored_pairs`, 130 `judged_pairs`,
the `numerical-input` files): `backup` (to `DPO_Training/_backup_before_rename/`, 359 MB), `stage` (renames and PROVES every file reverses to its
original rows exactly; staged copies in `DPO_Training/_staged_renamed/`), `apply` (overwrites S3 only if nothing changed since the backup). The bucket is
versioned, so the earlier versions are still on S3. Not renamed: the rewrite results (their `accepted` is a yes/no outcome), report.json files, the
nested `numeric_check.rejected_value` key of old document pairs (read as `not_preferred_value`), `numerical-input/capabilities.json` token keys, and your Excel.
The trainer (`../sft_common/dpo_data.py`) requires `preferred` / `not_preferred`; its internal token keys (`chosen_ids`) and wandb metric names (`logps/chosen`) are unchanged.

## Tear sheets (documents for the tear-sheet QA pairs)
60 tear-sheet PDFs (30 startup sheets + 30 sector sheets in `content generation/tear sheets/` and its `sectors/` sub-folder; `Tear sheet example 1.pdf` is a
template and is skipped) were converted to markdown with `convert_tearsheets.py` (LlamaParse **v2** Parse API through `requests`; keys `llamaparse1..33` in `.env`,
a key answering 401/402/403 is dropped and the next used, raw replies cached in `work/tearsheets_raw/` so a re-run costs no credits). Checks and cleanup:
every figure of the PDF's own text layer must be in the markdown and vice versa (60/60 pass); a markdown table with a figure the PDF text lacks is a chart
the converter read from a picture (the sheets' headcount donut shows no values) and is replaced by a one-line note (35 sheets); the `Trend` column (icons of tiny
charts), stray `[signature]` lines, doubled bold markers and the page banner repeated per page are removed. Output: `s3://ft-vaia/Corpus/Invest_dataset/100626_Dataset_DPO/documents/tear sheets/<NN_Name_Tear_Sheet>.md`
(60 files, only `.md`). `prep_corpus.document_keys` skips that folder, so the document pipeline never picks the sheets up. Prompts: `DPO_content_generation_prompts_tearsheet.txt`,
`DPO_pair_judge_prompt_tearsheet.txt`. Next: adapt the generator and run on Daytona (smoke test on 3 sheets, ~20 pairs per sheet).

## Hard-negative rewrite (sample stage)
Surface-tell audit (`../02_training/pairs/audit.py`) found the not-preferred answers separable from the preferred ones by text alone (93.9% of
pairs, against 77.6% for "pick the longer"; 88.1% on length-matched pairs): they are shorter, skip steps, or explain their own mistake. So the
plan is to keep the mistake and rewrite the not-preferred answer into the preferred answer's form.
- `DPO_hard_negative_rewrite_prompt.txt` + `rewrite_run.py`: `make-sample` (100 pairs: 10 per source/sheet), `run` (on the GPU box: `daytona_run.py rewrite --warm NAME --run rewrite-N`), `readable` (before/after JSONL).
- Each rewrite is checked: no admission words, length 0.8x-1.25x of the preferred, same number of lines, a different final answer, the judge still passes the preferred and now FAILS the rewrite, and a model check that it does not describe its own mistake. Up to 3 attempts, each told what was rejected.
- Round 1: 72 of 100 accepted. Round 2 (target word count, retry feedback, natural-reasoning rule, tell check): **82 of 100** (documents 21/30, numerical 61/70). Files: `DPO_Training/rewrite_sample_100_before_after.jsonl`, `rewrite_sample_100_round2.jsonl`; S3 `qa_pairs/rewrite-1/`, `rewrite-2/`.
- Not done: the full run over the 13,029 SEND pairs, and putting the accepted rewrites into a training file. The self-check is the same model judging its own output.

## S3 layout (`.../100626_Dataset_DPO/qa_pairs/`)
`corpus/` | `full-1/{generation,judge,final,logs}/` | `numerical-input/` (per-capability JSONL + `capabilities.json` + removed duplicates) |
`numerical-N/<sheet>/<capability>/{judge/,final/judged_pairs.jsonl}` + `numerical-N/final/report.json` | `rewrite-input/sample.jsonl`, `rewrite-N/final/` |
`training/{dpo_train.jsonl,manifest.json}`. Bucket versioning is on: deleted files still occupy storage.

## GPU
Use an **H200**. gpt-oss-120b is ~65 GB, so on an 80 GB H100 only ~9 long sequences fit beside it and the full job takes ~10-15 h (over the sandbox cap); an H200 fits ~190 and takes ~2-3 h. Estimates assume 50% memory-bandwidth use and 6,000 output tokens per call: the pilot prints the measured values.

## Tests
`python -m pytest tests -q` (CPU only; includes the whole path against a fake OpenAI-compatible server and a check with the real Qwen3.5 tokenizer).

## Notes from running on real services
vLLM serving gpt-oss-120b, the Daytona launcher and S3 writes are all exercised (document run, three numerical runs, two rewrite rounds). Needs
`VLLM_USE_FLASHINFER_SAMPLER=0` (the image has no nvcc). `reasoning_effort` only works as a top-level request field; low effort changes the
judge's verdicts too much, so the judge runs at the default (medium). On Windows, set `PYTHONIOENCODING=utf-8` for the launcher: its progress
output otherwise crashes the local script (the sandbox keeps running; check it with `daytona_run.py attach`).
If the JSON arrives empty because the answer stays in the reasoning field, set `VLLM_SERVE_ARGS` in `.env` to add `--reasoning-parser openai_gptoss` (name per your vLLM version).

### Tear-sheet generator (`--variant tearsheet`)
`python daytona_run.py --job smoke --variant tearsheet` (or `pipeline.py --stage smoke --variant tearsheet`): one chunk per sheet
(`prep_tearsheets.py`, from `documents/tear sheets/`), `assign.assign_grid` gives every sheet each compatible type x defect once
(Drafting 9 + Adjusted Metric 5 + Short Answer 4 = 18 pairs, one per call, 1,080 over 60 sheets), generated with
`DPO_content_generation_prompts_tearsheet.txt` and judged with `DPO_pair_judge_prompt_tearsheet.txt`. Smoke = 3 sheets (54 pairs),
pilot = 10, full = all 60 (`config.STAGE_LIMITS_TEARSHEET`). Source label in the pairs: `tear_sheets`.
