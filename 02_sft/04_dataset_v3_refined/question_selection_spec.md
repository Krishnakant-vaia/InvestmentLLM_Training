# Question Selection — Spec

**Goal:** select 140k standalone, high-quality questions from the existing SFT corpus, keeping each question's source file path, for response regeneration.

**Input:** `s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_kept.jsonl` — 647,066 rows, 74,401 source documents, 61 capability folders under 8 capabilities (DK, EK, FR, WF, CG, AR, CL, EA; see `260931_VAIA_GPT_Capability_Taxonomy_vSNG.docx`).

**Model:** `gpt-oss-120b` served by vLLM inside Daytona (OpenAI-compatible endpoint), prompts `question_selection_prompt.txt` and `question_dedup_prompt.txt`. No local embedding model.

## Steps

1. **Extract** per row: question (user text after `</document>`), `_source`, `_chunk_id`, `_question_type`, `_difficulty`, capability folder (4th segment of `_source`). Normalize label variants (non-breaking hyphens, case).
2. **Global exact dedup (code):** lowercase, strip punctuation and whitespace; keep the first occurrence.
3. **Classify, one question per call:** prompt `question_selection_prompt.txt`; the model returns `standalone`, `quality` (1-5), `depth` (1-3), `reason`. One question per call keeps scores independent; vLLM prefix caching reuses the shared prompt. Invalid replies are retried once, then logged.
4. **Dedup per source document:** only questions that pass step 6's filter, grouped by `_source` (up to 50 per call), go to `question_dedup_prompt.txt`, which returns `dup_of` per question. Sources with one passing question are skipped.
5. **Validate output (code):** enums and ids checked strictly; a failed call is retried once, then logged. Rerunning a stage retries only failed jobs.
6. **Filter:**
   - drop rows flagged as duplicates in step 4
   - keep `standalone = true` and `quality >= 4`
   - in capability folders with fewer than 2,000 questions, keep `quality >= 3`
7. **Template cap:** within each question type, no single opening template (first 5 normalized words) exceeds 15% of the kept rows; drop the lowest-ranked excess.
8. **Rank and cut:** sort by quality, then depth (both descending), take the top 140,000. Small-folder rows from step 6 are always kept.
9. **Save:** JSONL with question, `_source`, `_chunk_id`, `_question_type`, `_difficulty`, capability, quality, depth, reason, plus a stats file (counts by capability, type, reason, quality, depth).

## Decisions

- Document references in questions are allowed when the subject is clear from the question.
- No capability quotas; coverage of rare capabilities comes only from the small-folder rule.
- No few-shot examples and no hand-check step.
- Depth is used only as the tie-break in ranking, and is carried forward to choose answer length at regeneration.

## Open

- If fewer than 140k rows pass, the first lever to relax is the prompt's "if unsure, mark false" rule.
- Output location on S3 is not yet chosen.
