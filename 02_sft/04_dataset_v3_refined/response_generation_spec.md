# Response Generation — Spec

**Goal:** regenerate the answer and reasoning for the ~140k selected questions with gpt-oss-120b (vLLM in Daytona), using `qa_response_prompt.txt`.

**Input:** `selected_questions.jsonl` from question selection (question, `_source`, `_chunk_id`, type, capability, quality, depth). Question selection itself sends only the question text, never the document.

## Document window (code fills `{document}`)

The document is fetched from S3 by `_source` and re-chunked exactly as the original generator did (6,000-character chunks, 400-character overlap, at most 60 chunks), so `_chunk_id` locates the source chunk.

- **Whole document** if it fits the 31.5k-token budget below.
- **Otherwise, in document order:**
  1. **Head, ~3k tokens:** the opening chunks (title, publisher, date, abstract, definitions), so the model can name the source in Sources and references.
  2. **Before, 13k tokens:** the chunks immediately before the source chunk.
  3. **Source chunk:** the chunk the question was written from, always included.
  4. **After, 14k tokens:** the chunks immediately after the source chunk.
- **Unused tokens move to the other side.** If fewer than 13k tokens exist before the source chunk, the remainder goes after it, and the other way round.
- **No double counting.** If the source chunk is near the start, the head and the before-window overlap; each chunk is included once and the saved tokens go to the after-window.
- **Gaps are marked** with `[…]` where text is skipped.

## Request

- Messages: system = response prompt; user = `<document>…</document>` + `<question>…</question>`.
- Tools: `tavily_search`, at most one call per question (enforced in code).
- `reasoning_effort: medium`, `temperature: 0.4`, `max_tokens: 12000`.
- Per call: ~1.7k prompt + ≤31.5k document + question, plus ≤12k output → vLLM `--max-model-len 49152`.

## Output record

question, `_source`, `_chunk_id`, `_question_type`, type, reasoning (gpt-oss native reasoning, kept as the training thinking), answer, references, search query (or null), answer and reasoning token counts.

## Open

- Pilot to measure the answer-token median (target 4–5k), throughput, and whether native reasoning follows the five stages well enough without a cleanup pass.
- The judge prompt still expects a written `<think>` block and must be updated to score native reasoning.
