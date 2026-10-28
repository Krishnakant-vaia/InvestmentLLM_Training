# QA Quality Analysis Process — Architecture

## Purpose

The filtration pipeline (`qa_filter/`) decides ACCEPT/REJECT using an LLM judge with a
rubric prompt. This document describes a **separate, independent quality-analysis
process** that audits the *already-accepted* records (`sft_data_kept.jsonl` on S3) for
hidden defects the judge may have missed.

It is deliberately **not another LLM-judge-with-a-prompt** — running the same style of
check a second time would mostly re-detect the same blind spots, not find new ones. It
uses a different mechanism entirely: deterministic claim extraction plus embedding-based
semantic matching, both computed locally with no ongoing API cost.

## Non-goals

- Not a replacement for the filtration judge — this runs *after* filtration, on records
  already marked "keep."
- Not a second LLM judge, and does not reuse or resemble the judge's rubric prompt.
- Not a statistical/probabilistic risk score as the primary output — every per-record
  verdict is deterministic (same input always produces the same verdict) and carries
  concrete evidence a human can inspect, not a black-box probability.
- Not an ongoing-cost system — the one-time cost is downloading `sft_data_kept.jsonl`
  and running a local embedding model; there are no further API calls after that.

## Data sources

Reads directly from S3, not local `run_state/` (which is transient and gitignored):

- `s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_kept.jsonl` — the
  records under audit.
- `s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_rejected.jsonl` —
  read-only, for context (e.g. comparing flagged-kept patterns against real reject
  reasons), never for training anything.

Every record already carries `_judge_prompt_version` (which prompt/quality-bar accepted
it) and `_question_type`/`_difficulty` — all three become dimensions in the final report
without any extra extraction work.

## Core idea: sentence-level claim verification, not a black-box score

For every accepted record, the answer is split into sentences (plain sentence
splitting — no ML model involved in this step, kept deliberately simple and
deterministic). Each sentence is independently verified against the document using
**two different mechanisms depending on what the sentence contains**:

1. **Numbers, percentages, currency amounts, and dates** → exact or derivable matching.
   A claimed value passes if it appears verbatim in the document, or if it's a simple
   arithmetic derivation (sum, difference, percentage change) of two values that do
   appear, within a small rounding tolerance. Embeddings are the wrong tool for this —
   "150" and "151" can score nearly identical cosine similarity despite being a real
   numerical error, so numbers get exact/arithmetic checking instead.

2. **Everything else** (qualitative claims, causal statements, risk assessments,
   comparisons, recommendations) → embedding-based nearest-neighbor matching. The
   sentence and every document sentence are embedded with a local, free model
   (`sentence-transformers/all-MiniLM-L6-v2` — no API key, no per-call cost, runs on
   CPU). Cosine similarity is computed against every document sentence; the single
   highest-scoring document sentence is kept as the "best match," together with its
   score.

**The output is never a bare number.** A sentence fails only if its best match scores
below a fixed threshold (default 0.5, tunable — see Open Questions), and the failure
record includes the exact matching document sentence and its score, so a human reading
the report sees exactly what the system compared, not just a verdict to take on faith.

A record's overall verdict is `PASS` if every sentence passes both checks, `FLAGGED`
otherwise — deterministic and reproducible: rerunning the same record through the same
model version always produces the same verdict.

## Worked example

Record: document states "Revenue grew from ₹120 crore to ₹150 crore in FY25." Answer:
"Revenue grew by 30% in FY25, driven primarily by strong export demand."

Sentence 1 ("Revenue grew by 30%..."):
- Numeric claim: `30%`. Derivable check: (150-120)/120 = 25%, not 30% → **FAIL**,
  evidence: `claimed=30%, computed_from_document=25.0% (120→150 crore)`.

Sentence 2 ("...driven primarily by strong export demand"):
- No numbers; qualitative claim → embedding match. Best document sentence found:
  "Revenue grew from ₹120 crore to ₹150 crore in FY25" (no mention of export demand
  anywhere), score 0.31 → below threshold 0.5 → **FAIL**, evidence: `best_match="Revenue
  grew from ₹120 crore to ₹150 crore in FY25", score=0.31, claim="driven primarily by
  strong export demand"`.

Record verdict: **FLAGGED**, two independent, inspectable pieces of evidence — a
miscalculated percentage and an invented causal claim the document never makes.

## Components

```
qa_quality/
  __init__.py
  s3_io.py              # reuses qa_filter.s3_io.stream_jsonl_lines (already resumable)
  sentence_split.py      # deterministic sentence splitting for document + answer text
  numeric_extraction.py   # regex extraction of numbers/percentages/currency/dates
  numeric_verification.py  # exact-match + arithmetic-derivation check against document
  embeddings.py              # loads all-MiniLM-L6-v2 once, embeds a list of sentences
  semantic_verification.py    # nearest-neighbor matching + threshold decision
  record_verifier.py            # combines numeric + semantic checks into one record verdict
  report.py                      # aggregates per-record verdicts into a summary report
scripts/
  run_quality_analysis.py         # CLI: streams kept.jsonl from S3, runs record_verifier
                                   # on each record, writes flagged_records.jsonl +
                                   # quality_report.json, both uploaded back to S3
```

Reuses `qa_filter.s3_io.stream_jsonl_lines` as-is (already handles the 8.5GB-scale
resumable streaming problem) rather than reimplementing S3 access.

## Data flow

1. `run_quality_analysis.py` streams `sft_data_kept.jsonl` from S3, record by record.
2. Each record's document and answer are sentence-split.
3. Every answer sentence goes through `record_verifier`:
   - Numbers/dates extracted (`numeric_extraction`) and checked against the document
     (`numeric_verification`).
   - The sentence is embedded and matched against all document sentences
     (`embeddings` + `semantic_verification`).
   - Sentence-level result: PASS or FAIL with evidence.
4. Record-level verdict: PASS if all sentences pass, FLAGGED otherwise (with the list of
   failing sentences and their evidence attached).
5. Every FLAGGED record is written to `flagged_records.jsonl` (original record +
   evidence list). A running tally feeds `quality_report.json`.
6. At the end, `quality_report.json` and `flagged_records.jsonl` are uploaded to
   `s3://ft-vaia/.../quality_analysis/` — a new prefix, never overwriting the filtration
   pipeline's own output files.

## Output

**`quality_report.json`** — aggregate summary, entirely derived from the hard per-record
verdicts above (no separate statistical model):
```json
{
  "total_records_checked": 500000,
  "flagged_count": 12400,
  "flagged_pct": 2.48,
  "failure_reason_breakdown": {
    "numeric_mismatch": 7200,
    "unsupported_claim": 5600
  },
  "breakdown_by_question_type": { "...": "..." },
  "breakdown_by_judge_prompt_version": { "...": "..." }
}
```

**`flagged_records.jsonl`** — one line per flagged record: the original record plus an
`_evidence` list, each entry naming the failing sentence, the check that failed
(numeric/semantic), and the concrete evidence (computed value vs. claimed value, or best
document match + score).

Both breakdowns in the report (`breakdown_by_question_type`,
`breakdown_by_judge_prompt_version`) are plain aggregation over the hard verdicts — not a
second modeling step — so every number in the report traces back to an inspectable
per-record, per-sentence fact.

## Testing strategy

Following the existing project's TDD pattern (`qa_filter/tests/`):
- `numeric_extraction`: unit tests on known strings (percentages, crore/lakh amounts,
  dates) confirming exact regex extraction.
- `numeric_verification`: unit tests with a small synthetic document + claimed values,
  covering exact match, valid derivation, invalid derivation, and value absent entirely.
- `semantic_verification`: unit tests with a tiny fixed set of sentences and a mocked/
  deterministic embedding function (no real model download in unit tests), confirming
  correct nearest-neighbor selection and threshold behavior.
- `record_verifier`: integration-style tests combining both checks on a full synthetic
  record, matching the worked example above.
- One real-data smoke test (opt-in, like the filtration pipeline's own `run_smoke_test.py`
  pattern): run against a few hundred real accepted records and manually spot-check the
  flagged output.

## Open questions (to resolve before/while implementing)

1. **Similarity threshold (default proposed: 0.5).** Needs calibration — pull a small
   sample of records around the threshold boundary and manually check whether 0.5 is
   catching real issues or over/under-flagging paraphrased-but-correct claims.
2. **Currency/unit normalization for numeric matching** (crore vs. lakh vs. million vs.
   billion) — needs a conversion table so "₹15,000 crore" and "₹150 billion" can be
   recognized as the same value when comparing claim vs. document.
3. **Rounding tolerance for arithmetic derivation** — how much drift counts as "the same
   number" (e.g. 24.9% vs. 25.0% from real-world rounding) vs. a genuine error.
4. **Sentence-splitting edge cases** — financial documents often have list items, table
   fragments, and abbreviations (e.g. "Rs.", "FY25E") that a naive splitter could break
   incorrectly; needs a review pass against real document samples.
