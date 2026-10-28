"""Settings and prompts for the VAIA SFT generation pipeline.

Single-machine version: one run covers all 8 capabilities. (The original run was
split across PCs as first_4_capabilities/ and last_4_capabilities/ copies that
differed only in CAPABILITY_GROUP; those have been merged into this one folder.)

Structure follows the original template/config.py; the differences from that
template are deliberate and specific to this run:
  - input comes from S3 (the filtration pipeline's output), not a local directory
  - GENERATION_INSTRUCTIONS is the full taxonomy-driven prompt (was qa_prompt.txt,
    now embedded here as the single source of truth)
  - the generation schema is {question_type, difficulty, section_context, question,
    thinking, answer}, with `thinking` pre-wrapped in <think>...</think> by the
    prompt itself
  - documents are matched by capability code (see CAPABILITY_GROUPS below);
    codes outside the taxonomy are skipped and reported in the run summary
"""
import os

from dotenv import load_dotenv

# override=True: if you bothered to write a value in .env, that value should win --
# not silently lose to some unrelated AWS_* var already sitting in whoever's shell
# happens to be running this (a real failure mode hit in practice: a stale/invalid
# AWS_ACCESS_KEY_ID left over in one operator's environment shadowed a fresh, valid
# key written to .env, since load_dotenv() does not override existing env vars by
# default).
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

CLOUDFLARE_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CLOUDFLARE_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]

# Only one model in this pipeline, on purpose: everything runs through Cloudflare
# Workers AI, and gpt-oss-120b is the largest gpt-oss size Cloudflare offers (there
# is no bigger gpt-oss variant to move up to). It's also the strongest open-weight
# model in our own VAIA-FRB benchmark results (5th overall, ahead of GPT-4.1), which
# is the whole reason it's the one generating the training data we'll fine-tune on.
# Verified against Cloudflare's own docs (developers.cloudflare.com/workers-ai/models/gpt-oss-120b/)
# that this exact slug is what the /ai/run/{model} path expects; also re-verified live,
# at run startup, against the account's real model catalog (see verify_model_available
# in sft_generation_core.py) rather than just trusted blind.
MODEL_ID = "@cf/openai/gpt-oss-120b"

# Cloudflare Workers AI concurrency/rate limits are per-account and per-model, not
# publicly fixed -- if you start seeing HTTP 429s or a neuron-cap error in the logs,
# lower this. Each worker holds one in-flight request; retries/backoff below handle
# transient failures without killing the whole run.
MAX_WORKERS = int(os.environ.get("SFT_MAX_WORKERS", "24"))
MAX_RETRIES = 4
RETRY_BASE_SECONDS = 2  # exponential backoff: 2, 4, 8, 16s (plus jitter)
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("SFT_REQUEST_TIMEOUT", "180"))

# Reasoning effort passed to gpt-oss-120b for generation. "medium" balances quality
# against cost/latency at this concurrency; bump to "high" for a smaller, higher-
# quality batch, or "low" for a fast/cheap first pass to sanity-check the pipeline.
REASONING_EFFORT = os.environ.get("SFT_REASONING_EFFORT", "medium")

# Output cap per call. Verify via the --limit smoke test that replies for this
# taxonomy-driven prompt (up to 5 pairs, 100-200 word thinking traces, answers that
# can include HTML tables) aren't getting truncated at this budget before committing
# to a full run -- bump via SFT_MAX_OUTPUT_TOKENS if the error log shows truncation.
MAX_OUTPUT_TOKENS = int(os.environ.get("SFT_MAX_OUTPUT_TOKENS", "6000"))

# --- chunking ---
# Character-based, not token-based, to avoid a tokenizer dependency here. gpt-oss-120b
# has a large context window, so these are conservative enough to leave plenty of room
# for the generation instructions and the model's own output.
CHUNK_CHARS = int(os.environ.get("SFT_CHUNK_CHARS", "6000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("SFT_CHUNK_OVERLAP_CHARS", "400"))

# --- S3 input ---
# Reads directly from the filtration pipeline's output -- no local copy of the
# corpus. S3 listing/fetching is I/O-bound and has nothing to do with Cloudflare's
# rate limits, so it gets its own worker pool, separate from MAX_WORKERS (which
# throttles calls to gpt-oss-120b).
AWS_REGION = os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("SFT_S3_BUCKET", "ft-vaia")
S3_PREFIX = os.environ.get("SFT_S3_PREFIX", "Corpus/Invest_dataset/082126_fliteredextractionrun2/")
S3_FETCH_WORKERS = int(os.environ.get("SFT_S3_FETCH_WORKERS", "64"))

# Documents are fetched and generated in batches rather than all at once, to bound
# peak memory to roughly one batch of documents at a time regardless of corpus size.
DOC_BATCH_SIZE = int(os.environ.get("SFT_DOC_BATCH_SIZE", "400"))

# Cap the chunks taken from any one document so a handful of outlier-sized documents
# can't dominate the dataset or stall a batch. 0 disables the cap.
MAX_CHUNKS_PER_DOC = int(os.environ.get("SFT_MAX_CHUNKS_PER_DOC", "60"))

# --- capability-based corpus split ---
# The corpus is organized under S3_PREFIX with each document's key starting with a
# capability folder named "<CODE>_<SLUG> - <readable name>/" (e.g.
# "AR_AUTONOMOUS_CLOSE_THE_LOOP_EXECUTION - Autonomous close-the-loop execution/").
# CODE is the taxonomy code from 260931_VAIA_GPT_Capability_Taxonomy_vSNG.docx's 8
# top-level capabilities. Splitting by capability (not an arbitrary hash) means each
# machine owns a stable, meaningful set of domains, and any new document added under
# an existing capability folder automatically lands on the right machine with no
# extra coordination.
#
# 4/4 split by category count (not verified per-capability document count -- no
# working S3 access was available when this was set). If real counts turn out
# lopsided, this is a one-line rebalance.
CAPABILITY_GROUPS = {
    # "WF", not "WU" -- corrected 2026-09-03 against the real corpus: the initial
    # "WU" guess (before any real S3 access was available) skipped 5,542 real
    # documents under an unrecognized code on the first live smoke test.
    # DK Domain Knowledge, EK Enterprise Knowledge, FR Financial Reasoning, WF Workflow Understanding,
    # CG Content Generation, AR Agentic Reasoning, CL Continuous Learning, EA Enterprise AI
    "all_capabilities": {"DK", "EK", "FR", "WF", "CG", "AR", "CL", "EA"},
}

CAPABILITY_GROUP = "all_capabilities"

# --- paths (local, resume source of truth) ---
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_PATH = os.environ.get("SFT_OUTPUT_PATH", os.path.join(_RESULTS_DIR, f"sft_data_{CAPABILITY_GROUP}.jsonl"))
LOG_PATH = os.environ.get("SFT_LOG_PATH", os.path.join(_RESULTS_DIR, f"sft_generation_errors_{CAPABILITY_GROUP}.jsonl"))
# Everything that's normally only printed to the console, also persisted here so
# a multi-day unattended run's history survives a closed terminal, a scrollback
# limit, or a reboot -- mirrored to S3 on the same cadence as OUTPUT_PATH/LOG_PATH.
RUN_LOG_PATH = os.environ.get("SFT_RUN_LOG_PATH", os.path.join(_RESULTS_DIR, f"run_{CAPABILITY_GROUP}.log"))

# --- pause/resume ---
# Presence of this file is a request to stop cleanly after the document currently
# in flight finishes (never mid-document) -- created externally (by whatever is
# operating this run) to request a pause, consumed (deleted) by the script itself
# the moment it acts on it, so the very next run always starts clean with no
# manual cleanup step. Resuming after a pause is nothing more than re-running the
# same command -- load_progress() already picks up exactly where a pause (or any
# other interruption) left off.
PAUSE_SENTINEL_PATH = os.environ.get("SFT_PAUSE_SENTINEL_PATH", os.path.join(_RESULTS_DIR, f"PAUSE_{CAPABILITY_GROUP}"))

# --- S3 output (durability mirror; local OUTPUT_PATH/LOG_PATH stay authoritative for resume) ---
S3_OUTPUT_PREFIX = os.environ.get(
    "SFT_S3_OUTPUT_PREFIX",
    "s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/",
)
# Full re-upload of the local file every N completed chunks, plus once more
# unconditionally at the end of the run (finished, --limit, or aborted). Tune once
# real per-chunk throughput and file growth rate are observed on a real run.
S3_UPLOAD_INTERVAL_CHUNKS = int(os.environ.get("SFT_S3_UPLOAD_INTERVAL_CHUNKS", "2000"))

# File types picked up from S3_PREFIX. Extend this if the team feeds in other
# extensions (e.g. .csv), but anything added here should be plain-text-readable --
# binary formats (.pdf, .docx) need to be converted to text upstream of this script.
INPUT_EXTENSIONS = {".txt", ".md"}

# System prompt for the GENERATOR (gpt-oss-120b), not for the model being trained.
# Trimmed to a one-line reinforcement: GENERATION_INSTRUCTIONS below already opens
# with its own ROLE block, so the persona/expertise framing lives entirely in the
# user-turn prompt, not duplicated here.
SYSTEM_PROMPT = "Respond with only the JSON array specified. No other text, no Markdown fences."

# The full taxonomy-driven generation prompt (previously qa_prompt.txt, now the
# single source of truth here). {chunk_text} is the only placeholder -- the model
# decides pair count (1-5) and whether to return [] itself; there is no {n} to fill.
GENERATION_INSTRUCTIONS = """\
ROLE: You are a senior financial analyst creating Qwen3 SFT data for a finance and sustainability LLM. Your expertise includes sustainable finance frameworks (SFDR, CSRD, TCFD, GRI, TNFD, PCAF, IRIS+/IMP), Indian financial markets and regulation (SEBI, RBI, NSE debt markets, CCIL), and private markets (PE/VC structures, waterfalls, due diligence, IC memos, and LP reporting). Write with precision, commercial judgment, and no filler.

<document>
{chunk_text}
</document>

TASK:
Generate 1–5 high-quality Q&A pairs. Use each question type at most once and only when the document genuinely supports a non-trivial question. If more than 5 types qualify, select the 5 with the strongest and most distinct evidence. Return [] if the document is incomplete, boilerplate, or lacks sufficient decision-useful information.

QUESTION TYPES:
- High-Order Reasoning: Form a thesis, evaluate the strongest evidence-backed counterargument, and resolve it using the document.
- Calculation/Numerical: Show the formula, substitution, every arithmetic step, units, and practitioner implication. Skip if required figures, periods, units, or denominators are missing.
- Error Identification: Identify a demonstrable flaw, bias, inconsistency, or unsupported assumption; explain its likely effect and prescribe an appropriate correction technique. Do not invent a flaw or name a technique without explaining why it fits.
- Applied/Scenario: Pose a realistic practitioner decision; give one concrete recommendation and one named risk or condition that could invalidate it.
- Multi-Concept Relationship: Name two distinct concepts, explain the mechanism linking them, and end with a practical implication.
- Interpretation & Analysis: Separate what the reported information shows from what the underlying pattern may signal, clearly marking inference.
- Comparative Ranking: Rank at least two comparable items using a stated criterion and evidence; end with the structural implication. Skip non-equivalent comparisons unless they can be reconciled.
- Conceptual/Tutorial: Define and explain a concept using one document-specific example and one genuine misconception or edge case.
- Summary: Cover all material finding categories present, such as methodology, headline results, subgroup effects, secondary outcomes, limitations, and implications.

DIFFICULTY:
- Basic: Summary, Conceptual/Tutorial
- Applied: Applied/Scenario, Multi-Concept Relationship, Interpretation & Analysis, Comparative Ranking
- Advanced: High-Order Reasoning, Calculation/Numerical, Error Identification

GROUNDING:
- Use the document as the sole factual source. Established formulas may be applied, but do not introduce external facts, thresholds, assumptions, or regulations.
- Distinguish reported facts, calculated values, and analytical inferences. Never present an inference as a reported fact.
- Preserve qualifications such as estimated, adjusted, projected, unaudited, or subject to approval.
- Attribute every figure using its entity, metric, period, unit, and context where available. Never invent citations.
- Across the set, use distinct findings and avoid duplicate or superficial variations of the same question.
- Questions must be self-contained, specific, answerable from the document, and must not reveal the answer.

THINKING:
For each pair, the `thinking` field must contain exactly one 100–200 word reasoning trace enclosed in `<think>...</think>`. In first-person analytical prose, it must:
- name the specific figure or claim used and explain why it is relevant;
- distinguish facts, calculations, assumptions, and inferences;
- identify one mistake a hasty analyst might make;
- state the non-obvious insight;
- remain fully grounded without repeating the final answer.

For Calculation/Numerical, include every arithmetic step. For High-Order Reasoning, evaluate the strongest counterargument. For Error Identification, connect the flaw to a suitable correction technique. Do not use headings, bullets, or multiple `<think>` blocks inside the trace.

ANSWERS:
Open with the conclusion, explain every material inference, cite figures with their source context, address relevant caveats, and stop when complete. Use prose unless a list improves clarity. Use HTML, not Markdown, for necessary tables.

OUTPUT:
Return only a valid JSON array. Each object must contain exactly:
{{
  "question_type": "...",
  "difficulty": "...",
  "section_context": "...",
  "question": "...",
  "thinking": "<think>...</think>",
  "answer": "..."
}}

Use the document’s `[Section]` header for `section_context` when present; otherwise describe the topic concisely. Use exact question-type and difficulty labels. Output valid JSON with double quotes, escaped characters, no Markdown fences, comments, trailing commas, nulls, or additional keys.

Before returning, silently verify grounding, numerical accuracy, non-duplication, type fit, difficulty, `<think>` formatting, and JSON validity.
"""

# System prompt baked into every TRAINING example (what the fine-tuned model sees
# at inference time). Unchanged from the earlier pipeline -- separate from the
# generator's own expertise framing above.
TRAINED_SYSTEM_PROMPT = "You are a finance and investment assistant."
