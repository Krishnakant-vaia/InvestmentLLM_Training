"""Settings for the DPO QA-pair generation pipeline. Everything tunable is here; secrets live in .env."""
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ. A value in .env REPLACES one inherited from the shell (a stale key left in
    the Windows environment would otherwise silently win); blank values are skipped."""
    path = path or HERE / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[key.strip()] = value


load_env()

# ---- S3 ------------------------------------------------------------------------------------------
BUCKET = "ft-vaia"
DOCS_PREFIX = "Corpus/Invest_dataset/100626_Dataset_DPO/documents/"
OUT_ROOT = os.environ.get("DPO_S3_OUT_PREFIX", "Corpus/Invest_dataset/100626_Dataset_DPO/qa_pairs/").rstrip("/") + "/"
# Layout under OUT_ROOT:  corpus/{chunks.jsonl, corpus_report.json, dropped_docs.jsonl}   (built once, shared by every run)
#                         <run>/{assignments, pairs_raw, calls_done, errors, judge_in, judge_out, selected, dpo_pairs, reports ...}
CORPUS_DIR = "corpus"

# ---- model (served by vLLM on the GPU sandbox) -------------------------------------------------------
MODEL_REPO = "openai/gpt-oss-120b"
SERVED_NAME = "gpt-oss-120b"
SERVER_PORT = 8000
SERVER_URL = os.environ.get("VLLM_URL", f"http://127.0.0.1:{SERVER_PORT}")
# Base `vllm serve` flags. The pipeline adds --max-num-seqs from the GPU it finds (capacity.py). To replace ALL flags yourself
# (e.g. add "--reasoning-parser openai_gptoss" if the smoke test shows the answer left in the reasoning field), set
# VLLM_SERVE_ARGS in .env.
VLLM_BASE_ARGS = "--max-model-len 49152 --gpu-memory-utilization 0.92"   # 48k covers a judge batch (<= ~37k) plus its reply; a smaller limit also starts on an 80 GB card
VLLM_ARGS_OVERRIDE = os.environ.get("VLLM_SERVE_ARGS")
# Environment for the vLLM process. The sandbox image has no CUDA compiler (nvcc), and vLLM's default FlashInfer sampler compiles a
# kernel at warm-up ("Could not find nvcc ..."), so the built-in PyTorch sampler is used instead.
VLLM_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
SERVER_START_TIMEOUT_S = 60 * 60          # first start downloads ~65 GB of weights

# ---- generation ------------------------------------------------------------------------------------
GEN_TEMPERATURE = 0.4
GEN_MAX_TOKENS = 8000                     # reasoning + the JSON; the SFT run used 6000 for one answer per pair
GEN_REASONING_EFFORT = os.environ.get("DPO_REASONING_EFFORT", "medium")
GEN_WORKERS = int(os.environ.get("DPO_WORKERS", "48"))   # only without a GPU to size from; on a sandbox the pipeline sets it from capacity.py
GEN_RETRIES = 4
GEN_TIMEOUT_S = 900
GEN_SYSTEM_PROMPT = "Respond with only the JSON array specified. No other text, no Markdown fences."
UPLOAD_EVERY_CALLS = 200                  # progress files go to S3 this often

JUDGE_WORKERS = 8                         # fallback; on a sandbox the pipeline sets it from capacity.py
JUDGE_REASONING_EFFORT = os.environ.get("DPO_JUDGE_EFFORT", "medium")   # low | medium | high; "low" is ~2x faster, check agreement first (agreement.py)
JUDGE_WORKERS_OVERRIDE = os.environ.get("DPO_JUDGE_WORKERS")   # concurrent judge calls; default is sized from the GPU (capacity.py)
JUDGE_PROMPT = HERE / "prompts" / "DPO_pair_judge_prompt.txt"
JUDGE_PROMPT_NUMERICAL = HERE / "prompts" / "DPO_pair_judge_prompt_numerical.txt"
JUDGE_BATCH = 10                          # pairs per judge call (each carries its document)
JUDGE_MAX_TOKENS = 6000
JUDGE_TEMPERATURE = 0.0

SEED = 3407

# ---- corpus filters and chunking ---------------------------------------------------------------------
CIBIL_FOLDER = "CIBIL Credit Scoring"
TEARSHEET_FOLDER = "tear sheets"        # documents/tear sheets/*.md: converted tear sheets, generated with their own prompt; the document pipeline skips them
MIN_WORDS = 250                           # shorter pages are navigation / boilerplate
MIN_FINANCE_DENSITY = 40                  # front-matter score; applied to every folder except CIBIL (see below)
# finance_density does not separate on-topic from off-topic CIBIL pages (tax pages score low, boilerplate pages score
# very high), so that folder is filtered on title and URL path instead, in two groups:
#   topic words (tax, stocks, mutual funds ...): CIBIL_EXCLUDE_TOPIC. Set to None on purpose: those 17 pages are KEPT (decision of
#     the data owner); put the regex back to drop them again.
#   page-type words (contact, support, subscription ...): navigation / marketing pages, only when the page is short.
#     A long page with such a word is real content (e.g. the 6,400-word "Contact Us FAQ's" on dispute handling).
CIBIL_EXCLUDE_TOPIC = None   # was: re.compile(r"tax|income-tax|itr|capital[-_ ]?gain|mutual[-_ ]?fund|dividend|stock|share-market|crypto|elss", re.I)
CIBIL_EXCLUDE_IF_SHORT = re.compile(
    r"subscription|contact|alerts?|select-plan|freecibil|blog-main|cibilrank|support|watch[-_ ]and[-_ ]learn|"
    r"new[-_ ]to[-_ ]credit|credit[-_ ]myths|saksham|nodal|choose", re.I)
CIBIL_SHORT_WORDS = 1500
NEAR_DUP_JACCARD = 0.85
CHUNK_CHARS = 6000                        # about 1,500 tokens: keeps a pair inside the 5,000-token side budget
CHUNK_OVERLAP_CHARS = 400
MIN_TAIL_CHARS = 800                      # a shorter last chunk is merged into the one before it

# ---- assignment (which question types and defects each chunk gets) ----------------------------------------
# Target share of each question type in this corpus. Definitional web articles: Drafting, Adjusted Metric and
# Waterfall/Distribution are not supported by this text, so they are off (absent). A chunk gets PASSES x TYPES_PER_PASS
# different types, so no type can exceed 1 / (PASSES x TYPES_PER_PASS) = 16.7% of all pairs (assign() checks this).
TYPE_MIX = {
    "Terminology": 0.16, "Short Answer": 0.16, "Interpretation & Analysis": 0.13, "Applied/Scenario": 0.13,
    "Comparative Ranking": 0.12, "High-Order Reasoning": 0.12, "Error Identification": 0.10,
    "Calculation/Numerical": 0.04, "Rating/Notching": 0.04,   # only chunks with figures qualify, so these land nearer 2-3%
}
NUMERIC_TYPES_NEED_FIGURES = {"Calculation/Numerical", "Adjusted Metric", "Rating/Notching", "Waterfall/Distribution"}
MIN_QUANTITIES_FOR_NUMERIC = 8            # a chunk needs this many figures before a numerical type may be assigned
# Target share of each defect over ALL pairs, balanced globally (a defect is only assigned to the question types that allow it).
# No defect may exceed MAX_DEFECT_SHARE. `padded` is Short Answer's counter-pair (short correct answer chosen, padded rejected).
# calculation_slip / no_workings only fit the two numerical types, so they land near 2% each whatever the target.
DEFECT_TARGETS = {
    "unsupported_claim": 0.11, "omitted_element": 0.12, "generic_content": 0.11, "overstated_certainty": 0.12,
    "single_driver": 0.11, "wrong_definition": 0.10, "off_by_one": 0.10, "padded": 0.12,
    "calculation_slip": 0.06, "no_workings": 0.05,
}
MAX_DEFECT_SHARE = 0.15
PASSES = 2                                # passes over every chunk; a type is never repeated for the same chunk
TYPES_PER_PASS = 3

# ---- content generation (DPO_content_generation_prompts.txt) -----------------------------------------------
# One layout line is drawn per call and passed to the prompt as {style}; both answers of a pair follow it, so style cancels inside the
# pair and varies across the dataset. The web-corpus prompt (qa_prompt.txt) has no {style}, so it ignores this.
STYLES = [
    "No headings; flowing prose; open with the main finding.",
    "No headings; flowing prose; open with the context, then the finding.",
    "Plain headings that use the question's own element names; open with the main finding.",
    "Plain headings that use the question's own element names; open with the context, then the finding.",
    "Numbered headings that use the question's own element names; short paragraphs.",
    "Numbered headings that use the question's own element names; open with the main finding.",
]
# ---- management-presentation set (DPO_content_generation_prompts_mgmt_presentation.txt) ----------------------
# Separate mix so the financial config above is untouched: assign.assign(..., **config.DECK_ASSIGN) (or `assign.py --deck`).
# One type pair per call (PASSES x TYPES_PER_PASS = 2), so each type is capped at 50% and each of 9 defects sits near 11%.
JUDGE_PROMPT_DECK = HERE / "prompts" / "DPO_pair_judge_prompt_deck.txt"
# ---- CIM set (DPO_content_generation_prompts_cim.txt) -------------------------------------------------------
# assign.assign(..., **config.CIM_ASSIGN). One pass of three types, so each type holds a third of the calls. The defects cover exactly
# what CIM Drafting, Adjusted Metric and Short Answer allow; none exceeds MAX_DEFECT_SHARE. omitted_element leads (completeness is the main
# gap); the basis defects together (headline_number, wrong_adjustment, wrong_definition, calculation_slip) carry the figures challenge.
JUDGE_PROMPT_CIM = HERE / "prompts" / "DPO_pair_judge_prompt_cim.txt"
CIM_TYPE_MIX = {"CIM Drafting": 1, "Adjusted Metric": 1, "Short Answer": 1}
CIM_DEFECT_TARGETS = {
    "omitted_element": 0.13, "advice_not_draft": 0.05, "generic_content": 0.10, "unsupported_claim": 0.09, "overstated_certainty": 0.07,
    "headline_number": 0.10, "wrong_adjustment": 0.07, "calculation_slip": 0.06, "wrong_definition": 0.08, "cross_section_conflict": 0.10,
    "no_workings": 0.05, "off_by_one": 0.05, "padded": 0.05,
}
CIM_FOLDER = "cims"
CIM_SECTION_MARK = "[SECTION "            # prep_cims.py opens every section of a chunk with this label
# The basis defects here mean actual vs estimate, historical vs projected CAGR, segment vs consolidated: a chunk can carry them when it
# shows FY...E years, a CAGR, a segment table or net debt, not only when it mentions adjustments (the CIMs never say "adjusted").
CIM_BASIS_WORDS = re.compile(r"FY\d{2,4}E|CAGR|net debt|leverage|segment|EBITDA", re.I)
CIM_PAIRS_PER_CHUNK = 5                   # at most; assign.assign_balanced (a type may repeat in a chunk with a different defect)
CIM_ASSIGN = {"per_chunk": CIM_PAIRS_PER_CHUNK, "type_mix": CIM_TYPE_MIX, "defect_targets": CIM_DEFECT_TARGETS, "styles": None,
              "adjustment_words": CIM_BASIS_WORDS, "cross_marker": CIM_SECTION_MARK}
GENERATION_PROMPT_CIM = HERE / "prompts" / "DPO_content_generation_prompts_cim.txt"
STAGE_LIMITS_CIM = {"smoke": 24, "pilot": 90, "full": None}      # chunks per run (24 chunks = up to 120 pairs)
# Planning figures for a CIM run (capacity.py estimates only; the hard limits are GEN_MAX_TOKENS and JUDGE_MAX_TOKENS above). The CIM
# prompt measures about 2,450 tokens (11,240 characters / 4.6), and a CIM Drafting call returns up to two 900-word answers, two 100-200-word
# traces and the evidence, about 3,500 tokens before reasoning. Replace both with measured values after the first smoke run.
CIM_TEMPLATE_TOKENS = 2450
CIM_OUT_TOKENS_PER_CALL = 3500
DECK_TYPE_MIX = {"Slide Drafting": 0.5, "Deck Outline": 0.5}
DECK_DEFECT_TARGETS = {
    "omitted_element": 0.12, "wrong_order": 0.08, "wrong_slide_type": 0.12, "overstuffed_slide": 0.12, "advice_not_draft": 0.11,
    "generic_content": 0.12, "unsupported_claim": 0.11, "off_by_one": 0.11, "overstated_certainty": 0.11,
}   # wrong_order only fits Deck Outline, so it lands near 8% whatever the target
DECK_STYLES = ["One block per slide: Slide n | Type: <type> | Headline: <headline> | Body: <items separated by semicolons>."]
DECK_ASSIGN = {"passes": 1, "per_pass": 2, "type_mix": DECK_TYPE_MIX, "defect_targets": DECK_DEFECT_TARGETS, "styles": DECK_STYLES}

# ---- tear-sheet set (DPO_content_generation_prompts_tearsheet.txt) -------------------------------------------
# One chunk per sheet; every compatible (type, defect) pair of Drafting, Adjusted Metric and Short Answer is asked once per sheet
# (assign.assign_grid): 9 + 5 + 4 = 18 pairs per sheet, one pair per call.
GENERATION_PROMPT_TEARSHEET = HERE / "prompts" / "DPO_content_generation_prompts_tearsheet.txt"
JUDGE_PROMPT_TEARSHEET = HERE / "prompts" / "DPO_pair_judge_prompt_tearsheet.txt"
TEARSHEET_TYPES = ["Drafting", "Adjusted Metric", "Short Answer"]
TEARSHEET_STYLES = [                      # layouts for Drafting (both answers of a pair follow the one drawn)
    "A short heading, then flowing paragraphs; open with the main finding.",
    "Plain headings that use the question's own element names; open with the main finding.",
    "Numbered headings that use the question's own element names; short paragraphs.",
    "A bulleted list, one bullet per element asked for, then a one-line conclusion.",
]
TEARSHEET_NEUTRAL_STYLE = "Follow the structure the question type calls for."
STAGE_LIMITS_TEARSHEET = {"smoke": 3, "pilot": 10, "full": None}    # sheets per run
TEARSHEET_ADJUSTMENT_EXEMPT = True        # the sheets are tear sheets, not credit files: no adjustment-word gating

# headline_number and wrong_adjustment are only assigned to a file that mentions adjustments (assign.py); otherwise they cannot apply.
ADJUSTMENT_DEFECTS = {"headline_number", "wrong_adjustment"}
ADJUSTMENT_WORDS = re.compile(r"add-?backs?|adjust|normalis|underwritten|pro[- ]forma|reported", re.I)

# ---- run-time planning (capacity.py) -------------------------------------------------------------------
TEMPLATE_TOKENS = 1975                    # generation prompt template, measured with the Qwen tokenizer
CHARS_PER_TOKEN = 4.6                     # measured on the real chunks
OUT_TOKENS_PER_CALL = 2400                # smoke test measured 2,045 (reasoning + up to 3 pairs); a little extra for the longer traces now asked
JUDGE_PASS_ESTIMATE = 0.9                 # share of pairs that survive the code gates and reach the judge (assumed)

# ---- Daytona -------------------------------------------------------------------------------------------
DAYTONA_GPU = "H200,H100"                 # in order of preference. H100: only ~9 long sequences fit beside the 65 GB of weights (capacity.py), ~4-5x slower: fine for a smoke test, too slow for the full run
DAYTONA_SPOT = True
DAYTONA_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
TTL_MINUTES = {"warm": 360, "smoke": 120, "pilot": 240, "full": 480, "calibrate": 120, "numerical": 480, "rewrite": 120}   # hard lifetime caps: the leak backstop
STAGE_LIMITS = {"smoke": 20, "pilot": 200, "full": None}  # chunks per run
