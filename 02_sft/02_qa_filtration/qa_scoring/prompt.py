import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = "prompts/VAIA_Financial_QA_Quality_Analysis_GEPA_Strict_run_2.txt"
PLACEHOLDER = "{{batch_records_jsonl}}"
VERSION_PREFIX = "gepa_quality_run2-"

CATEGORY_CODES = frozenset("CATEGORY_" + c for c in [
    "FINANCIAL_STATEMENT_ANALYSIS",
    "ACCOUNTING_STANDARDS_AND_POLICY",
    "VALUATION_AND_DEAL_MODELING",
    "WORKING_CAPITAL_AND_DEBT_SCHEDULES",
    "DEBT_INSTRUMENTS_AND_CAPITAL_STRUCTURE",
    "COVENANT_COMPLIANCE_AND_CREDIT_AGREEMENTS",
    "CREDIT_RATING_AND_RISK_ASSESSMENT",
    "COLLATERAL_SECURITY_AND_LTV_ANALYSIS",
    "REGULATORY_CAPITAL_AND_PRUDENTIAL_METRICS",
    "DEAL_STRUCTURING_AND_TRANSACTION_TERMS",
    "REGULATORY_COMPLIANCE_AND_DISCLOSURE_REPORTING",
    "ESG_AND_SUSTAINABILITY_DISCLOSURE",
    "ENTERPRISE_AND_RELATIONSHIP_KNOWLEDGE",
    "RESTRUCTURING_DEFAULT_AND_RECOVERY",
    "GENERAL_FINANCIAL_REASONING_AND_RATIOS",
])

FAILURE_CODES = frozenset([
    "F01_UNANSWERABLE", "F02_AMBIGUOUS_QUESTION", "F03_FALSE_PREMISE",
    "F04_UNSUPPORTED_CLAIM", "F05_CONTRADICTED_CLAIM", "F06_NUMERICAL_ERROR",
    "F07_UNIT_CURRENCY_ERROR", "F08_PERIOD_SCOPE_ERROR", "F09_ROUNDING_PRECISION_ERROR",
    "F10_INCOMPLETE", "F11_SOURCE_CONFLICT", "F12_OVERSTATED_CERTAINTY",
    "F13_BAD_INFERENCE", "F14_NOT_TRAINABLE", "F15_INJECTION_OR_LEAKAGE",
    "F16_OUTPUT_DEFECT",
])


def load_prompt(path: str | None = None) -> tuple[str, str]:
    """Return (prompt text, version). The version hashes the newline-normalised
    text, so editing the prompt changes it and CRLF/LF checkouts agree."""
    chosen = path or os.environ.get("QUALITY_PROMPT_PATH") or DEFAULT_PROMPT_PATH
    p = Path(chosen)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / chosen
    text = p.read_text(encoding="utf-8")  # universal newlines -> LF
    if PLACEHOLDER not in text:
        raise ValueError(f"prompt {p} has no {PLACEHOLDER} placeholder")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return text, VERSION_PREFIX + digest
