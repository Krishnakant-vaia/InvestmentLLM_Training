import re
from datasketch import MinHash

_REFUSAL_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bas an ai\b",
    r"\bi cannot (?:verify|answer|determine)\b",
    r"\bi'?m not able to\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def extract_final_answer(assistant_content: str) -> str:
    if "</think>" not in assistant_content:
        raise ValueError("assistant content has no closing </think> tag")
    return assistant_content.split("</think>", 1)[1].strip()


def is_refusal_or_empty(final_answer: str) -> bool:
    stripped = final_answer.strip()
    if len(stripped) < 3:
        return True
    return bool(_REFUSAL_RE.search(stripped))


def is_structurally_valid(record: dict) -> tuple[bool, str]:
    user_content = record["messages"][1]["content"]
    assistant_content = record["messages"][2]["content"]

    if "<document>" not in user_content or "</document>" not in user_content:
        return False, "missing_document_tag"
    if "<think>" not in assistant_content or "</think>" not in assistant_content:
        return False, "missing_think_tag"
    return True, ""


def _question_shingles(question: str, k: int = 5) -> set:
    tokens = question.lower().split()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _minhash_for(question: str) -> MinHash:
    mh = MinHash(num_perm=64)
    for shingle in _question_shingles(question):
        mh.update(shingle.encode("utf-8"))
    return mh


def _question_text(record: dict) -> str:
    user_content = record["messages"][1]["content"]
    if "</document>" in user_content:
        return user_content.split("</document>", 1)[1].strip()
    return user_content


class PreFilter:
    def __init__(self, dedup_jaccard_threshold: float = 0.9):
        self._threshold = dedup_jaccard_threshold

    def filter(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        kept: list[dict] = []
        rejected: list[dict] = []
        seen_minhashes: list[MinHash] = []

        for record in records:
            valid, reason = is_structurally_valid(record)
            if not valid:
                rejected.append({**record, "_reject_reason": reason})
                continue

            final_answer = extract_final_answer(record["messages"][2]["content"])
            if is_refusal_or_empty(final_answer):
                rejected.append({**record, "_reject_reason": "refusal_or_empty_answer"})
                continue

            mh = _minhash_for(_question_text(record))
            is_dup = any(mh.jaccard(seen) >= self._threshold for seen in seen_minhashes)
            if is_dup:
                rejected.append({**record, "_reject_reason": "near_duplicate_question"})
                continue

            seen_minhashes.append(mh)
            kept.append(record)

        return kept, rejected
