"""The VAIA eval bank (an .xlsx workbook) and its train-overlap check.

Workbook: one sheet per category, named "<n>. <Category>" (e.g. "1. Domain Knowledge"), each row a
question: QA_ID, Sub-Capability, Question Type, Context, Question, Options, Gold Answer, ...,
Answer Format, Difficulty, Scoring Rubric. Other sheets (README, Index, QA Lookup, Results Log,
Computation Map) are not questions and are skipped.

Barrier: `question_prompt` uses only Context, Question, Options and Answer Format. Gold answers and
rubrics never reach the model; they stay in the workbook for the judge (matched on QA_ID).
"""
import json
import re
from pathlib import Path

TRAINING_FILES = {"train.jsonl", "val.jsonl", "dpo_pairs.jsonl", "batch_plan.jsonl", "optimizer.pt", "adapter_model.safetensors"}
CATEGORY_SHEET = re.compile(r"^\d+\.\s")          # "1. Domain Knowledge" ... "8. Enterprise AI Deployment"
OVERLAP_N = 13                                   # word n-gram size for the overlap check (as in GPT-3's)
OVERLAP_MIN_WORDS = 8                            # shorter questions are matched whole instead
OVERLAP_FLAG = 0.5                               # flag a question if >= this share of its n-grams is in train


def barrier_violations(root) -> list[str]:
    """Training material under `root`: training/val data, the batch plan, checkpoints, optimizer state,
    adapters. An eval sandbox must hold none of it, so the eval job stops if this list is not empty."""
    root = Path(root)
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.name in TRAINING_FILES or (p.is_dir() and p.name.startswith(("checkpoint-", "final_adapter"))))


def _cell(v) -> str:
    """A spreadsheet cell as clean text ('' for empty / NaN)."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def load_bank(path) -> list[dict]:
    """Every question of the workbook, in sheet order, as plain dicts (all cells as text)."""
    import pandas as pd
    book = pd.ExcelFile(path)
    rows = []
    for sheet in book.sheet_names:
        if not CATEGORY_SHEET.match(sheet):
            continue
        for rec in book.parse(sheet).to_dict("records"):
            r = {k: _cell(v) for k, v in rec.items()}
            if r.get("QA_ID") and r.get("Question"):
                rows.append({**r, "Category": sheet})
    ids = [r["QA_ID"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate QA_IDs in the eval bank")
    return rows


def question_prompt(r: dict) -> str:
    """The user message for one question: context, question, options, answer format. Nothing else."""
    parts = []
    if r.get("Context"):
        parts.append(r["Context"])
    parts.append(r["Question"])
    if r.get("Options"):
        parts.append("Options: " + r["Options"])
    if r.get("Answer Format"):
        parts.append("Answer format: " + r["Answer Format"])
    return "\n\n".join(parts)


# ---- overlap with the training data -----------------------------------------------------------

def words(text: str) -> list[str]:
    """Lowercase alphanumeric words: case, punctuation and spacing do not hide a match."""
    return re.findall(r"[a-z0-9]+", text.lower())


def question_ngrams(r: dict) -> set[tuple]:
    """The n-grams that identify a question (Question + Context), or the whole short question."""
    out = set()
    for field in ("Question", "Context"):
        w = words(r.get(field, ""))
        if len(w) >= OVERLAP_N:
            out |= {tuple(w[i:i + OVERLAP_N]) for i in range(len(w) - OVERLAP_N + 1)}
        elif field == "Question" and len(w) >= OVERLAP_MIN_WORDS:
            out.add(tuple(w))
    return out


_WANTED: dict = {}                               # per worker process: n-gram length -> (n-grams, first words)


def _init_worker(wanted: dict) -> None:
    global _WANTED
    _WANTED = {n: (grams, {g[0] for g in grams}) for n, grams in wanted.items()}


def _scan(lines: list[str]) -> set[tuple]:
    """Which wanted n-grams occur in these training-file lines (SFT: "text"; DPO pairs: prompt + chosen + rejected)."""
    found = set()
    for line in lines:
        row = json.loads(line)
        w = words(row.get("text") or " ".join(str(row.get(k, "")) for k in ("prompt", "chosen", "rejected")))
        for n, (grams, starts) in _WANTED.items():
            for i in range(len(w) - n + 1):
                if w[i] in starts and tuple(w[i:i + n]) in grams:
                    found.add(tuple(w[i:i + n]))
    return found


def train_overlap(rows: list[dict], train_jsonl, workers: int = 32, batch: int = 2000) -> dict[str, dict]:
    """For every QA_ID: how many of its n-grams occur anywhere in the training file.

    Returns {QA_ID: {"matched": k, "total": t, "fraction": k/t, "flagged": bool}}. A report only:
    nothing is removed from the eval bank or added to training.
    """
    from multiprocessing import Pool
    per_q = {r["QA_ID"]: question_ngrams(r) for r in rows}
    wanted: dict[int, set] = {}
    for grams in per_q.values():
        for g in grams:
            wanted.setdefault(len(g), set()).add(g)
    found: set = set()
    with open(train_jsonl, encoding="utf-8") as f, Pool(workers, _init_worker, (wanted,)) as pool:
        chunk, jobs = [], []
        for line in f:
            chunk.append(line)
            if len(chunk) >= batch:
                jobs.append(pool.apply_async(_scan, (chunk,)))
                chunk = []
        if chunk:
            jobs.append(pool.apply_async(_scan, (chunk,)))
        for j in jobs:
            found |= j.get()
    report = {}
    for qid, grams in per_q.items():
        k = len(grams & found)
        frac = k / len(grams) if grams else 0.0
        report[qid] = {"matched": k, "total": len(grams), "fraction": round(frac, 3), "flagged": frac >= OVERLAP_FLAG}
    return report
