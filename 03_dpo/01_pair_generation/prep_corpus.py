"""Step 1: documents on S3 -> filtered, de-duplicated chunks (chunks.jsonl) plus a report of what was dropped and why.

    python prep_corpus.py                  # reads s3://.../documents/, writes work/corpus/ and uploads it to OUT_ROOT/corpus/
    python prep_corpus.py --local-only     # no upload (e.g. to look at the report first)
    python prep_corpus.py --limit-docs 50  # quick try

Per document: strip the front-matter block (url, hop, finance_density ...), drop pages that are too short or off-topic,
drop near-duplicates (MinHash over 5-word shingles), then split into ~6,000-character chunks on paragraph boundaries
with a 400-character overlap. Every chunk starts with a "Source: <title>" line so a later chunk still says what it is.
"""
import argparse
import hashlib
import json
import re
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import config
import s3util

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
QUANTITY = re.compile(r"(?:₹|Rs\.?|\$|USD|INR|€)\s?\d[\d,\.]*|\d[\d,\.]*\s?(?:%|crore|lakh|million|billion|bn|mn|bps|x\b)", re.I)
THRESHOLDS = (30, 40, 50, 60, 70, 80)       # finance_density cut-offs shown in the report
NPERM, PRIME = 64, (1 << 31) - 1


# ---- documents ---------------------------------------------------------------------------------------

def _unescape(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except ValueError:
        return value


def parse_front_matter(text: str) -> tuple[dict, str]:
    """(metadata dict, body). A file without a front-matter block gives ({}, text)."""
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = _unescape(v.strip().strip('"'))
    return meta, text[m.end():]


def folder_of(key: str, prefix: str = config.DOCS_PREFIX) -> str:
    rel = key[len(prefix):] if key.startswith(prefix) else key
    return rel.split("/")[0] if "/" in rel else "(root)"


def document_keys(keys: list[str]) -> list[str]:
    """The markdown documents of the document pipeline: every .md except the tear sheets and the CIMs (their own prompts and judges)."""
    return [k for k in keys if k.endswith(".md") and folder_of(k) not in (config.TEARSHEET_FOLDER, config.CIM_FOLDER)]


def number(meta: dict, name: str, default: float = float("nan")) -> float:
    try:
        return float(meta.get(name, default))
    except (TypeError, ValueError):
        return default


def keep_document(folder: str, meta: dict, body: str) -> tuple[bool, str]:
    """(keep?, reason when dropped)."""
    if len(body.split()) < config.MIN_WORDS:
        return False, "too short"
    if folder == config.CIBIL_FOLDER:
        path_and_title = re.sub(r"^https?://[^/]+", "", meta.get("url", "")) + " " + meta.get("title", "")
        if config.CIBIL_EXCLUDE_TOPIC and config.CIBIL_EXCLUDE_TOPIC.search(path_and_title):
            return False, "cibil: off-topic (tax, stocks, mutual funds)"
        if len(body.split()) < config.CIBIL_SHORT_WORDS and config.CIBIL_EXCLUDE_IF_SHORT.search(path_and_title):
            return False, "cibil: short navigation or marketing page"
        return True, ""
    density = number(meta, "finance_density")
    if density == density and density < config.MIN_FINANCE_DENSITY:       # a missing value is kept, not dropped
        return False, "low finance_density"
    return True, ""


# ---- near-duplicates ---------------------------------------------------------------------------------

_rng = np.random.RandomState(1)
_A = _rng.randint(1, PRIME, size=NPERM, dtype=np.int64)
_B = _rng.randint(0, PRIME, size=NPERM, dtype=np.int64)


def signature(text: str) -> np.ndarray:
    """MinHash signature (NPERM values) of the 5-word shingles of a text."""
    words = re.findall(r"\w+", text.lower())
    shingles = {zlib.crc32(" ".join(words[i:i + 5]).encode()) for i in range(max(1, len(words) - 4))}
    h = np.fromiter(shingles, dtype=np.int64, count=len(shingles)) % PRIME
    return ((_A[:, None] * h[None, :] + _B[:, None]) % PRIME).min(axis=1)


def dedupe(docs: list[dict], threshold: float = config.NEAR_DUP_JACCARD) -> tuple[list[dict], list[dict]]:
    """(kept, duplicates). The longest copy of each near-identical group is kept; each duplicate records which one."""
    ordered = sorted(docs, key=lambda d: (-d["words"], d["key"]))
    kept, dups, sigs = [], [], np.empty((0, NPERM), dtype=np.int64)
    for d in ordered:
        sig = signature(d["body"])
        similarity = (sigs == sig).mean(axis=1) if len(sigs) else np.zeros(0)
        if len(sigs) and similarity.max() >= threshold:
            best = int(similarity.argmax())
            dups.append({**d, "reason": "near-duplicate", "duplicate_of": kept[best]["key"],
                         "similarity": round(float(similarity[best]), 2)})
            continue
        kept.append(d)
        sigs = np.vstack([sigs, sig])
    return kept, dups


# ---- chunking ----------------------------------------------------------------------------------------

def _split_long(paragraph: str, size: int) -> list[str]:
    """A paragraph longer than `size`, cut at sentence ends (or spaces) so no piece exceeds it."""
    pieces = []
    while len(paragraph) > size:
        cut = max(paragraph.rfind(". ", 0, size), paragraph.rfind("\n", 0, size))
        cut = cut + 1 if cut > size * 0.5 else paragraph.rfind(" ", 0, size)
        cut = cut if cut > size * 0.5 else size
        pieces.append(paragraph[:cut].strip())
        paragraph = paragraph[cut:].strip()
    return pieces + ([paragraph] if paragraph else [])


def chunk_text(body: str, size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP_CHARS,
               min_tail: int = config.MIN_TAIL_CHARS) -> list[str]:
    """Split on paragraph boundaries into chunks of about `size` characters. Each chunk after the first starts with the last
    `overlap` characters of the one before. A final chunk shorter than `min_tail` is merged into the previous one."""
    pieces = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", body)):
        if p:
            pieces += _split_long(p, size)
    chunks, fresh_of, cur, fresh = [], [], "", ""
    for p in pieces:
        if cur and len(cur) + 2 + len(p) > size:
            chunks.append(cur)
            fresh_of.append(fresh)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail      # start the overlap at a word boundary
            cur, fresh = (tail + "\n\n" + p) if tail else p, p
        else:
            cur, fresh = ((cur + "\n\n" + p) if cur else p), ((fresh + "\n\n" + p) if fresh else p)
    if cur:
        chunks.append(cur)
        fresh_of.append(fresh)
    if len(chunks) > 1 and len(chunks[-1]) < min_tail:
        chunks[-2] = chunks[-2] + "\n\n" + fresh_of[-1]
        chunks.pop()
    return chunks


def chunk_id(key: str, index: int) -> str:
    return hashlib.sha1(f"{key}|{index}".encode()).hexdigest()[:12]


def build_chunks(doc: dict) -> list[dict]:
    texts = chunk_text(doc["body"])
    header = f"Source: {doc['title']}\n\n" if doc["title"] else ""
    return [{"chunk_id": chunk_id(doc["key"], i), "source_key": doc["key"], "folder": doc["folder"], "title": doc["title"],
             "url": doc["url"], "chunk_index": i, "n_chunks": len(texts), "n_quantities": len(QUANTITY.findall(t)),
             "text": header + t} for i, t in enumerate(texts)]


# ---- run ---------------------------------------------------------------------------------------------

def fetch_all(s3, keys: list[str], workers: int = 32) -> dict[str, str]:
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(keys, pool.map(lambda k: s3util.get_text(s3, k), keys)))


def threshold_table(docs: list[dict]) -> dict:
    """For every folder except CIBIL: how many files each finance_density cut-off would remove."""
    table = defaultdict(lambda: {t: 0 for t in THRESHOLDS})
    for d in docs:
        if d["folder"] == config.CIBIL_FOLDER or d["finance_density"] != d["finance_density"]:
            continue
        for t in THRESHOLDS:
            table[d["folder"]][t] += d["finance_density"] < t
    return {f: dict(v) for f, v in table.items()}


def prepare(texts: dict[str, str]) -> dict:
    """texts: key -> raw file. Returns {"chunks": [...], "dropped": [...], "report": {...}}."""
    docs, dropped = [], []
    for key in sorted(texts):
        meta, body = parse_front_matter(texts[key])
        doc = {"key": key, "folder": folder_of(key), "title": meta.get("title", ""), "url": meta.get("url", ""),
               "finance_density": number(meta, "finance_density"), "words": len(body.split()), "body": body.strip()}
        ok, why = keep_document(doc["folder"], meta, body)
        (docs if ok else dropped).append(doc if ok else {**doc, "reason": why})
    thresholds = threshold_table(docs + [d for d in dropped if d["reason"] == "low finance_density"])
    listed_by_folder = Counter(d["folder"] for d in docs + dropped)
    kept, dups = dedupe(docs)
    chunks = [c for d in sorted(kept, key=lambda d: d["key"]) for c in build_chunks(d)]
    dropped += dups
    report = {"listed": len(texts), "kept_documents": len(kept), "chunks": len(chunks),
              "dropped_by_reason": dict(Counter(d["reason"] for d in dropped)),
              "per_folder": {f: {"listed": listed_by_folder[f], "kept": sum(d["folder"] == f for d in kept),
                                 "chunks": sum(c["folder"] == f for c in chunks)}
                             for f in sorted(listed_by_folder)},
              "finance_density_thresholds_files_removed": thresholds,
              "chunk_chars_median": int(np.median([len(c["text"]) for c in chunks])) if chunks else 0,
              "chunks_with_8plus_figures": sum(c["n_quantities"] >= config.MIN_QUANTITIES_FOR_NUMERIC for c in chunks)}
    return {"chunks": chunks, "dropped": [{k: v for k, v in d.items() if k != "body"} for d in dropped], "report": report}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.HERE / "work" / "corpus"))
    ap.add_argument("--local-only", action="store_true", help="do not upload to S3")
    ap.add_argument("--limit-docs", type=int, help="only the first N documents (quick try)")
    a = ap.parse_args()

    s3 = s3util.client()
    keys = document_keys(s3util.list_keys(s3, config.DOCS_PREFIX))
    if a.limit_docs:
        keys = keys[:a.limit_docs]
    print(f"[prep] {len(keys)} documents under s3://{config.BUCKET}/{config.DOCS_PREFIX}", flush=True)
    result = prepare(fetch_all(s3, keys))
    out = Path(a.out)
    write_jsonl(out / "chunks.jsonl", result["chunks"])
    write_jsonl(out / "dropped_docs.jsonl", result["dropped"])
    (out / "corpus_report.json").write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
    print(json.dumps(result["report"], indent=2))
    if not a.local_only:
        failed = s3util.upload_files([out / n for n in ("chunks.jsonl", "dropped_docs.jsonl", "corpus_report.json")],
                                     config.CORPUS_DIR, s3)
        print(f"[prep] uploaded to s3://{config.BUCKET}/{config.OUT_ROOT}{config.CORPUS_DIR}/" + (f" (failed: {failed})" if failed else ""))


if __name__ == "__main__":
    main()
