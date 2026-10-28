"""Pure logic for question selection: no network, no filesystem.

Stages that share this module:
  extract_row      one corpus record -> one question row (or None)
  parse_label      raw model text -> validated label for one question
  dedup_batches    labelled rows -> per-source batches of passing questions
  parse_dedup      raw model text -> dup_of for one dedup batch
  select           labelled rows + duplicates -> the final kept set plus stats
"""
import json
import math
import re
from collections import Counter, defaultdict

REASONS_TRUE = {"OK", "TRIVIAL", "AMBIGUOUS", "OFF_DOMAIN", "MALFORMED"}
REASONS_FALSE = {"DOC_ONLY_SUBJECT", "UNDEFINED_REFERENT", "UNIDENTIFIED_FIGURE"}
REASONS = REASONS_TRUE | REASONS_FALSE

_CAPABILITY_SEGMENT = re.compile(r"^([A-Z]{2}_[A-Z0-9_]+)")
_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize_label(value):
    """'Multi‑Concept Relationship' (non-breaking hyphen) and 'advanced' fold onto
    the canonical spelling so counts and caps are not split across variants."""
    if not isinstance(value, str):
        return None
    v = value.replace("‑", "-").replace("‐", "-").strip()
    return v[:1].upper() + v[1:] if v else None


def normalize_question(q):
    """Key for global exact dedup: case, punctuation and spacing ignored."""
    q = q.replace("‑", "-").lower()
    return _SPACE.sub(" ", _PUNCT.sub(" ", q)).strip()


def template_key(q, n_words=5):
    return " ".join(normalize_question(q).split()[:n_words])


def capability_folder(source):
    """'Corpus/.../DK_ACCOUNTING_STANDARDS - desc/text/doc.md' -> 'DK_ACCOUNTING_STANDARDS'."""
    for seg in (source or "").split("/"):
        m = _CAPABILITY_SEGMENT.match(seg)
        if m:
            return m.group(1)
    return None


def extract_row(record):
    """One corpus JSONL record -> question row, or None if it has no usable question."""
    msgs = record.get("messages") or []
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    _, sep, q = user.rpartition("</document>")
    q = (q if sep else user).strip()
    if not q:
        return None
    source = record.get("_source")
    folder = capability_folder(source)
    return {
        "question": q,
        "_source": source,
        "_chunk_id": record.get("_chunk_id"),
        "_question_type": normalize_label(record.get("_question_type")),
        "_difficulty": normalize_label(record.get("_difficulty")),
        "capability_folder": folder,
        "capability": folder.split("_", 1)[0] if folder else None,
    }


def _json_object(raw):
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(raw[start:end + 1])


# --- per-question classification ---

def render_prompt(template, question):
    return template.replace("{{question}}", question)


def parse_label(raw):
    """Raw model text -> {standalone, quality, depth, reason}. Raises ValueError on any defect."""
    r = _json_object(raw)
    if not isinstance(r.get("standalone"), bool):
        raise ValueError("standalone not bool")
    if r.get("quality") not in (1, 2, 3, 4, 5):
        raise ValueError("quality out of range")
    if r.get("depth") not in (1, 2, 3):
        raise ValueError("depth out of range")
    if r.get("reason") not in REASONS:
        raise ValueError(f"unknown reason {r.get('reason')!r}")
    return {k: r[k] for k in ("standalone", "quality", "depth", "reason")}


def folder_sizes(rows):
    return Counter(r["capability_folder"] for r in rows)


def passes(row, label, sizes, small_folder_max=2_000):
    """Standalone, and quality >= 4 (>= 3 in folders with fewer than small_folder_max questions)."""
    if label is None or not label["standalone"]:
        return False
    floor = 3 if sizes[row["capability_folder"]] < small_folder_max else 4
    return label["quality"] >= floor


# --- per-document duplicate check ---

def dedup_batches(rows, labels, small_folder_max=2_000, max_batch=50):
    """Passing rows grouped by _source, in corpus order, split into <= max_batch.

    Returns [(batch_id, [row_index, ...])] for sources with at least two passing
    questions; a lone question has nothing to duplicate. batch_id is stable so a
    resumed run skips finished batches.
    """
    sizes = folder_sizes(rows)
    by_source = defaultdict(list)
    for i, (row, lab) in enumerate(zip(rows, labels)):
        if passes(row, lab, sizes, small_folder_max):
            by_source[row["_source"]].append(i)
    batches = []
    for source, idxs in by_source.items():
        if len(idxs) < 2:
            continue
        for start in range(0, len(idxs), max_batch):
            chunk = idxs[start:start + max_batch]
            if len(chunk) >= 2:
                batches.append((f"{source}#{start // max_batch}", chunk))
    return batches


def render_dedup_prompt(template, questions):
    lines = "\n".join(json.dumps({"id": i, "q": q}, ensure_ascii=False) for i, q in enumerate(questions))
    return template.replace("{{batch_jsonl}}", lines)


def parse_dedup(raw, n):
    """Raw model text -> list of n dup_of values (None or a lower batch-local id)."""
    data = _json_object(raw)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ValueError("missing results list")
    dup = {}
    for r in results:
        rid, d = r.get("id"), r.get("dup_of")
        if not isinstance(rid, int) or not 0 <= rid < n or rid in dup:
            raise ValueError(f"bad or duplicate id: {rid!r}")
        if d is not None and (not isinstance(d, int) or not 0 <= d < rid):
            raise ValueError(f"id {rid}: dup_of must point to an earlier id")
        dup[rid] = d
    if len(dup) != n:
        raise ValueError(f"expected {n} results, got {len(dup)}")
    return [dup[i] for i in range(n)]


# --- final selection ---

def select(rows, labels, duplicates=frozenset(), target=140_000, small_folder_max=2_000, template_cap=0.15):
    """rows[i] pairs with labels[i] (None = never labelled); duplicates is the set of
    row indices flagged as repeats. Returns (kept, stats).

    Rules, in order:
      1. drop unlabelled rows and duplicates
      2. keep standalone with quality >= 4; quality >= 3 in small folders
      3. rank by quality, then depth, descending
      4. within each question type, no opening template exceeds template_cap
         of that type's eligible rows
      5. small-folder rows are always kept; the rest fill up to target by rank
    """
    sizes = folder_sizes(rows)
    stats = Counter()
    eligible = []
    for i, (row, lab) in enumerate(zip(rows, labels)):
        if lab is None:
            stats["unlabelled"] += 1
        elif i in duplicates:
            stats["duplicate"] += 1
        elif not passes(row, lab, sizes, small_folder_max):
            stats["rejected"] += 1
        else:
            eligible.append((i, sizes[row["capability_folder"]] < small_folder_max))

    rank = lambda i: (-labels[i]["quality"], -labels[i]["depth"], i)
    eligible.sort(key=lambda e: rank(e[0]))

    type_total = Counter(rows[i]["_question_type"] for i, _ in eligible)
    type_template = Counter()
    capped = []
    for i, small in eligible:
        qt = rows[i]["_question_type"]
        key = (qt, template_key(rows[i]["question"]))
        if type_template[key] >= max(1, math.floor(template_cap * type_total[qt])):
            stats["template_capped"] += 1
            continue
        type_template[key] += 1
        capped.append((i, small))

    must = [i for i, small in capped if small]
    rest = [i for i, small in capped if not small]
    chosen = must + rest[:max(0, target - len(must))]
    stats["cut_by_target"] = len(capped) - len(chosen)

    kept = [rows[i] | {k: labels[i][k] for k in ("quality", "depth", "reason")}
            for i in sorted(chosen, key=rank)]
    stats["kept"] = len(kept)
    return kept, summarize(kept, stats)


def summarize(kept, stats):
    by = lambda field: dict(Counter(r[field] for r in kept).most_common())
    return {
        "funnel": dict(stats),
        "by_capability": by("capability"),
        "by_capability_folder": by("capability_folder"),
        "by_question_type": by("_question_type"),
        "by_quality": by("quality"),
        "by_depth": by("depth"),
    }
