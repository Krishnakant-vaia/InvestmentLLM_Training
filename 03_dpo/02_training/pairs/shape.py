"""Shape the pairs for training: evaluation groups, leak-free split groups, the <think> format, and the length-bias guard.

    shape(rows, policy) -> (rows, report)

`policy` is the `dpo.data.shape` block of config.yaml. Every row comes back with
    source  the EVALUATION group (a domain, not one of ~60 capabilities): the trainer evaluates per `source`;
    group   what the train/val split keeps together: a source document (document pairs) or a question template (numerical pairs).
Nothing is dropped unless `max_preferred_longer_share` asks for it, and then every dropped pair is counted in the report.

Why the groups (issues found in the notebook review):
  * the numerical questions are templated ("LCR disclosure for a bank: HQLA X, outflows Y ..."): a split by exact prompt puts near
    copies on both sides and inflates validation accuracy. The `group` is the question with numbers and capitalised names masked;
  * document pairs share one document across many questions: the group is the chunk the pairs were written from.
"""
import hashlib
import re

from pairs.clean import clean_answer

SHEET_OF_PREFIX = {"DK": "domain_knowledge", "EK": "enterprise_knowledge", "FR": "financial_reasoning", "WU": "workflow_understanding",
                   "CG": "content_generation", "AR": "agentic_reasoning", "CL": "continuous_learning", "EA": "enterprise_ai_applications"}
_PREFIX = re.compile(r"^([A-Z]{2})-\d")
_ANSWER_LINE = re.compile(r"^Answer:", re.M)
_NUMBER_OR_NAME = re.compile(r"\d[\d,.]*|\b[A-Z][a-zA-Z&'-]*")        # digits, and capitalised words (companies, tools, months)
_BRACKET_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")                  # "[Deccan Dairy, file ref 438]" in front of numerical questions


def domain_of(row: dict) -> str:
    """Document pairs keep their source label; numerical pairs get their sheet (the pair id starts with DK-, WU-, ...)."""
    m = _PREFIX.match(str(row.get("chunk_id") or row.get("call_id") or ""))
    return SHEET_OF_PREFIX.get(m.group(1), row["source"]) if m else row["source"]


def user_text(row: dict) -> str:
    msgs = row["prompt"]
    return msgs if isinstance(msgs, str) else next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")


def skeleton(text: str) -> str:
    """A question with its numbers and names masked, so questions made from one template share a key."""
    text = _BRACKET_PREFIX.sub("", text)
    return " ".join(_NUMBER_OR_NAME.sub("#", text).lower().split())


def group_of(row: dict, by: str) -> str:
    if by == "prompt":
        return hashlib.sha1(user_text(row).encode()).hexdigest()[:12]
    if row.get("origin", "").startswith("full-"):                   # document pairs: the chunk, or one document leaks across splits
        return f"doc:{row.get('chunk_id') or hashlib.sha1(user_text(row)[:2000].encode()).hexdigest()[:12]}"
    return "tpl:" + hashlib.sha1(skeleton(user_text(row)).encode()).hexdigest()[:12]


# ---- the <think> format ------------------------------------------------------------------------------------

def workings_in_think(chosen: str, rejected: str) -> tuple[str, str]:
    """Move the working of an answer into a <think> block and keep only the final "Answer:" line outside, for BOTH sides.

    Only when neither side has a <think> block already and both have an "Answer:" line with working before it: otherwise the pair is
    returned unchanged (never one side transformed, which would teach "no think block = wrong")."""
    def split(t: str):
        ms = list(_ANSWER_LINE.finditer(t))
        if not ms or "<think>" in t:
            return None
        pre, ans = t[:ms[-1].start()].strip(), t[ms[-1].start():].strip()
        return (pre, ans) if pre else None
    c, r = split(chosen), split(rejected)
    if c is None or r is None:
        return chosen, rejected
    return f"<think>\n{c[0]}\n</think>\n\n{c[1]}", f"<think>\n{r[0]}\n</think>\n\n{r[1]}"


# ---- the length-bias guard ---------------------------------------------------------------------------------

def cap_preferred_longer(rows: list[dict], max_share: float) -> tuple[list[dict], int]:
    """Drop the pairs where the preferred answer is longest by the widest margin until its share of all pairs is <= max_share."""
    words = lambda t: len(t.split())  # noqa: E731
    gap = [(words(r["preferred"]) - words(r["not_preferred"]), i) for i, r in enumerate(rows)]
    longer = sorted((g, i) for g, i in gap if g > 0)
    n_drop = 0
    total, n_longer = len(rows), len(longer)
    while n_drop < len(longer) and (n_longer - n_drop) / (total - n_drop) > max_share:
        n_drop += 1
    drop = {i for _, i in longer[len(longer) - n_drop:]}            # widest gaps go first
    return [r for i, r in enumerate(rows) if i not in drop], n_drop


# ---- all of it ---------------------------------------------------------------------------------------------

def shape(rows: list[dict], policy: dict) -> tuple[list[dict], list[str]]:
    out, changed = [], 0
    for r in rows:
        r = dict(r)
        r["domain"] = domain_of(r)
        r["source"] = r["domain"] if policy.get("eval_group", "domain") == "domain" else r["source"]
        r["group"] = group_of(r, policy.get("split_by", "template"))
        if policy.get("clean_remarks", True) and not r.get("origin", "").startswith("full-"):     # document pairs were never Excel pairs
            r["preferred"], r["not_preferred"] = clean_answer(r["preferred"]), clean_answer(r["not_preferred"])
        if policy.get("think_format") == "workings_in_think":
            c, j = workings_in_think(r["preferred"], r["not_preferred"])
            changed += (c, j) != (r["preferred"], r["not_preferred"])
            r["preferred"], r["not_preferred"] = c, j
        out.append(r)
    report = [f"[pairs shape] {len(out)} pairs in {len({r['source'] for r in out})} evaluation groups, {len({r['group'] for r in out})} split groups "
              f"(split_by={policy.get('split_by', 'template')})"]
    if policy.get("think_format") == "workings_in_think":
        report.append(f"    <think> format: working moved into <think> for {changed} pairs (both sides); the rest are unchanged")
    cap = policy.get("max_preferred_longer_share")
    if cap is not None:
        out, n = cap_preferred_longer(out, cap)
        report.append(f"    length guard: {n} pairs dropped (widest preferred-longer gaps) to keep the preferred-longer share at or below {cap:.0%}")
    return out, report
