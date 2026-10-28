"""Dataset curation: remove the "according to the document" habit, cap templated openers, build DPO pairs, measure leaks.

Why: 36% of the pool's reasoning (and 10% of its final answers) narrate a document ("The document states that ...")
although the question carries no document, so the model learns to invent one. Fix, per row:
  * rewrite the narration away by rule (rule_clean);
  * a row whose content still depends on a source (source_bound) is trained WITH its document (grounded), when it can
    be matched back to the scored corpus, and dropped otherwise; a self-contained row is trained without a document.
Everything here is pure text logic (no S3, no model), so it is unit-tested.
"""
import random
import re
from collections import Counter, defaultdict

_I = re.I
SOURCE = r"(?:(?:the|this|these|that|such|said|provided|given|above|attached)\s+)*(?:document|passage|excerpt|paper|study|text|article|filing|report|source|material|context)s?"
VERBS = (r"states?|says?|notes?|mentions?|indicates?|explains?|shows?|reports?|highlights?|specif(?:y|ies)|describes?|clarif(?:y|ies)|"
         r"observes?|reveals?|emphasi[sz]es?|asserts?|confirms?|suggests?|argues?|finds?|points out|stipulates?|requires?")
ADV = r"(?:(?:also|further|explicitly|clearly|then|again|expressly|specifically)\s+)?"

LEAD_IN = re.compile(rf"^(\W*)(?:according to|based on|per|as (?:stated|mentioned|noted|described|shown|indicated|outlined|discussed|reported|specified) (?:in|by)|"
                     rf"from|in|within)\s+{SOURCE}\s*,\s*", _I)
TRAIL = re.compile(rf",?\s+(?:according to|as (?:stated|mentioned|noted|described|shown|indicated) (?:in|by)|as per|per)\s+{SOURCE}(?=\s*[.;,:)]|$)", _I)
SUBJECT = re.compile(rf"^(\W*){SOURCE}\s+{ADV}(?:{VERBS})\s+that\s+(.+)$", _I | re.S)
SUBJECT_NO_THAT = re.compile(rf"^(\W*){SOURCE}\s+{ADV}(?:{VERBS})\s*[:,]\s*(.+)$", _I | re.S)

# What is left after the rules and still ties a row to a source we cannot show: not usable without its document.
RESIDUAL = re.compile(r"\b(?:documents?|passages?|excerpts?|the paper|this paper|the study|this study|the article|the text|the authors?|"
                      r"the table|the figure|the appendix|section\s+\d|paragraphs?|chapter\s+\d|the filing|the report)\b", _I)
# A question that points at something we do not give: "the study", "the description of ...", "in the passage".
QUESTION_DEICTIC = re.compile(r"\b(?:the|this|these|given|provided|above|following|described|mentioned|stated)\s+"
                              r"(?:documents?|passages?|excerpts?|texts?|papers?|stud(?:y|ies)|articles?|tables?|figures?|filings?|reports?|"
                              r"disclosures?|clauses?|sections?|authors?|samples?|regressions?|descriptions?|examples?|scenarios?)\b"
                              r"|\b(?:in|from|per|according to|based on)\s+the\s+(?:study|paper|text|passage|document)\b", _I)


# ---- reasoning / answer text -----------------------------------------------------------------

_THINK = re.compile(r"\s*<think>(.*?)</think>(.*)", re.S)


def split_think(content: str) -> tuple[str, str]:
    """Pool assistant content "<think>reasoning</think>answer" -> (reasoning, answer). No block: ("", content)."""
    m = _THINK.match(content or "")
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", (content or "").strip())


def join_think(reasoning: str, answer: str) -> str:
    return f"<think>{reasoning.strip()}</think>{answer.strip()}"


def mentions_source(text: str) -> bool:
    """True if the text still narrates or cites a source document (rules above, before cleaning)."""
    return bool(LEAD_IN.search(text) or TRAIL.search(text) or SUBJECT.search(text) or SUBJECT_NO_THAT.search(text)
                or re.search(rf"\b(?:{SOURCE})\s+{ADV}(?:{VERBS})\b", text, _I) or RESIDUAL.search(text))


def _cap(s: str) -> str:
    """Capitalise the first letter (leading quotes or brackets are skipped)."""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def _clean_sentence(s: str) -> str:
    original = s
    for _ in range(2):                                            # a lead-in can sit in front of a subject form
        s = LEAD_IN.sub(lambda m: m.group(1), s, count=1)
        m = SUBJECT.match(s) or SUBJECT_NO_THAT.match(s)
        if m:
            s = m.group(1) + m.group(2)
    s = TRAIL.sub("", s)
    return _cap(s) if s != original else s


def rule_clean(text: str) -> str:
    """Rewrite document narration out of reasoning or an answer, keeping every other word.

    "The document states that X."  ->  "X."     "According to the passage, X"  ->  "X"     "X, as noted in the text."  ->  "X."
    Sentences it cannot rewrite safely ("The document lists A, B and C.") are left as they are; `source_bound` finds them.
    """
    out = []
    for line in text.split("\n"):
        out.append(" ".join(_clean_sentence(s) for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", line)) if line.strip() else line)
    return "\n".join(out)


def source_bound(question: str, reasoning: str, answer: str) -> bool:
    """True if the row cannot stand without its document, even after rule_clean."""
    return bool(QUESTION_DEICTIC.search(question) or mentions_source(rule_clean(reasoning)) or mentions_source(rule_clean(answer)))


# ---- balance --------------------------------------------------------------------------------

def opener(text: str, n: int = 3) -> str:
    """The first n words of an answer: the template fingerprint."""
    return " ".join(re.findall(r"[a-z0-9']+", text.lower())[:n])


def cap_by_key(items: list, key, max_share: float, seed: int = 0, rounds: int = 5) -> list:
    """Keep at most max_share of the (remaining) items per key value, dropping random extras; order is preserved."""
    rng, kept = random.Random(seed), list(range(len(items)))
    for _ in range(rounds):
        limit = max(1, int(max_share * len(kept)))
        groups = defaultdict(list)
        for i in kept:
            groups[key(items[i])].append(i)
        drop = {i for g in groups.values() if len(g) > limit for i in rng.sample(g, len(g) - limit)}
        if not drop:
            break
        kept = [i for i in kept if i not in drop]
    return [items[i] for i in kept]


def dedupe(items: list, key, per_key: int = 1) -> list:
    """Keep the first `per_key` items for each key value (e.g. per source chunk)."""
    seen, out = Counter(), []
    for it in items:
        k = key(it)
        if seen[k] < per_key:
            seen[k] += 1
            out.append(it)
    return out


# ---- DPO pairs from the fix ----------------------------------------------------------------

def dpo_pair(question: str, original: str, cleaned: str, system: str | None = None, source: str = "doc_fix") -> dict | None:
    """chosen = the cleaned answer (reasoning + answer), rejected = the original; None if cleaning changed nothing."""
    if original.strip() == cleaned.strip():
        return None
    prompt = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return {"prompt": prompt, "chosen": cleaned, "rejected": original, "source": source}


# ---- guard metrics -------------------------------------------------------------------------

def doc_mention_rate(texts: list[str]) -> float:
    """Share of texts that narrate or cite a source document (use on answers to questions that carry no document)."""
    return sum(mentions_source(t) for t in texts) / len(texts) if texts else 0.0
