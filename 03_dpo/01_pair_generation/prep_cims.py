"""CIM PDFs -> chunks.jsonl, one chunk per CIM section plus a few two-section chunks.

    python prep_cims.py --pdf-dir "../../content generation/CIMs" [--out work/corpus_cims/chunks.jsonl]

A CIM has six sections (Executive Summary, Investment Considerations, Business History and Description with an Industry and Market
Overview and one sub-section per segment, Financial Review, Management, Process Considerations). Each section becomes a chunk that opens
with a "[SECTION n: title]" label. Two-section chunks (Executive Summary + Financial Review, Investment Considerations + Financial
Review) carry the pairs that turn on one section agreeing with another (assign.py gives cross_section_conflict only to those).
Same record fields as the other chunk files, folder "cims"."""
import argparse
import json
import re
import sys
from pathlib import Path

import config

QUANTITY = re.compile(r"\d[\d,]*(?:\.\d+)?")
SECTION_START = re.compile(r"^Section (\d)(?:-([0A-D]))?\b")
FOOTER = re.compile(r"^Strictly Private and Confidential.*Page \d+\s*$", re.M)
MIN_CHARS = 300                           # a divider page alone is not a chunk
MAX_CHARS = 12000                         # a section longer than this is split at page boundaries (about 2,600 tokens a chunk)
PART_CHARS = 10000                        # what a two-section chunk takes from each section: its first pages, up to this many characters
TITLES = {"1": "Executive Summary", "2": "Investment Considerations", "3": "Business History and Description",
          "3-0": "Industry and Market Overview (with the key risks and mitigants)", "4": "Financial Review", "5": "Management",
          "6": "Process Considerations"}
SLUGS = {"1": "exec_summary", "2": "investment_considerations", "3": "business_history", "3-0": "industry_and_risks",
         "4": "financial_review", "5": "management", "6": "process"}
COMBINED = [("1", "4", "exec_plus_financials"), ("2", "4", "considerations_plus_financials")]


def clean(page: str) -> str:
    """Page text without the repeating footer; the PDF's private-use bullet glyph becomes a hyphen."""
    page = re.sub(r"[-]", "-", page)
    return FOOTER.sub("", page).strip()


def sections(pages: list[str]) -> dict[str, dict]:
    """{key: {"title", "pages"}} in document order, "pages" being the section's cleaned page texts. A page that starts "Section N" or "Section N-X" opens a section; later pages follow it."""
    out: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    key = None
    for raw in pages:
        text = clean(raw)
        m = SECTION_START.match(text)
        if m:
            key = f"{m.group(1)}-{m.group(2)}" if m.group(2) else m.group(1)
            first = text.split("\n")[0]
            titles[key] = first.split(":", 1)[1].strip() if ":" in first else TITLES.get(key, "")
            if m.group(2) is None and len(text.split("\n")) <= 2:     # a bare divider page: nothing else on it
                out.setdefault(key, [])
                continue
        if key is not None:
            out.setdefault(key, []).append(text)
    return {k: {"title": titles.get(k) or TITLES.get(k, k), "pages": [t for t in v if t]} for k, v in out.items()}


def pack(pages: list[str], limit: int) -> list[str]:
    """Consecutive pages joined into parts of at most `limit` characters (a single longer page is its own part)."""
    parts, cur = [], ""
    for pg in pages:
        if cur and len(cur) + len(pg) + 1 > limit:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n{pg}" if cur else pg
    return parts + ([cur] if cur else [])


def label(key: str, sec: dict) -> str:
    title = TITLES.get(key, sec["title"]) if key in TITLES else f"Business segment {key.split('-')[1]}: {sec['title']}"
    return f"[SECTION {key}: {title}]"


def slug(key: str) -> str:
    return SLUGS.get(key) or "segment_" + key.split("-")[1].lower()


def company_name(pages: list[str], stem: str) -> str:
    m = re.search(r'on behalf of (.+?) \(the "Company"\)', " ".join(" ".join(pages[:3]).split()))
    return m.group(1) if m else stem.replace("_CIM", "")


NL = chr(10)
PAGE_BREAK = NL + NL + "<!-- page -->" + NL + NL


def md_of(pages: list[str]) -> str:
    """The text of one CIM as a markdown file: its cleaned pages, a page-break comment between them (pages_of_md reverses it)."""
    return PAGE_BREAK.join(clean(p) for p in pages if clean(p)) + NL


def pages_of_md(md: str) -> list[str]:
    return [p.strip() for p in md.split(PAGE_BREAK.strip()) if p.strip()]


def facts_of(data: dict) -> list[float]:
    """Every number of a CIM's structured JSON that the sheets and answers can quote: the yearly series (revenue, EBITDA, net income, each
    segment), net debt, leverage, and the two CAGRs also as percentages. Sorted, no duplicates."""
    out: set[float] = set()

    def walk(o, key=""):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            out.add(float(o))
            if key.startswith("cagr") or key.startswith("leverage"):
                out.add(float(o) * 100 if key.startswith("cagr") else float(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)

    for k in ("revenue", "ebitda", "net_income", "segments", "net_debt_fy25", "leverage_fy25", "cagr_rev_hist", "cagr_rev_proj"):
        walk(data.get(k), k)
    return sorted(out)


def cim_chunks(pages: list[str], stem: str, facts: list[float] | None = None) -> list[dict]:
    """The chunks of one CIM given its page texts: each section (split into parts if long), then the two-section chunks."""
    secs, company = sections(pages), company_name(pages, stem)
    base = re.sub(r"[^A-Za-z0-9]+", "_", stem.replace("_CIM", "")).strip("_")
    secs = {k: s for k, s in secs.items() if len("".join(s["pages"])) >= MIN_CHARS}
    items = []                                     # (chunk name, [labelled section texts])
    for k, sec in secs.items():
        parts = pack(sec["pages"], MAX_CHARS)
        for i, part in enumerate(parts):
            tag = f" (part {i + 1} of {len(parts)})" if len(parts) > 1 else ""
            items.append((slug(k) + (f"_p{i + 1}" if len(parts) > 1 else ""), [f"{label(k, sec)}{tag}\n{part}"]))
    for a, b, name in COMBINED:
        if a in secs and b in secs:
            items.append((name, [f"{label(k, secs[k])} (first pages)\n{pack(secs[k]['pages'], PART_CHARS)[0]}" for k in (a, b)]))
    out = []
    for i, (name, texts) in enumerate(items):
        text = f"CIM of {company}\n\n" + "\n\n".join(texts)
        out.append({"chunk_id": f"{base}_{name}", "source_key": stem, "folder": config.CIM_FOLDER, "title": f"{company}: {name.replace('_', ' ')}",
                    "url": "", "chunk_index": i, "n_chunks": len(items), "n_quantities": len(QUANTITY.findall(text)), "facts": facts or [], "text": text})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", default=str(config.HERE / "work" / "corpus_cims" / "chunks.jsonl"))
    ap.add_argument("--md-dir", help="also write one <name>.md per CIM here (what goes to documents/cims/)")
    a = ap.parse_args()
    import pypdf
    chunks = []
    for pdf in sorted(Path(a.pdf_dir).glob("*_CIM.pdf")):
        pages = [p.extract_text() or "" for p in pypdf.PdfReader(pdf).pages]
        js = pdf.with_name(pdf.stem + "_structured.json")
        facts = facts_of(json.loads(js.read_text(encoding="utf-8"))) if js.exists() else []
        chunks += cim_chunks(pages, pdf.stem, facts)
        if a.md_dir:
            Path(a.md_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.md_dir) / (pdf.stem + ".md")).write_text(md_of(pages), encoding="utf-8")
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("two chunks share a chunk_id")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    print(f"[cims] {len(chunks)} chunks -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
