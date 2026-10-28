"""Tear sheet PDFs -> one markdown file each, with LlamaParse (v2 Parse API), and a check that no figure changed on the way.

    python convert_tearsheets.py [--limit N] [--tier cost_effective] [--first-key 31] [--out work/tearsheets_md]

Reads the PDFs under `content generation/tear sheets/` (the 30 numbered startup sheets and `sectors/<Sector>/*.pdf`; the file
`Tear sheet example 1.pdf` is a template and is skipped). Writes a FLAT folder of .md files named like the PDFs (`&` -> `and`), plus
`conversion_report.json`. Nothing is uploaded here.

Keys: the environment variables llamaparse1 .. llamaparse33 (qa_pairs_prep/.env). A key that answers 401/402/403 (no credit, bad key) is
dropped and the next one is used; keys are never printed. Every raw reply is cached in `<out>/../tearsheets_raw/`, so a re-run costs no credits.

Check (the numbers of the PDF's own text layer, read with PyMuPDF, against the converted markdown): every figure in the PDF must be in the
markdown and every figure in the markdown must be in the PDF. Digits only are compared, so "$21,984,590" and "21984590" match; a figure the
conversion dropped, split or invented is reported, and the sheet is marked not clean.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

import config

BASE = "https://api.cloud.llamaindex.ai"
SHEETS_ROOT = config.HERE.parent.parent / "content generation" / "tear sheets"
SKIP_NAMES = {"Tear sheet example 1.pdf"}
DEAD = (401, 402, 403)
NL = chr(10)
GENERIC = {"region", "category", "country", "count", "percentage", "share", "metric", "label", "value", "headcount", "total headcount", "%"}
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class NoKeyLeft(RuntimeError):
    pass


def collect_pdfs(root: Path = SHEETS_ROOT) -> list[Path]:
    """The startup sheets in the folder, then the sector sheets, in name order; the example template is left out."""
    return sorted((p for p in root.rglob("*.pdf") if p.name not in SKIP_NAMES), key=lambda p: (p.parent != root, p.parent.name, p.name))


def flat_name(pdf: Path) -> str:
    return pdf.stem.replace("&", "and") + ".md"


def numbers(text: str) -> list[str]:
    """Every figure of a text as digits and dots only ("$21,984,590" -> "21984590", "6.65%" -> "6.65")."""
    return [m.replace(",", "") for m in NUMBER.findall(text)]


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def strip_unverified_tables(md: str, pdf_text_: str) -> tuple[str, list[dict]]:
    """Replace every markdown table holding a figure that the PDF's text layer does not hold.

    Such a table is something the converter read from a picture (a chart whose slices carry no numbers in the sheet): its values are
    estimates, not the sheet's. The table becomes a one-line note that names the chart and its labels and says no values are shown.
    Returns the new markdown and, per removed table, its header and labels (for the report)."""
    known, removed, out, block = set(numbers(pdf_text_)), [], [], []

    def flush():
        if not block:
            return
        rows = [_cells(l) for l in block if not re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", l)]
        extra = sorted(set(numbers(NL.join(block))) - known)
        if extra and rows:
            # Labels = the first-column words of every row (the converter sometimes takes a data row for the header and repeats the header row);
            # a cell with a digit or a generic column word is not a label, so no figure can leak into the note.
            labels = list(dict.fromkeys(c[0] for c in rows if c and c[0] and not re.search(r"\d", c[0]) and c[0].lower() not in GENERIC))
            removed.append({"header": rows[0], "labels": labels, "figures_removed": extra})
            out.append(f"> Chart in the source, with no values shown for its slices: {', '.join(labels)}.")
        else:
            out.extend(block)
        block.clear()

    for line in md.split(NL):
        if re.match(r"^\s*\|.*\|\s*$", line):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return NL.join(out), removed


def drop_trend_column(md: str) -> str:
    """Remove the whole 'Trend' column from every markdown table that has one: its cells are pictures of tiny charts (icons such as
    an up or down arrow), which the converter can only guess at."""
    out, block = [], []

    def flush():
        if not block:
            return
        rows = [_cells(l) for l in block]
        col = next((i for i, c in enumerate(rows[0]) if c.strip("* ").lower() == "trend"), None)
        if col is None:
            out.extend(block)
        else:
            for cells in rows:
                kept = [c for i, c in enumerate(cells) if i != col]
                out.append("| " + " | ".join(kept) + " |")
        block.clear()

    for line in md.split(NL):
        if re.match(r"^\s*\|.*\|\s*$", line):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return NL.join(out)


def clean_markdown(md: str) -> str:
    """Remove conversion leftovers without touching a figure: the Trend column, stray '[signature]' lines, doubled bold markers
    ('****x****'), and the page-header banner ('# SYNTHETIC ...') repeated on every page (the first stays)."""
    md = drop_trend_column(md)
    md = re.sub(r"\*{4}([^*\n]+)\*{4}", r"**\1**", md)
    lines, seen_banner = [], False
    for line in md.split(NL):
        if line.strip() == "[signature]":
            continue
        if line.startswith("# SYNTHETIC"):
            if seen_banner:
                continue
            seen_banner = True
        lines.append(line)
    return re.sub(r"\n{4,}", NL * 3, NL.join(lines)).strip() + NL


def compare_figures(pdf_text: str, md_text: str) -> dict:
    """Figures only in the PDF (dropped or split by the conversion) and only in the markdown (invented or merged)."""
    a, b = set(numbers(pdf_text)), set(numbers(md_text))
    return {"missing_from_markdown": sorted(a - b), "not_in_pdf": sorted(b - a), "figures_in_pdf": len(a), "clean": a == b}


class Parser:
    def __init__(self, keys: list[str], tier: str, markdown_tables: bool = True, session=requests):
        self.keys, self.tier, self.http, self.i = keys, tier, session, 0
        self.config = {"tier": tier, "version": "latest"}
        if markdown_tables:
            self.config["output_options"] = {"markdown": {"tables": {"output_tables_as_markdown": True}}}

    def _headers(self) -> dict:
        if self.i >= len(self.keys):
            raise NoKeyLeft("every LlamaParse key is out of credit or rejected")
        return {"Authorization": "Bearer " + self.keys[self.i]}

    def parse(self, pdf: Path) -> dict:
        """The raw v2 reply for one PDF (markdown_full + per-page markdown); rotates keys on 401/402/403."""
        while True:
            with open(pdf, "rb") as f:
                r = self.http.post(f"{BASE}/api/v2/parse/upload", headers=self._headers(), files={"file": f},
                                   data={"configuration": json.dumps(self.config)}, timeout=120)
            if r.status_code in DEAD:
                self.i += 1
                continue
            if r.status_code == 422 and "output_options" in self.config:          # an option this account's API does not know
                self.config.pop("output_options")
                continue
            r.raise_for_status()
            job = r.json()["id"]
            break
        for _ in range(150):
            g = self.http.get(f"{BASE}/api/v2/parse/{job}", headers=self._headers(), timeout=60)
            g.raise_for_status()
            status = g.json()["job"]["status"]
            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(2)
        if status != "COMPLETED":
            raise RuntimeError(f"{pdf.name}: parse job ended {status}: {g.json()['job'].get('error_message')}")
        d = self.http.get(f"{BASE}/api/v2/parse/{job}", headers=self._headers(), params={"expand": "markdown,markdown_full"}, timeout=60)
        d.raise_for_status()
        out = d.json()
        out.pop("debug", None)
        out.pop("raw_parameters", None)
        return out


def markdown_of(reply: dict) -> str:
    full = reply.get("markdown_full")
    if isinstance(full, str) and full.strip():
        return full.strip() + "\n"
    pages = (reply.get("markdown") or {}).get("pages") or []
    text = "\n\n".join(p.get("markdown", "") if isinstance(p, dict) else str(p) for p in pages).strip()
    if not text:
        raise ValueError("the reply holds no markdown")
    return text + "\n"


def pdf_text(pdf: Path) -> str:
    import fitz
    with fitz.open(pdf) as d:
        return "\n".join(page.get_text() for page in d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.HERE / "work" / "tearsheets_md"))
    ap.add_argument("--limit", type=int, help="only the first N sheets (a trial)")
    ap.add_argument("--tier", default="cost_effective", choices=["fast", "cost_effective", "agentic", "agentic_plus"])
    ap.add_argument("--first-key", type=int, default=1, help="start with llamaparse<N> (earlier keys are known to be out of credit)")
    ap.add_argument("--no-markdown-tables", action="store_true", help="do not ask for pipe tables (the API then returns HTML tables)")
    a = ap.parse_args()
    out = Path(a.out)
    raw_dir = out.parent / "tearsheets_raw"
    out.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    keys = [os.environ[f"llamaparse{i}"].strip() for i in range(a.first_key, 34) if os.environ.get(f"llamaparse{i}", "").strip()]
    parser = Parser(keys, a.tier, not a.no_markdown_tables)
    pdfs = collect_pdfs()[:a.limit]
    print(f"[tearsheets] {len(pdfs)} PDFs, tier {a.tier}, {len(keys)} keys available from llamaparse{a.first_key}", flush=True)
    report, started = [], time.time()
    for n, pdf in enumerate(pdfs, 1):
        cache = raw_dir / (pdf.stem.replace("&", "and") + ".json")
        try:
            if cache.exists():
                reply = json.loads(cache.read_text(encoding="utf-8"))
            else:
                reply = parser.parse(pdf)
                cache.write_text(json.dumps(reply, ensure_ascii=False), encoding="utf-8")
            md = markdown_of(reply)
        except NoKeyLeft as e:
            print(f"[tearsheets] stopped at {n - 1} of {len(pdfs)}: {e}", flush=True)
            break
        text = pdf_text(pdf)
        md, removed = strip_unverified_tables(md, text)
        md = clean_markdown(md)
        (out / flat_name(pdf)).write_text(md, encoding="utf-8")
        cmp = compare_figures(text, md)
        report.append({"pdf": str(pdf.relative_to(SHEETS_ROOT)), "md": flat_name(pdf), "words": len(md.split()), "tables_as_html": "<table" in md,
                       "chart_tables_replaced_by_a_note": removed, **cmp})
        flag = "ok" if cmp["clean"] else f"MISMATCH missing={cmp['missing_from_markdown'][:6]} extra={cmp['not_in_pdf'][:6]}"
        print(f"[tearsheets] {n}/{len(pdfs)} {pdf.name}: {len(md.split())} words, key llamaparse{a.first_key + parser.i}, {flag}" + (f" | chart table replaced by a note: {removed[0]['labels']}" if removed else ""), flush=True)
    (out.parent / "conversion_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    clean = sum(r["clean"] for r in report)
    print(f"[tearsheets] done: {len(report)} converted, {clean} with every figure matching, {len(report) - clean} to look at; "
          f"{time.time() - started:.0f}s; output {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
