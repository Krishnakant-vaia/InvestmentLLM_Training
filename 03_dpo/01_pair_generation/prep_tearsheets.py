"""Tear-sheet markdown -> chunks.jsonl, one chunk per sheet (a sheet is ~1,000-2,000 words, so it is never split).

    python prep_tearsheets.py [--out work/corpus_tearsheets/chunks.jsonl] [--local-dir work/tearsheets_md]

Reads `documents/tear sheets/*.md` from S3 (or a local folder). Same record fields as prep_corpus chunks, folder "tear sheets"."""
import argparse
import json
import re
import sys
from pathlib import Path

import config

QUANTITY = re.compile(r"\d[\d,]*(?:\.\d+)?")


def sheet_chunks(texts: dict[str, str]) -> list[dict]:
    """{file name or key: markdown} -> one chunk per non-empty sheet, in name order."""
    out = []
    for i, (name, text) in enumerate(sorted(texts.items())):
        stem = Path(name).stem
        if not text.strip():
            continue
        out.append({"chunk_id": re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_"), "source_key": name, "folder": config.TEARSHEET_FOLDER,
                    "title": stem.replace("_", " "), "url": "", "chunk_index": 0, "n_chunks": 1,
                    "n_quantities": len(QUANTITY.findall(text)), "text": text.strip()})
    ids = [c["chunk_id"] for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError("two sheets map to the same chunk_id")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.HERE / "work" / "corpus_tearsheets" / "chunks.jsonl"))
    ap.add_argument("--local-dir", help="read the .md files from this folder instead of S3")
    a = ap.parse_args()
    if a.local_dir:
        texts = {p.name: p.read_text(encoding="utf-8") for p in Path(a.local_dir).glob("*.md")}
    else:
        import s3util
        s3 = s3util.client()
        prefix = config.DOCS_PREFIX + config.TEARSHEET_FOLDER + "/"
        keys = [k for k in s3util.list_keys(s3, prefix) if k.endswith(".md")]
        texts = {k: s3.get_object(Bucket=config.BUCKET, Key=k)["Body"].read().decode("utf-8") for k in keys}
    chunks = sheet_chunks(texts)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    print(f"[tearsheets] {len(chunks)} sheets -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
