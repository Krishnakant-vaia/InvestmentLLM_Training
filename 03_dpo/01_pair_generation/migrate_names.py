"""Rename chosen/rejected/accepted to preferred/not_preferred in every pair file on S3, in place, with a backup and a proof.

    python migrate_names.py backup              # download every affected file to --backup-dir (S3 is only read)
    python migrate_names.py stage               # rename into --stage-dir and prove each file lost nothing (S3 is only read)
    python migrate_names.py apply               # overwrite the S3 files from --stage-dir (refuses any file changed since the backup)

Affected files (under <OUT_ROOT>): training/dpo_train.jsonl; full-1/generation/pairs_raw.jsonl and full-1/pairs_raw.jsonl;
full-1/final/scored_pairs.jsonl; every numerical-*/**/final/judged_pairs.jsonl; every numerical-input/**/*.jsonl.
Not touched: rewrite results (their `accepted` is a yes/no), judge checkpoints, logs, reports, the Excel.

"stage" proves each file: the renamed rows, renamed back, equal the original rows exactly (same count, same order, same values).
The bucket is versioned, so the overwritten versions also stay on S3 as earlier versions.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import config
import s3util
from schema import normalize_row, to_old_names

ROOT_DIR = config.HERE.parent.parent                                   # DPO_Training/
KINDS = [(re.compile(r"^training/dpo_train\.jsonl$"), "training"),
         (re.compile(r"^full-1/(generation/)?pairs_raw\.jsonl$"), "raw"),
         (re.compile(r"^full-1/final/scored_pairs\.jsonl$"), "scored"),
         (re.compile(r"^numerical-[^/]+/.*/final/judged_pairs\.jsonl$"), "numerical"),
         (re.compile(r"^numerical-input/.*\.jsonl$"), "numerical")]


def kind_of(rel: str) -> str | None:
    return next((k for rx, k in KINDS if rx.match(rel)), None)


def affected(s3) -> list[tuple[str, str, int, str]]:
    """(relative key, kind, size, etag) of every file to migrate."""
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=config.BUCKET, Prefix=config.OUT_ROOT):
        for o in page.get("Contents", []):
            rel = o["Key"][len(config.OUT_ROOT):]
            if kind_of(rel):
                out.append((rel, kind_of(rel), o["Size"], o["ETag"]))
    return sorted(out)


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def migrate_file(src: Path, dst: Path, kind: str) -> dict:
    """Write the renamed copy of `src` to `dst` and prove it is lossless. Returns counts."""
    old = read_rows(src)
    new = [normalize_row(r, accepted_is_answer=(kind == "numerical")) for r in old]
    back = [to_old_names(r, kind) for r in new]
    if back != old:
        bad = next(i for i, (a, b) in enumerate(zip(back, old)) if a != b)
        raise SystemExit(f"[migrate] {src}: renamed rows do not reverse to the originals (first difference at row {bad})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    renamed = sum(1 for a, b in zip(old, new) if list(a) != list(b))
    flags = sum(1 for a, b in zip(old, new) if a.get("flags") != b.get("flags"))
    return {"rows": len(old), "rows_renamed": renamed, "rows_with_flag_text_renamed": flags}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["backup", "stage", "apply"])
    ap.add_argument("--backup-dir", default=str(ROOT_DIR / "_backup_before_rename"))
    ap.add_argument("--stage-dir", default=str(ROOT_DIR / "_staged_renamed"))
    a = ap.parse_args()
    backup, stage, s3 = Path(a.backup_dir), Path(a.stage_dir), s3util.client()
    files = affected(s3)
    if a.step == "backup":
        total = 0
        for rel, _, size, etag in files:
            s3util.download(s3, config.OUT_ROOT + rel, backup / rel)
            total += size
        (backup / "_manifest.json").write_text(json.dumps([{"key": r, "kind": k, "size": s, "etag": e} for r, k, s, e in files], indent=1), encoding="utf-8")
        print(f"[migrate] backed up {len(files)} files ({total / 1e6:.0f} MB) to {backup}")
        return 0
    manifest = {m["key"]: m for m in json.loads((backup / "_manifest.json").read_text(encoding="utf-8"))}
    if a.step == "stage":
        summary = {}
        for rel, kind, _, _ in files:
            if rel not in manifest:
                raise SystemExit(f"[migrate] {rel} is on S3 but not in the backup: run `backup` again")
            r = migrate_file(backup / rel, stage / rel, kind)
            for k, v in r.items():
                summary[(kind, k)] = summary.get((kind, k), 0) + v
            summary[(kind, "files")] = summary.get((kind, "files"), 0) + 1
        for kind in ("training", "raw", "scored", "numerical"):
            print(f"[migrate] {kind:10} files={summary.get((kind, 'files'), 0):3} rows={summary.get((kind, 'rows'), 0):6} "
                  f"rows renamed={summary.get((kind, 'rows_renamed'), 0):6} flag texts renamed in {summary.get((kind, 'rows_with_flag_text_renamed'), 0)} rows")
        print(f"[migrate] every file reverses to its original rows exactly. Staged in {stage}")
        return 0
    changed = [rel for rel, _, _, etag in files if manifest.get(rel, {}).get("etag") != etag]
    if changed or len(files) != len(manifest):
        raise SystemExit(f"[migrate] S3 changed since the backup ({len(changed)} files differ, {len(files)} vs {len(manifest)} files): nothing was written")
    for rel, kind, _, _ in files:
        body = (stage / rel).read_bytes()
        s3.put_object(Bucket=config.BUCKET, Key=config.OUT_ROOT + rel, Body=body)
    print(f"[migrate] overwrote {len(files)} files on S3 with the new names (earlier versions stay on S3: the bucket is versioned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
