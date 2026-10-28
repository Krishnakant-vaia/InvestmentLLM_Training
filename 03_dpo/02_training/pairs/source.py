"""Where the preference pairs come from: the training set that QA_pairs_prep builds on S3.

    load_pairs(data_cfg) -> (rows, report)

The file is `<OUT_ROOT>training/dpo_train.jsonl` with `training/manifest.json` beside it (QA_pairs_prep/make_training_set.py appends
one batch per source run to both). This module is the only place that knows that layout. It downloads the file, and refuses to go on
unless it is the file the manifest describes: the same number of pairs, no repeated pair_id, every row complete. A half-uploaded or
hand-edited file stops here, not after the GPU is running.

Each row is {prompt: [system, user], preferred, not_preferred, source, origin, pair_id, preferred_score, not_preferred_score, ...}.
`origin` names the run that made the pair (full-1 = documents; numerical-N = the Excel sets).
"""
import json
from pathlib import Path

from sft_common.s3_fetch import fetch_file

REQUIRED = ("prompt", "preferred", "not_preferred", "source", "origin", "pair_id")


def _read(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, text in enumerate(f):
            if text.strip():
                r = json.loads(text)
                r["line"] = n
                rows.append(r)
    return rows


def check_against_manifest(rows: list[dict], manifest: dict | None) -> list[str]:
    """Problems that make the file unsafe to train on (empty list = fine)."""
    problems = []
    ids = [r.get("pair_id") for r in rows]
    if len(set(ids)) != len(ids):
        problems.append(f"{len(ids) - len(set(ids))} repeated pair_id values: the same pair would be trained on twice")
    bad = sum(any(not r.get(k) for k in REQUIRED) for r in rows)
    if bad:
        problems.append(f"{bad} rows are missing one of {REQUIRED}")
    if manifest is not None:
        if manifest.get("total_pairs") != len(rows):
            problems.append(f"the manifest says {manifest.get('total_pairs')} pairs but the file has {len(rows)}: an incomplete or edited upload")
        by_origin = {b["origin"]: b["pairs"] for b in manifest.get("batches", [])}
        seen = {}
        for r in rows:
            seen[r["origin"]] = seen.get(r["origin"], 0) + 1
        if seen != by_origin:
            problems.append(f"pairs per origin differ from the manifest: file {seen} vs manifest {by_origin}")
    return problems


def load_pairs(data_cfg: dict) -> tuple[list[dict], list[str]]:
    """Download (if an S3 uri is set) and verify the pairs file. Raises SystemExit if it fails the manifest check."""
    local = Path(data_cfg["local_path"])
    if data_cfg.get("s3_uri"):
        fetch_file(data_cfg["s3_uri"], local)
    elif not local.exists():
        raise SystemExit(f"No pairs file: set dpo.data.s3_uri in config.yaml (or put the JSONL at {local})")
    manifest = None
    if data_cfg.get("manifest_s3_uri"):
        manifest_path = local.with_name("manifest.json")
        fetch_file(data_cfg["manifest_s3_uri"], manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read(local)
    problems = check_against_manifest(rows, manifest)
    if problems:
        raise SystemExit("[pairs source] the pairs file is not safe to train on:\n  " + "\n  ".join(problems))
    origins = {}
    for r in rows:
        origins[r["origin"]] = origins.get(r["origin"], 0) + 1
    report = [f"[pairs source] {len(rows)} pairs from {local}" + (" (checked against the manifest)" if manifest else " (no manifest to check against)"),
              "    origins: " + ", ".join(f"{k}={v}" for k, v in sorted(origins.items()))]
    return rows, report
