"""Structured, resumable judge checkpoints on S3.

    <OUT_ROOT><run>/judge/manifest.json              settings, input fingerprint, counts, status
    <OUT_ROOT><run>/judge/verdicts/part-00001.jsonl  the verdicts, about 50 batches (~500 pairs) per part
    <OUT_ROOT><run>/judge/verdicts/part-00002.jsonl  ...

Every part is written once and never overwritten, so an interrupted upload can only lose the newest part. `pull` brings all the
parts back (and refuses to mix them with a different judge input); `push` uploads only the verdicts that are not on S3 yet.
The manifest says whether the stage is complete. A part line is the judge's own verdict: {"id", "a", "b", "effort"}.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import config
import s3util

PART_PREFIX = "verdicts/part-"


def fingerprint(path: Path) -> str:
    """sha256 of the judge input file. Verdict ids are positions in that file, so verdicts only belong to this exact input."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def judge_prefix(run: str) -> str:
    return f"{config.OUT_ROOT}{run}/judge/"


def _valid_lines(text: str) -> dict[int, str]:
    """id -> raw line for every complete JSON verdict line (a half-written line is skipped)."""
    out = {}
    for line in text.split("\n"):
        if line.strip():
            try:
                out[json.loads(line)["id"]] = line
            except (ValueError, KeyError):
                continue
    return out


def read_manifest(s3, run: str) -> dict | None:
    key = judge_prefix(run) + "manifest.json"
    return json.loads(s3util.get_text(s3, key)) if s3util.exists(s3, key) else None


def pull(s3, run: str, out_path: Path, input_sha: str) -> tuple[set[int], int, dict | None]:
    """Merge every saved part into the local verdict file. Returns (ids saved on S3, number of parts, manifest or None).
    Raises RuntimeError if the saved verdicts were made for a different judge input."""
    manifest = read_manifest(s3, run)
    if manifest and manifest.get("input_sha256") not in (None, input_sha):
        raise RuntimeError(f"the judge checkpoints of run {run} were made for a different judge input "
                           f"(fingerprint {str(manifest.get('input_sha256'))[:12]} vs {input_sha[:12]}); refusing to mix them. "
                           f"Use a new run name, or remove {judge_prefix(run)}")
    parts = sorted(k for k in s3util.list_keys(s3, judge_prefix(run) + PART_PREFIX))
    saved: dict[int, str] = {}
    for key in parts:
        saved.update(_valid_lines(s3util.get_text(s3, key)))
    local = _valid_lines(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    merged = {**saved, **local}
    if merged:
        out_path.write_text("\n".join(merged[i] for i in sorted(merged)) + "\n", encoding="utf-8")
    print(f"[checkpoint] {len(saved)} verdicts restored from {len(parts)} parts on S3; {len(merged)} verdicts locally", flush=True)
    return set(saved), len(parts), manifest


def push(s3, run: str, out_path: Path, saved_ids: set[int], n_parts: int, meta: dict, complete: bool = False) -> int:
    """Upload the verdicts not on S3 yet as a new part and refresh the manifest. Returns the new number of parts.
    A failed upload is reported, not raised: the run goes on and the next push retries the same verdicts."""
    try:
        local = _valid_lines(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
        new = {i: line for i, line in local.items() if i not in saved_ids}
        if new:
            key = judge_prefix(run) + f"{PART_PREFIX}{n_parts + 1:05d}.jsonl"
            s3util.put_text(s3, key, "\n".join(new[i] for i in sorted(new)) + "\n")
            saved_ids |= set(new)
            n_parts += 1
        manifest = {**meta, "judged": len(saved_ids), "parts": n_parts, "complete": complete,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        s3util.put_text(s3, judge_prefix(run) + "manifest.json", json.dumps(manifest, indent=2))
        print(f"[checkpoint] {len(saved_ids)}/{meta.get('total')} verdicts saved on S3 in {n_parts} parts"
              + (" (complete)" if complete else ""), flush=True)
    except Exception as e:  # noqa: BLE001 - a failed progress upload must never kill a long run
        print(f"[checkpoint] upload failed ({type(e).__name__}: {e}); will retry at the next checkpoint", flush=True)
    return n_parts
