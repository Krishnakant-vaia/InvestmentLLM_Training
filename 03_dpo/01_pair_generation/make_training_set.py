"""Add a batch of pairs to the DPO training set on S3: the file the trainer is pointed at.

    python make_training_set.py --run full-1 [--dry-run] [--tokenizer-dir PATH]

Reads <OUT_ROOT><run>/final/scored_pairs.jsonl (every pair, with the judge's verdicts and flags), labels each pair SEND, HOLD or
EXCLUDE, and appends the SEND pairs, in the trainer's format, to
    <OUT_ROOT>training/dpo_train.jsonl        one pair per line: prompt, preferred, not_preferred, source + provenance
    <OUT_ROOT>training/manifest.json          the selection rule and one entry per batch (origin, counts, score distributions)
Nothing is removed from the run: scored_pairs.jsonl stays the complete dataset. A run can be added once; a later batch from another
source (for example the templated numerical pairs) is added the same way, with its own origin name.

SEND = the preferred answer passes the judge (score >= 8), no broken-format flag, the not-preferred answer is a real mistake the judge
caught (score < 7) OR a padded counter-pair (correct but too long, which the judge passes on purpose), and the two answers are within
2x of each other in tokens (the trainer drops anything beyond that).
HOLD = fixable: the answers differ by more than 2x, or the not-preferred answer passed the judge although it should be a mistake.
EXCLUDE = the preferred answer fails the judge, or a broken-format flag (giveaway phrase, identical answers, too short, bad numbers).
"""
import argparse
import json
import statistics as st
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import build_dpo_pairs
import config
from schema import normalize_row
import s3util

TRAIN_KEY = f"{config.OUT_ROOT}training/dpo_train.jsonl"
MANIFEST_KEY = f"{config.OUT_ROOT}training/manifest.json"
MIN_PREFERRED_SCORE = 8
NOT_PREFERRED_FAILS_BELOW = 7
TOKEN_RATIO = (0.5, 2.0)                      # not-preferred / preferred, whole answer incl. reasoning: the trainer's own 2x rule
CRITICAL_FLAGS = ("preferred identical to not-preferred", "preferred answer too short", "not-preferred answer has a tell", "numeric:",
                  "missing field", "thinking not wrapped")
RULE = {"preferred_judge_score_at_least": MIN_PREFERRED_SCORE,
        "not_preferred_judge_score_below": NOT_PREFERRED_FAILS_BELOW,
        "padded_counter_pairs_allowed_when_not_preferred_passes": True,
        "token_ratio_not_preferred_over_preferred": list(TOKEN_RATIO),
        "excluded_flags": list(CRITICAL_FLAGS), "length_cap_on_preferred_longer_share": None}


def label(row: dict, ratio: float) -> tuple[str, list[str]]:
    """(SEND | HOLD | EXCLUDE, reasons) for one scored pair; `ratio` = not-preferred tokens / preferred tokens."""
    chosen, rejected = row.get("judge_preferred"), row.get("judge_not_preferred")
    if not chosen or not rejected:
        return "EXCLUDE", ["no judge verdict"]
    reasons = []
    if chosen["score"] < MIN_PREFERRED_SCORE:
        reasons.append("preferred answer fails the judge")
    reasons += [f"broken format: {c.rstrip(':')}" for c in CRITICAL_FLAGS if any(f.startswith(c) for f in row.get("flags", []))]
    if reasons:
        return "EXCLUDE", reasons
    if not (rejected["score"] < NOT_PREFERRED_FAILS_BELOW or row.get("defect_type") == "padded"):
        reasons.append("not-preferred answer passes the judge (no mistake to learn from)")
    if not TOKEN_RATIO[0] <= ratio <= TOKEN_RATIO[1]:
        reasons.append("answers differ more than 2x in length (the trainer would drop it)")
    return ("HOLD", reasons) if reasons else ("SEND", ["preferred passes the judge; not-preferred is a caught mistake or a padded counter-pair; lengths within 2x"])


def numerical_rows(s3, run: str) -> list[dict]:
    """The numerical run's judged pairs (<run>/<sheet>/<capability>/final/judged_pairs.jsonl) in the scored-pair schema the rest of
    this module reads: no document, no <think> block, the golden answer is the preferred one."""
    rows = []
    for key in sorted(k for k in s3util.list_keys(s3, f"{config.OUT_ROOT}{run}/") if k.endswith("/final/judged_pairs.jsonl")):
        for line in s3util.get_text(s3, key).splitlines():
            if line.strip():
                r = normalize_row(json.loads(line), accepted_is_answer=True)          # old files used accepted / rejected
                rows.append({"document": "", "question": r["question"], "question_type": r["question_type"], "defect_type": "numerical",
                             "folder": r["capability"], "preferred_thinking": "", "preferred": str(r["preferred"]), "not_preferred_thinking": "",
                             "not_preferred": str(r["not_preferred"]), "chunk_id": r["pair_id"], "call_id": r["pair_id"],
                             "judge_preferred": r["judge_preferred"], "judge_not_preferred": r["judge_not_preferred"], "flags": r["flags"],
                             "passes_all_gates": not r["flags"]})
    return rows


def training_line(row: dict, system: str, run: str, batch: int) -> dict:
    """One pair in the trainer's format plus provenance."""
    pair = build_dpo_pairs.to_dpo_pair(row, system)
    pair.update({"origin": run, "batch": batch, "pair_id": f"{run}:{row['call_id']}:{row['question_type']}",
                 "preferred_score": row["judge_preferred"]["score"], "not_preferred_score": row["judge_not_preferred"]["score"]})
    return pair


def summarize(lines: list[dict]) -> dict:
    words = lambda t: len(t.split())  # noqa: E731
    return {"pairs": len(lines), "by_source": dict(Counter(p["source"] for p in lines)),
            "by_defect": dict(Counter(p["defect_type"] for p in lines)),
            "by_question_type": dict(Counter(p["question_type"] for p in lines)),
            "preferred_scores": dict(sorted(Counter(p["preferred_score"] for p in lines).items())),
            "not_preferred_scores": dict(sorted(Counter(p["not_preferred_score"] for p in lines).items())),
            "preferred_longer_share": round(sum(words(p["preferred"]) > words(p["not_preferred"]) for p in lines) / max(1, len(lines)), 3)}


def add_batch(s3, run: str, rows: list[dict], count_tokens, system: str, dry_run: bool = False) -> dict:
    """Label `rows`, append the SEND pairs to the training file and the manifest. Returns this batch's manifest entry."""
    manifest = json.loads(s3util.get_text(s3, MANIFEST_KEY)) if s3util.exists(s3, MANIFEST_KEY) else {
        "file": "training/dpo_train.jsonl", "rule": RULE, "batches": [], "total_pairs": 0}
    if any(b["origin"] == run for b in manifest["batches"]):
        raise SystemExit(f"run {run} is already in the training set (batch {[b['batch'] for b in manifest['batches'] if b['origin'] == run][0]}); "
                         "it is added once so the same pairs are never trained on twice")
    batch, labels, why, lines = len(manifest["batches"]) + 1, Counter(), Counter(), []
    for r in rows:
        probe = build_dpo_pairs.to_dpo_pair(r, "") if r.get("judge_preferred") else None
        ratio = count_tokens(probe["not_preferred"]) / max(1, count_tokens(probe["preferred"])) if probe else 1.0
        lab, reasons = label(r, ratio)
        labels[lab] += 1
        if lab != "SEND":
            why.update(f"{lab}: {x}" for x in reasons)
        else:
            lines.append(training_line(r, system, run, batch))
    entry = {"batch": batch, "origin": run, "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "scored_pairs_read": len(rows), "labels": dict(labels), "not_added_because": dict(why.most_common()), **summarize(lines)}
    if dry_run:
        return entry
    existing = s3util.get_text(s3, TRAIN_KEY) if s3util.exists(s3, TRAIN_KEY) else ""
    existing = existing if not existing or existing.endswith("\n") else existing + "\n"
    s3util.put_text(s3, TRAIN_KEY, existing + "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in lines))
    manifest["batches"].append(entry)
    manifest["total_pairs"] += len(lines)
    manifest["updated_at"] = entry["added_at"]
    s3util.put_text(s3, MANIFEST_KEY, json.dumps(manifest, indent=2))
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run whose final/scored_pairs.jsonl is read (also the batch's origin name)")
    ap.add_argument("--numerical", action="store_true", help="--run is a numerical run (judged_pairs.jsonl per capability), not a document run")
    ap.add_argument("--dry-run", action="store_true", help="count and report only; write nothing")
    ap.add_argument("--tokenizer-dir", default=str(config.HERE.parent.parent / "02_sft" / "03_training" / "models" / "pretuned"))
    ap.add_argument("--system-prompt", default=str(config.HERE / "prompts" / "system_prompt.txt"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    s3 = s3util.client()
    if a.numerical:
        rows = numerical_rows(s3, a.run)
    else:
        text = s3util.get_text(s3, s3util.stage_key(a.run, "scored_pairs.jsonl"))
        rows = [normalize_row(json.loads(line)) for line in text.splitlines() if line.strip()]
    entry = add_batch(s3, a.run, rows, lambda t: len(tok(t)["input_ids"]), Path(a.system_prompt).read_text(encoding="utf-8"), a.dry_run)
    print(json.dumps(entry, indent=2))
    print(("DRY RUN: nothing written. " if a.dry_run else f"Added to s3://{config.BUCKET}/{TRAIN_KEY} . ") + f"{entry['pairs']} pairs.")


if __name__ == "__main__":
    main()
