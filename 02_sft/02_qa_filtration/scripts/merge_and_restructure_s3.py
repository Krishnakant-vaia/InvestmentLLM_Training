"""Final merge + S3 restructure, run once after both the main filtration run and
the shard 0-96 re-filter measurement have completed.

MERGE (local, then upload):
  new sft_data_kept.jsonl     = kept records already judged by gepa_strict_v2
                                (shards 97+, from run_state/kept.jsonl)
                                + every record from refilter_state/kept.jsonl
                                (shards 0-96 kept records that survived GEPA-strict)
  new sft_data_rejected.jsonl = everything in run_state/rejected.jsonl (unchanged --
                                records rejected under a weaker prompt are still
                                rejected under a stricter one, so they need no
                                re-judging)
                                + every record from refilter_state/rejected.jsonl
                                (shards 0-96 kept records that failed GEPA-strict)
  Result: the whole corpus judged under one consistent standard (gepa_strict_v2).

RESTRUCTURE S3 (base: Corpus/Invest_dataset/09032026_SFTQ&Apairs/):
  top level keeps ONLY sft_data_all_capabilities.jsonl (untouched source) and
  sft_data_kept.jsonl (the final merged deliverable). Everything else from this
  run moves under pc_krishnakant/qa_filtration_run/ (its own subfolder, so it
  can't collide with the generation-stage files already in pc_krishnakant/).

SAFETY: backs up the pre-merge kept/rejected/report to
pc_krishnakant/qa_filtration_run/backup_pre_merge/ BEFORE touching anything, checks
count invariants on the merged files BEFORE any upload, and only deletes old
top-level objects after the new copies are verified on S3 by size.
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qa_filter.config import load_settings
from qa_filter.s3_io import upload_json, upload_jsonl, _client

RUN_STATE = "./run_state"
REFILTER_STATE = "./refilter_state"
MERGED_DIR = "./merged_output"

# Known totals from the completed runs -- the merged files must reproduce these
# exactly or nothing gets uploaded.
EXPECTED_TOTAL_RECORDS = 1_118_795
EXPECTED_GEPA_KEPT_97_PLUS = 376_828
EXPECTED_REFILTER_KEPT = 270_238
EXPECTED_REFILTER_REJECTED = 89_107
EXPECTED_OLD_REJECTED = 382_622


def _copy_lines(src_path, dst_f, keep_predicate=None):
    """Streams src into dst_f; returns (lines_written, lines_seen)."""
    written = 0
    seen = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            if keep_predicate is not None and not keep_predicate(json.loads(line)):
                continue
            dst_f.write(line if line.endswith("\n") else line + "\n")
            written += 1
    return written, seen


def build_merged_files():
    os.makedirs(MERGED_DIR, exist_ok=True)
    kept_out = os.path.join(MERGED_DIR, "sft_data_kept.jsonl")
    rejected_out = os.path.join(MERGED_DIR, "sft_data_rejected.jsonl")

    print("building merged kept.jsonl ...", flush=True)
    with open(kept_out, "w", encoding="utf-8", newline="\n") as f:
        gepa_kept, _ = _copy_lines(
            os.path.join(RUN_STATE, "kept.jsonl"), f,
            keep_predicate=lambda r: r.get("_judge_prompt_version") == "gepa_strict_v2",
        )
        refilter_kept, _ = _copy_lines(os.path.join(REFILTER_STATE, "kept.jsonl"), f)
    print(f"  gepa_strict_v2 kept from main run (shards 97+): {gepa_kept:,}", flush=True)
    print(f"  re-filter kept (shards 0-96 that survived):     {refilter_kept:,}", flush=True)

    print("building merged rejected.jsonl ...", flush=True)
    with open(rejected_out, "w", encoding="utf-8", newline="\n") as f:
        old_rejected, _ = _copy_lines(os.path.join(RUN_STATE, "rejected.jsonl"), f)
        refilter_rejected, _ = _copy_lines(os.path.join(REFILTER_STATE, "rejected.jsonl"), f)
    print(f"  old rejected (unchanged):                       {old_rejected:,}", flush=True)
    print(f"  re-filter rejected (newly failed GEPA-strict):  {refilter_rejected:,}", flush=True)

    checks = {
        "gepa kept (97+)": (gepa_kept, EXPECTED_GEPA_KEPT_97_PLUS),
        "re-filter kept": (refilter_kept, EXPECTED_REFILTER_KEPT),
        "old rejected": (old_rejected, EXPECTED_OLD_REJECTED),
        "re-filter rejected": (refilter_rejected, EXPECTED_REFILTER_REJECTED),
        "total records": (gepa_kept + refilter_kept + old_rejected + refilter_rejected, EXPECTED_TOTAL_RECORDS),
    }
    failed = {name: (got, want) for name, (got, want) in checks.items() if got != want}
    if failed:
        print("\nINVARIANT CHECK FAILED -- aborting before any upload or S3 change:")
        for name, (got, want) in failed.items():
            print(f"  {name}: got {got:,}, expected {want:,}")
        sys.exit(1)
    print("\nall count invariants passed (total = 1,118,795 records preserved)", flush=True)

    # Every kept record must now carry the single unified standard.
    print("verifying every merged kept record is tagged gepa_strict_v2 ...", flush=True)
    bad = 0
    with open(kept_out, "r", encoding="utf-8") as f:
        for line in f:
            if json.loads(line).get("_judge_prompt_version") != "gepa_strict_v2":
                bad += 1
    if bad:
        print(f"  {bad:,} merged kept records are NOT gepa_strict_v2 -- aborting")
        sys.exit(1)
    print("  all merged kept records are gepa_strict_v2", flush=True)

    return kept_out, rejected_out, {
        "kept": gepa_kept + refilter_kept,
        "rejected": old_rejected + refilter_rejected,
        "gepa_kept_shards_97_plus": gepa_kept,
        "refilter_kept": refilter_kept,
        "old_rejected": old_rejected,
        "refilter_rejected": refilter_rejected,
    }


def s3_size(client, bucket, key):
    return client.head_object(Bucket=bucket, Key=key)["ContentLength"]


def s3_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False


def managed_copy(client, bucket, src_key, dst_key):
    # client.copy (managed transfer) does multipart server-side copy, so it works
    # above the 5GB single-request copy_object limit.
    client.copy({"Bucket": bucket, "Key": src_key}, bucket, dst_key)
    assert s3_size(client, bucket, src_key) == s3_size(client, bucket, dst_key), \
        f"size mismatch after copying {src_key} -> {dst_key}"


def main():
    settings = load_settings()
    bucket = settings.s3_bucket
    base = settings.output_prefix  # Corpus/Invest_dataset/09032026_SFTQ&Apairs/
    run_dir = base + "pc_krishnakant/qa_filtration_run/"
    backup_dir = run_dir + "backup_pre_merge/"
    client = _client()

    # ---- Phase 1: build + verify merged files locally (nothing on S3 touched yet)
    kept_out, rejected_out, counts = build_merged_files()

    # ---- Phase 2: backup current top-level files (server-side, no download)
    print("\nbacking up current top-level files ...", flush=True)
    for name in ["sft_data_kept.jsonl", "sft_data_rejected.jsonl", "filter_report.json"]:
        if not s3_exists(client, bucket, base + name):
            print(f"  skip (not present): {name}")
            continue
        # A backup, once it exists, is NEVER overwritten -- even if sizes differ.
        # After Phase 3 the top-level kept file is the NEW merged one, so a re-run
        # comparing sizes would "refresh" the backup with the new file and destroy
        # the only copy of the pre-merge original. S3 multipart copies aren't
        # visible until complete, so an existing backup object is a complete one.
        if s3_exists(client, bucket, backup_dir + name):
            print(f"  already backed up (left untouched): {name}")
            continue
        managed_copy(client, bucket, base + name, backup_dir + name)
        print(f"  backed up: {name}", flush=True)

    # ---- Phase 3: upload merged files
    print("\nuploading merged kept (overwrites top-level sft_data_kept.jsonl) ...", flush=True)
    upload_jsonl(bucket, base + "sft_data_kept.jsonl", kept_out)
    assert s3_size(client, bucket, base + "sft_data_kept.jsonl") == os.path.getsize(kept_out), "kept upload size mismatch"

    print("uploading merged rejected into pc_krishnakant/qa_filtration_run/ ...", flush=True)
    upload_jsonl(bucket, run_dir + "sft_data_rejected.jsonl", rejected_out)
    assert s3_size(client, bucket, run_dir + "sft_data_rejected.jsonl") == os.path.getsize(rejected_out), "rejected upload size mismatch"

    report = {
        "total_records": counts["kept"] + counts["rejected"],
        "kept": counts["kept"],
        "rejected": counts["rejected"],
        "pass_rate": counts["kept"] / (counts["kept"] + counts["rejected"]),
        "judge_standard": "gepa_strict_v2 (entire corpus, single consistent standard)",
        "lineage": {
            "gepa_strict_v2_kept_from_main_run_shards_97_plus": counts["gepa_kept_shards_97_plus"],
            "shards_0_96_kept_that_survived_gepa_strict_refilter": counts["refilter_kept"],
            "rejected_in_original_run_unchanged": counts["old_rejected"],
            "shards_0_96_kept_that_failed_gepa_strict_refilter": counts["refilter_rejected"],
        },
        "note": "shards 0-96 were originally judged under weaker prompts (original ~74% pass, "
                "max_stringent for shard 96) and every record they kept was re-judged under "
                "gepa_strict_v2 before this merge. Pre-merge originals are in backup_pre_merge/.",
    }
    upload_json(bucket, run_dir + "filter_report.json", report)

    # ---- Phase 4: move refilter_shards_0_96/ into the run folder
    print("\nmoving refilter_shards_0_96/ into pc_krishnakant/qa_filtration_run/ ...", flush=True)
    for name in ["refilter_kept.jsonl", "refilter_rejected.jsonl", "refilter_report.json"]:
        src = base + "refilter_shards_0_96/" + name
        if s3_exists(client, bucket, src):
            managed_copy(client, bucket, src, run_dir + "refilter_shards_0_96/" + name)
            client.delete_object(Bucket=bucket, Key=src)
            print(f"  moved: refilter_shards_0_96/{name}", flush=True)

    # ---- Phase 5: remove the old top-level copies (new copies verified above)
    print("\nremoving old top-level sft_data_rejected.jsonl + filter_report.json ...", flush=True)
    for name in ["sft_data_rejected.jsonl", "filter_report.json"]:
        if s3_exists(client, bucket, base + name):
            # only delete once the replacement exists in the run folder
            assert s3_exists(client, bucket, run_dir + name), f"replacement for {name} missing -- refusing to delete"
            client.delete_object(Bucket=bucket, Key=base + name)
            print(f"  removed: {name}", flush=True)

    print("\nDONE. final counts:", json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
