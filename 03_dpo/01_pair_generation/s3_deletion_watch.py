"""Watch the project's S3 folders for deletions of any kind, using the bucket's own version history (read only).

    python s3_deletion_watch.py [--every 900]

CloudTrail is not readable with this IAM user (cloudtrail:LookupEvents is denied), so this watches what S3 itself records. The bucket is
versioned, so:
  * a normal delete adds a DELETE MARKER to the key;
  * a permanent delete removes a VERSION (or an object) that was there before.
At start it takes a snapshot of every (key, version) under the watched prefixes; every `--every` seconds it lists them again and prints
one line for each new delete marker and each version that has disappeared. It prints nothing when nothing changed (so a Monitor stays
quiet), apart from the first line. It never writes or deletes anything.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import config
import s3util

WATCH = ("Corpus/Invest_dataset/100626_Dataset_DPO/", "Corpus/Invest_dataset/092826_SFT_FineTune_Model/")


def snapshot(s3, prefixes=WATCH) -> tuple[set, set]:
    """(versions, delete_markers): sets of (key, version_id) under the prefixes."""
    versions, markers = set(), set()
    for prefix in prefixes:
        for page in s3.get_paginator("list_object_versions").paginate(Bucket=config.BUCKET, Prefix=prefix):
            versions |= {(v["Key"], v["VersionId"]) for v in page.get("Versions", [])}
            markers |= {(d["Key"], d["VersionId"]) for d in page.get("DeleteMarkers", [])}
    return versions, markers


def changes(before: tuple[set, set], after: tuple[set, set]) -> list[str]:
    """One line per new delete marker and per version that has vanished."""
    out = [f"NEW DELETE MARKER (deleted): {k}" for k, _ in sorted(after[1] - before[1])]
    out += [f"VERSION GONE (permanently removed): {k} [{v}]" for k, v in sorted(before[0] - after[0])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=900, help="seconds between checks (default 15 min)")
    a = ap.parse_args()
    s3 = s3util.client()
    base = snapshot(s3)
    print(f"[s3 watch] started {datetime.now(timezone.utc):%H:%M} UTC: {len(base[0])} versions and {len(base[1])} existing delete markers "
          f"under {len(WATCH)} folders; checking every {a.every // 60} min, silent unless something is deleted", flush=True)
    while True:
        time.sleep(a.every)
        try:
            now = snapshot(s3)
        except Exception as e:  # noqa: BLE001 - a failed listing is reported, never fatal
            print(f"[s3 watch] check failed ({type(e).__name__}): will retry", flush=True)
            continue
        for line in changes(base, now):
            print(f"[s3 watch] {datetime.now(timezone.utc):%H:%M} UTC  {line}", flush=True)
        base = now                                                       # compare each check with the one before it


if __name__ == "__main__":
    sys.exit(main())
