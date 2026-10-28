"""Download the training pool from S3.

Credentials come from the standard AWS env vars / profile (never hardcode keys):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.config import load_config, resolve


def download(s3_uri: str, dest: Path) -> Path:
    u = urlparse(s3_uri)
    bucket, key = u.netloc, u.path.lstrip("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    print(f"Downloading {s3_uri} ({size / 1e6:.1f} MB) -> {dest}")
    client.download_file(bucket, key, str(dest))
    return dest


if __name__ == "__main__":
    cfg = load_config()["data"]["pool"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s3-uri", default=cfg["s3_uri"])
    ap.add_argument("--dest", default=str(resolve(cfg["local_path"])))
    ap.add_argument("--force", action="store_true", help="re-download if file exists")
    a = ap.parse_args()
    if Path(a.dest).exists() and not a.force:
        print(f"Already present: {a.dest} (use --force to refresh)")
    else:
        download(a.s3_uri, Path(a.dest))
