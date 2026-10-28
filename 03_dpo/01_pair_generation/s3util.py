"""Small S3 helpers shared by the scripts. Credentials come from the environment (.env)."""
from pathlib import Path

import boto3

import config


def client():
    return boto3.client("s3")


def list_keys(s3, prefix: str, bucket: str = config.BUCKET) -> list[str]:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


def get_text(s3, key: str, bucket: str = config.BUCKET) -> str:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "replace")


def exists(s3, key: str, bucket: str = config.BUCKET) -> bool:
    return bool(s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1).get("KeyCount"))


def upload(s3, local: Path, key: str, bucket: str = config.BUCKET) -> None:
    s3.upload_file(str(local), bucket, key)


def download(s3, key: str, local: Path, bucket: str = config.BUCKET) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local))


# Where each of a run's files lives on S3: <OUT_ROOT><run>/<stage>/<name>. Anything not listed here (corpus files, console logs)
# is stored directly under the prefix it is uploaded with.
STAGE_DIR = {
    "assignments.jsonl": "generation", "pairs_raw.jsonl": "generation", "pairs_unusable.jsonl": "generation",
    "calls_done.jsonl": "generation", "errors.jsonl": "generation", "first_response.json": "generation",
    "judge_out.jsonl": "judge",
    "scored_pairs.jsonl": "final", "dpo_pairs.jsonl": "final", "report.json": "final",
    "server.log": "logs",
}


def stage_key(prefix: str, name: str) -> str:
    """S3 key of file `name` under `prefix` (a run name, or e.g. 'corpus'): inside its stage folder if it has one."""
    stage = STAGE_DIR.get(name)
    return f"{config.OUT_ROOT}{prefix}/{stage + '/' if stage else ''}{name}"


def put_text(s3, key: str, text: str, bucket: str = config.BUCKET) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def copy(s3, src_key: str, dst_key: str, bucket: str = config.BUCKET) -> None:
    """Server-side copy; the source is left in place."""
    s3.copy_object(Bucket=bucket, Key=dst_key, CopySource={"Bucket": bucket, "Key": src_key})


def upload_files(files, run_prefix: str, s3=None) -> list[str]:
    """Best-effort upload of local files to <OUT_ROOT><run_prefix>/[<stage>/]<name>; returns the names that failed."""
    s3, failed = s3 or client(), []
    for f in files:
        f = Path(f)
        if not f.exists():
            continue
        try:
            upload(s3, f, stage_key(run_prefix, f.name))
        except Exception as e:  # noqa: BLE001 - a failed progress upload must never kill a long run
            print(f"[s3] upload of {f.name} failed: {type(e).__name__}: {e}", flush=True)
            failed.append(f.name)
    return failed
