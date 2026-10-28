"""Verify AWS credentials from .env, that the S3 inputs in config.yaml are readable, and that
the run output folder (storage.s3_root) is writable.

    python 01_data_prep/check_aws.py
"""
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.config import load_config
from sft_common.s3_models import parse_uri

PROBE = "_access_check.txt"   # tiny object written (and overwritten on every check, never deleted), to prove write access


def main() -> int:
    cfg = load_config()  # also loads .env
    try:
        who = boto3.client("sts").get_caller_identity()
        print("Credentials OK:", who["Arn"])
    except NoCredentialsError:
        print("No credentials found. Fill in .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).")
        return 1
    except ClientError as e:
        print("Credentials rejected:", e.response["Error"]["Code"])
        return 1

    s3, ok = boto3.client("s3"), True
    # Inputs: must exist and be listable.
    for name, uri in [("training pool", cfg["data"]["pool"]["s3_uri"]),
                      ("pre-tuned model", cfg["model"]["pretuned"]["s3_prefix"])]:
        bucket, key = parse_uri(uri)
        try:
            n = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=5).get("KeyCount", 0)
            print(f"[{'ok' if n else 'EMPTY'}] {name}: {uri}")
            ok &= bool(n)
        except ClientError as e:
            print(f"[FAIL] {name}: {e.response['Error']['Code']}")
            ok = False

    # Output folder: every run saves checkpoints, adapter, merged model and eval results here. For this DPO project that is dpo.s3_root
    # (100626_Dataset_DPO/DPO_Run_1/), never the SFT folder. The probe is overwritten each time and NEVER deleted: no delete call at all.
    root = (cfg.get("dpo") or {}).get("s3_root") or cfg["storage"]["s3_root"]
    bucket, prefix = parse_uri(root)
    probe_key = f"{prefix.rstrip('/')}/{PROBE}"
    try:
        s3.put_object(Bucket=bucket, Key=probe_key, Body=b"write access check")
        print(f"[ok] run storage (writable): {root}")
    except ClientError as e:
        print(f"[FAIL] run storage not writable: {root} ({e.response['Error']['Code']})")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
