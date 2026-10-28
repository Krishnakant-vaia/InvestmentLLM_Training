"""Locate and download the pre-tuned full-weights model from S3.

    python 02_training/fetch_pretuned.py --list   # report what is under the prefix
    python 02_training/fetch_pretuned.py --meta   # config/tokenizer/template only (~20 MB)
    python 02_training/fetch_pretuned.py          # weights + metadata (~18 GB)

Auto-picks the largest full-weights folder. If several exist, set
`model.pretuned.full_weights_subdir` in config.yaml (path relative to the bucket root).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.config import load_config
from sft_common.pretuned import ensure_local_model, ensure_tokenizer_files, locate

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="only report, do not download")
    g.add_argument("--meta", action="store_true", help="download metadata only, not weights")
    a = ap.parse_args()
    cfg = load_config()
    if a.list:
        import boto3
        bucket, chosen, records = locate(cfg, boto3.client("s3"))
        for r in records:
            print(f"{r['kind']:16} {r['weight_gb']:7} GB  s3://{bucket}/{r['folder']}/")
        print(f"\nChosen: s3://{bucket}/{chosen['folder']}/")
    elif a.meta:
        print("Local metadata:", ensure_tokenizer_files(cfg))
    else:
        print("Local model:", ensure_local_model(cfg))
