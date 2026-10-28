"""Generates the --doc-keys-from input file pretune_dump.py needs for a real
S3 run: one s3://bucket/key doc_key per line, filtered to real document
files (.md by default, this corpus's actual format).

Standalone script, no import dependency on pretune_dump.py, consistent with
every other utility in this project.
"""

import argparse
import os


def list_corpus_doc_keys(bucket: str, prefix: str, s3_client, extensions: set[str] = {".md"}) -> list[str]:
    """Paginates the given S3 prefix and returns each matching object as an
    s3://bucket/key doc_key string — exactly the format pretune_dump.py's
    --doc-keys-from expects, one per line when written to a file."""
    doc_keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if os.path.splitext(key)[1].lower() in extensions:
                doc_keys.append(f"s3://{bucket}/{key}")
    return doc_keys


def main():
    parser = argparse.ArgumentParser(
        description="List real document doc_keys under an S3 prefix, for pretune_dump.py's --doc-keys-from.",
    )
    parser.add_argument("--bucket", default="ft-vaia", help="S3 bucket (default: ft-vaia)")
    parser.add_argument("--prefix", required=True, help="S3 prefix to scan")
    parser.add_argument("--out", required=True, help="Output file path, one doc_key per line")
    args = parser.parse_args()

    import boto3
    s3_client = boto3.client("s3")
    doc_keys = list_corpus_doc_keys(args.bucket, args.prefix, s3_client)

    with open(args.out, "w", encoding="utf-8") as f:
        for doc_key in doc_keys:
            f.write(doc_key + "\n")

    print(f"Found {len(doc_keys)} document(s) under s3://{args.bucket}/{args.prefix}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
