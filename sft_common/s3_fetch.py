"""Fetch things from S3 by URI: one file, or a finished merged-model folder.

Used by DPO, whose inputs are a merged SFT model and a pairs file that are not part of any run folder.
"""
from pathlib import Path

from .pretuned import fix_generation_config
from .s3_models import download_folder, parse_uri
from .s3_sync import COMPLETE_MARKER


def fetch_file(uri: str, dest: Path, client=None) -> Path:
    """Download one S3 object to `dest` (skipped when a file of the same size is already there)."""
    bucket, key = parse_uri(uri)
    if client is None:
        import boto3
        client = boto3.client("s3")
    size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[s3] downloading {uri} ({size / 1e6:.1f} MB) -> {dest}", flush=True)
    client.download_file(bucket, key, str(dest))
    return dest


def fetch_model(uri: str, dest: Path, client=None) -> Path:
    """Download a merged model folder (weights, config, tokenizer) and return its local path.

    Refuses a folder without its `_COMPLETE` marker (an unfinished upload) and one without weights.
    """
    dest = Path(dest)
    if (dest / "config.json").exists() and any(dest.glob("*.safetensors")):
        return dest
    bucket, prefix = parse_uri(uri)
    prefix = prefix.rstrip("/")
    if client is None:
        import boto3
        client = boto3.client("s3")
    marker = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/{COMPLETE_MARKER}")
    if not marker.get("KeyCount"):
        raise SystemExit(f"{uri} has no {COMPLETE_MARKER} marker: the model upload never finished, or the path is wrong")
    print(f"[s3] downloading model {uri} -> {dest}", flush=True)
    download_folder(client, bucket, prefix, dest, only=lambda name: name != COMPLETE_MARKER)
    if not any(dest.glob("*.safetensors")):
        raise SystemExit(f"no .safetensors weights found under {uri}")
    fix_generation_config(dest)
    return dest
