"""Find and fetch a full-weights model folder under an S3 prefix.

A prefix may hold several things (adapters, checkpoints, a merged model). We group
objects by folder and call a folder "full weights" only if it has config.json plus
weight files large enough to be a whole model.
"""
from collections import defaultdict
from pathlib import PurePosixPath
from urllib.parse import urlparse

WEIGHT_EXT = (".safetensors", ".bin")
MIN_FULL_BYTES = 5e9  # a 9B model in 16-bit is ~18 GB; adapters are far smaller


def parse_uri(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    return u.netloc, u.path.lstrip("/")


def classify(objects: list[tuple[str, int]]) -> list[dict]:
    """objects = [(key, size)]. Return one record per folder that holds model files."""
    folders: dict[str, dict] = defaultdict(lambda: {"files": [], "weight_bytes": 0})
    for key, size in objects:
        p = PurePosixPath(key)
        f = folders[str(p.parent)]
        f["files"].append(p.name)
        if p.name.endswith(WEIGHT_EXT):
            f["weight_bytes"] += size
    out = []
    for folder, f in folders.items():
        names = set(f["files"])
        if not any(n.endswith(WEIGHT_EXT) for n in names):
            continue
        is_adapter = "adapter_config.json" in names
        full = "config.json" in names and not is_adapter and f["weight_bytes"] >= MIN_FULL_BYTES
        out.append({
            "folder": folder,
            "kind": "full_weights" if full else "adapter" if is_adapter else "partial/unknown",
            "weight_gb": round(f["weight_bytes"] / 1e9, 2),
            "has_tokenizer": "tokenizer.json" in names or "tokenizer_config.json" in names,
            "files": sorted(names),
        })
    return sorted(out, key=lambda r: (r["kind"] != "full_weights", -r["weight_gb"]))


def list_objects(s3, bucket: str, prefix: str) -> list[tuple[str, int]]:
    objs = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        objs += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return objs


def download_folder(s3, bucket: str, folder: str, dest, only=None) -> int:
    """Download files directly inside `folder` (skips existing same-size files).

    `only(name) -> bool` optionally filters which files to fetch.
    """
    from pathlib import Path
    dest, n = Path(dest), 0
    prefix = folder.rstrip("/") + "/"
    for key, size in list_objects(s3, bucket, prefix):
        rel = key[len(prefix):]
        if not rel or "/" in rel or (only and not only(rel)):  # direct children only
            continue
        target = dest / rel
        if target.exists() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))
        n += 1
        print(f"  downloaded {rel} ({size / 1e6:.0f} MB)")
    return n
