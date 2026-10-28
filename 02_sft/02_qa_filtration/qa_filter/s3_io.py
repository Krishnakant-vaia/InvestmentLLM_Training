import json
import time
from typing import Iterator

import boto3


def _client():
    return boto3.client("s3")


def stream_jsonl_lines(bucket: str, key: str, max_retries: int = 5) -> Iterator[str]:
    """Streams a large S3 object line by line, resuming from the exact byte
    offset already consumed if the connection drops mid-stream.

    A single GetObject holds one HTTP connection open for as long as the read
    takes -- for an object this large (many hours at real throughput), a
    transient network blip anywhere during that window is basically guaranteed
    at some point and would otherwise kill the whole run (observed in practice: a
    ConnectionResetError partway through shard 1 of a real run). Restarting
    from byte 0 on every blip would mean re-downloading gigabytes just to skip
    shards the checkpoint already knows are done, so this tracks `offset` (total
    bytes actually pulled off the network, including whatever's still sitting
    unprocessed in `buffer`) and reissues GetObject with `Range: bytes={offset}-`
    to continue exactly where the failed read left off -- byte-exact, so no line
    is skipped or duplicated across a resume.
    """
    client = _client()
    total_size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    offset = 0
    buffer = b""
    retries = 0

    while offset < total_size:
        try:
            response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-")
            for chunk in response["Body"].iter_chunks(chunk_size=1024 * 1024):
                offset += len(chunk)
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        yield line.decode("utf-8")
                retries = 0  # a chunk landed cleanly -- this attempt is healthy again
        except Exception:
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(min(2 ** retries, 30))

    if buffer.strip():
        yield buffer.decode("utf-8")


def upload_jsonl(bucket: str, key: str, local_path: str) -> None:
    # upload_file (not put_object with the whole file read into memory) -- boto3's
    # S3Transfer manager automatically switches to multipart upload above its
    # threshold (default 8MB). A single put_object caps out at 5GB per S3's own
    # limit and fails outright above that -- hit in practice: kept.jsonl reached
    # 5.85GB across the full ~1.1M-record run and the final upload failed with
    # EntityTooLarge, only after every shard had already finished successfully.
    client = _client()
    client.upload_file(local_path, bucket, key)


def upload_json(bucket: str, key: str, data: dict) -> None:
    client = _client()
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2).encode("utf-8"))
