"""Online serving on Modal: OpenAI-compatible streaming API backed by vLLM.

Deploy:   modal deploy 08_model_deployment/serve.py
Endpoint: https://<workspace>--sft-serve-serve.modal.run/v1/chat/completions
Secrets (create once):
    modal secret create inference-api-key INFERENCE_API_KEY=<token>
Callers send `Authorization: Bearer <token>`. Streaming works with "stream": true.
"""
import os
import subprocess

import modal

GPU = os.environ.get("MODAL_INFERENCE_GPU", "A100-80GB")
VOLUME_NAME, MODEL_DIR = "sft-models", "/models/merged"  # keep in sync with config.yaml `deployment`
SERVED_NAME, PORT = "vaia-qwen35-9b", 8000
# Prompt + answer per request. The model's native context; only 8 of 32 layers keep a KV cache
# (32 KB/token), so a full-length request needs ~8.6 GB. Needs an 80 GB GPU: on smaller cards
# (e.g. L4, 24 GB) the 18 GB of weights leave too little room and vLLM refuses to start.
MAX_MODEL_LEN = 262144

app = modal.App("sft-serve")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm", "transformers", "boto3")


@app.function(image=image, volumes={"/models": volume}, secrets=[modal.Secret.from_name("aws-creds")], timeout=3600)
def download_model(s3_uri: str):
    """Copy a merged model from S3 into the volume (alternative to `modal volume put`)."""
    from urllib.parse import urlparse

    import boto3
    u = urlparse(s3_uri)
    prefix, s3 = u.path.strip("/"), boto3.client("s3")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=u.netloc, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):].lstrip("/")
            if not rel:
                continue
            dest = os.path.join(MODEL_DIR, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(u.netloc, obj["Key"], dest)
    volume.commit()


@app.function(
    image=image, gpu=GPU, volumes={"/models": volume},
    secrets=[modal.Secret.from_name("inference-api-key")],
    scaledown_window=600, timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=900)
def serve():
    """Start vLLM's OpenAI server; Modal proxies it and handles streaming."""
    cmd = [
        "vllm", "serve", MODEL_DIR,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", str(MAX_MODEL_LEN),
        "--api-key", os.environ["INFERENCE_API_KEY"],
    ]
    subprocess.Popen(cmd)
