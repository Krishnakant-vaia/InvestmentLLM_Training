"""Access to the pre-tuned full-weights model on S3.

Interface: `ensure_local_model(cfg)` (weights + metadata), `load_tokenizer(cfg)` (metadata
only, ~20 MB), `fix_generation_config(dir)`. All results land in `model.pretuned.local_dir`.
"""
import json
import sys
from pathlib import Path

from .config import resolve
from .s3_models import WEIGHT_EXT, classify, download_folder, list_objects, parse_uri


def locate(cfg: dict, s3) -> tuple[str, dict, list[dict]]:
    """Return (bucket, chosen_folder_record, all_records). Exits if nothing usable."""
    pre = cfg["model"]["pretuned"]
    bucket, prefix = parse_uri(pre["s3_prefix"])
    records = classify(list_objects(s3, bucket, prefix))
    fulls = [r for r in records if r["kind"] == "full_weights"]
    if pre.get("full_weights_subdir"):
        want = pre["full_weights_subdir"].rstrip("/")
        fulls = [r for r in records if r["folder"].rstrip("/") == want]
    if not fulls:
        sys.exit(f"No full-weights folder under s3://{bucket}/{prefix}. Found: "
                 + json.dumps([(r["folder"], r["kind"], r["weight_gb"]) for r in records]))
    return bucket, fulls[0], records


def fix_generation_config(model_dir: str | Path) -> None:
    """Make generation stop at <|im_end|> as well as <|endoftext|> (idempotent).

    The pre-tuned model lists only <|endoftext|>, but chat turns end with <|im_end|>,
    so without this HF/vLLM would run past the end of a turn.
    """
    from transformers import AutoTokenizer
    d = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(d)
    ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    f = d / "generation_config.json"
    gen = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if gen.get("eos_token_id") != ids:
        gen["eos_token_id"] = ids
        f.write_text(json.dumps(gen, indent=2), encoding="utf-8")


def _is_meta(name: str) -> bool:
    return not name.endswith(WEIGHT_EXT)


def _download(cfg: dict, meta_only: bool) -> Path:
    import boto3
    local = resolve(cfg["model"]["pretuned"]["local_dir"])
    s3 = boto3.client("s3")
    bucket, chosen, _ = locate(cfg, s3)
    what = "metadata" if meta_only else f"weights ({chosen['weight_gb']} GB)"
    print(f"Downloading {what} from s3://{bucket}/{chosen['folder']} -> {local}")
    download_folder(s3, bucket, chosen["folder"], local, only=_is_meta if meta_only else None)
    if not chosen["has_tokenizer"]:
        # Merged folders sometimes omit the tokenizer; borrow it from the public base model.
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(cfg["model"]["base"], trust_remote_code=True).save_pretrained(local)
    fix_generation_config(local)
    return local


def ensure_tokenizer_files(cfg: dict) -> Path:
    local = resolve(cfg["model"]["pretuned"]["local_dir"])
    if not ((local / "tokenizer.json").exists() and (local / "chat_template.jinja").exists()):
        _download(cfg, meta_only=True)
    return local


def ensure_local_model(cfg: dict) -> Path:
    """Make sure the pre-tuned weights + metadata are on disk; return the local path."""
    local = resolve(cfg["model"]["pretuned"]["local_dir"])
    if (local / "config.json").exists() and any(local.glob("*.safetensors")):
        return local
    return _download(cfg, meta_only=False)


def load_tokenizer(cfg: dict):
    """Tokenizer (with chat template) of the pre-tuned model; needs no GPU or weights."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ensure_tokenizer_files(cfg))
