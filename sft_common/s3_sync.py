"""S3 storage for one run. Every upload/download of a run goes through `RunStore`.

Layout under storage.s3_root (config.yaml), one folder per run:
    <run_name>/config/          config snapshot + pip freeze
    <run_name>/data/            stats.json, val.jsonl, train.jsonl
    <run_name>/logs/            startup report, fit analysis, trainer log history
    <run_name>/checkpoints/     checkpoint-500/, checkpoint-1000/, ... (every checkpoint is kept)
    <run_name>/final_adapter/   LoRA adapter after training
    <run_name>/eval/            step 4 results
Merged models (step 3) are NOT inside the run folder but next to it:
    merged_models/<run_name>_checkpoint-N/   everything for one merged model, in one folder:
        model-*.safetensors, config, tokenizer, chat template, merge_info.json, _COMPLETE, logs/
    merged_models/<run_name>_final/          the same, merged from the final adapter

Uploads never raise: a failure is printed and reported as False, so a network blip cannot
kill a multi-hour training run. Callers retry later (the final sync re-uploads anything missed).
"""
import time
from pathlib import Path

from .s3_models import parse_uri

COMPLETE_MARKER = "_COMPLETE"      # written last into a checkpoint folder: the upload finished
NEVER_UPLOAD = {".env"}            # secrets must never leave the machine
# Training-only files in a checkpoint: a merge needs just the adapter, so these are not downloaded.
TRAINING_STATE_FILES = ("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state")


MERGED_ROOT = "merged_models"      # merged models live here, next to (not inside) the run folders


def merged_name(run: str, checkpoint: int | None) -> str:
    """Folder name of a merged model: <run>_checkpoint-N, or <run>_final."""
    return f"{run}_checkpoint-{checkpoint}" if checkpoint is not None else f"{run}_final"


def merged_store(root_uri: str, run: str, checkpoint: int | None, client=None) -> "RunStore":
    """Store for one merged model: <storage.s3_root>/merged_models/<merged_name>/."""
    return RunStore(root_uri, f"{MERGED_ROOT}/{merged_name(run, checkpoint)}", client=client)


def is_adapter_file(name: str) -> bool:
    """True for the checkpoint files a merge needs (adapter, tokenizer, trainer_state for its eval loss)."""
    return not any(name.startswith(p) for p in TRAINING_STATE_FILES)


class RunStore:
    def __init__(self, root_uri: str, run_name: str, client=None, retries: int = 3, wait: float = 5.0):
        bucket, prefix = parse_uri(root_uri)
        self.bucket = bucket
        self.prefix = f"{prefix.rstrip('/')}/{run_name}" if prefix.strip("/") else run_name
        self._client = client               # injected in tests; real boto3 client created on first use
        self.retries, self.wait = retries, wait

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("s3")  # credentials come from .env / the AWS chain
        return self._client

    def key(self, rel: str) -> str:
        """S3 key for a path relative to the run folder."""
        return f"{self.prefix}/{rel.strip('/')}" if rel.strip("/") else self.prefix

    def uri(self, rel: str = "") -> str:
        return f"s3://{self.bucket}/{self.key(rel)}"

    # ---- uploads ---------------------------------------------------------------------------

    def upload_file(self, local: str | Path, rel: str) -> bool:
        """Upload one file with retries. Returns True on success, False (never raises) on failure."""
        local = Path(local)
        if local.name in NEVER_UPLOAD:
            return True                         # silently skipped by design
        for attempt in range(1, self.retries + 1):
            try:
                self.client.upload_file(str(local), self.bucket, self.key(rel))
                return True
            except Exception as e:  # noqa: BLE001 - any failure (network, auth) is retried then reported
                print(f"[s3] upload failed ({attempt}/{self.retries}) {local} -> {self.uri(rel)}: {e}")
                if attempt < self.retries:
                    time.sleep(self.wait * attempt)
        return False

    def upload_dir(self, local_dir: str | Path, rel: str) -> bool:
        """Upload every file under `local_dir` (recursively) to `rel/`. True if all succeeded."""
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            print(f"[s3] nothing to upload: {local_dir} does not exist")
            return False
        ok = True
        for f in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            ok &= self.upload_file(f, f"{rel}/{f.relative_to(local_dir).as_posix()}")
        return ok

    def upload_checkpoint(self, ckpt_dir: str | Path) -> bool:
        """Upload a checkpoint folder, then mark it complete. Half-uploaded ones stay unmarked."""
        ckpt_dir = Path(ckpt_dir)
        rel = f"checkpoints/{ckpt_dir.name}"
        if not self.upload_dir(ckpt_dir, rel):
            return False
        try:
            self.client.put_object(Bucket=self.bucket, Key=self.key(f"{rel}/{COMPLETE_MARKER}"), Body=b"")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[s3] could not mark {rel} complete: {e}")
            return False

    def upload_model(self, model_dir: str | Path) -> bool:
        """Upload a merged model into this store's folder, then mark it complete (_COMPLETE last).

        Only a merged_models/<name> store is allowed: a merge never writes into a run's folder
        (checkpoints/, logs/ ... belong to training).
        """
        parts = self.prefix.split("/")
        if len(parts) < 2 or parts[-2] != MERGED_ROOT:
            raise ValueError(f"merged models go under {MERGED_ROOT}/<name>, refusing to write to {self.uri()}")
        if not self.upload_dir(model_dir, ""):
            return False
        try:
            self.client.put_object(Bucket=self.bucket, Key=self.key(COMPLETE_MARKER), Body=b"")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[s3] could not mark {self.uri()} complete: {e}")
            return False

    # ---- listing / downloads ---------------------------------------------------------------

    def _keys(self, rel: str) -> list[str]:
        """All object keys under `rel/`, relative to it."""
        base = self.key(rel) + "/"
        out = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=base):
            out += [o["Key"][len(base):] for o in page.get("Contents", [])]
        return out

    def checkpoint_steps(self) -> list[int]:
        """Steps of the checkpoints fully uploaded to S3 (those with the completion marker)."""
        steps = []
        for k in self._keys("checkpoints"):
            name, _, rest = k.partition("/")
            if rest == COMPLETE_MARKER and name.startswith("checkpoint-") and name[11:].isdigit():
                steps.append(int(name[11:]))
        return sorted(steps)

    def download_dir(self, rel: str, local_dir: str | Path, keep=None) -> int:
        """Download everything under `rel/` into `local_dir` (only files whose path relative to
        `rel/` passes keep(path), if given). Returns the number of files."""
        local_dir, n = Path(local_dir), 0
        for k in self._keys(rel):
            if not k or k.endswith(COMPLETE_MARKER) or (keep and not keep(k)):
                continue
            target = local_dir / k
            target.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, self.key(f"{rel}/{k}"), str(target))
            n += 1
        return n
