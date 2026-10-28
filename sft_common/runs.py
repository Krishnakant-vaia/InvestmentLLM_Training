"""A run = one fine-tuning attempt. Training, merging and evaluation all use the same Run,
so their files land in the same local folders and the same S3 folder.

    new_run(cfg)             training: pick a fresh name (sft-YYYY-MM-DD_HH-MM) and remember it
    existing_run(cfg, name)  resume / merge / eval: use `name`, or the last run started here

The last run's name is kept in outputs/last_run.txt, so `--run` is only needed to pick an
older run or to continue on another machine.
"""
from pathlib import Path

from .config import ROOT, resolve
from .s3_sync import RunStore
from .tracking import make_run_name

LAST_RUN_FILE = ROOT / "outputs" / "last_run.txt"


class Run:
    def __init__(self, cfg: dict, name: str, store: RunStore | None = None):
        self.name = name
        self.train_dir = resolve(cfg["training"]["output_dir"]) / name   # checkpoints + logs while training
        self.logs_dir = self.train_dir / "logs"                          # small text files, synced to S3 logs/
        self.adapter_dir = self.train_dir / "final"                      # LoRA adapter after training
        self.merged_dir = resolve(cfg["merging"]["merged_dir"]) / name   # full model after step 3
        self.eval_dir = resolve(cfg["evaluation"]["out_dir"]) / name     # step 4 results
        self.store = store or RunStore(cfg["storage"]["s3_root"], name)  # this run's S3 folder


DPO_PREFIX = "dpo-"


def is_dpo_run(name: str) -> bool:
    """DPO runs are named dpo-YYYY-MM-DD_HH-MM; that prefix is how every step knows which stage a run belongs to."""
    return name.startswith(DPO_PREFIX)


def dpo_view(cfg: dict) -> dict:
    """The config as the shared run / S3 / tracking / merge code should see it for a DPO run:
    `training` is the dpo section (output_dir, wandb, ...) and storage.s3_root is dpo.s3_root.
    Everything else (model, merging, evaluation, daytona) is shared. Idempotent."""
    return {**cfg, "training": cfg["dpo"], "storage": {**cfg["storage"], "s3_root": cfg["dpo"]["s3_root"]}}


def stage_cfg(cfg: dict, run: str | None) -> dict:
    """`cfg` for the stage `run` belongs to: DPO runs get dpo_view, everything else is unchanged."""
    return dpo_view(cfg) if run and is_dpo_run(run) else cfg


def _remember(name: str) -> None:
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(name, encoding="utf-8")


def new_run(cfg: dict, name: str | None = None) -> Run:
    """Start a new run (fresh timestamped name unless one is given) and remember it."""
    name = name or make_run_name()
    _remember(name)
    return Run(cfg, name)


def existing_run(cfg: dict, name: str | None = None) -> Run:
    """Open a run by name, or the last run started on this machine."""
    if not name:
        if not LAST_RUN_FILE.exists():
            raise SystemExit("No run to continue: pass --run <run_name> (see the S3 folder names).")
        name = LAST_RUN_FILE.read_text(encoding="utf-8").strip()
    _remember(name)
    return Run(cfg, name)
