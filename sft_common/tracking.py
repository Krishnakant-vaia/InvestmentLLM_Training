"""Wandb run tracking, isolated so the rest of training never hard-depends on wandb
being installed or configured. If it's off or unconfigured, every function here is a
harmless no-op and training proceeds with console/file logging only.
"""
from datetime import datetime


def resolve_report_to(cfg: dict) -> str:
    """"wandb" if training.wandb.enabled and WANDB_API_KEY is set (via .env), else "none"."""
    import os
    wb = cfg["training"].get("wandb") or {}
    return "wandb" if wb.get("enabled") and os.environ.get("WANDB_API_KEY") else "none"


def make_run_name(prefix: str = "sft", now: datetime | None = None) -> str:
    """Timestamped, unique-enough run name (e.g. sft-2026-09-27_14-30)."""
    return f"{prefix}-{(now or datetime.now()):%Y-%m-%d_%H-%M}"


def init_wandb(cfg: dict, run_name: str) -> bool:
    """Start (or continue) the wandb run for `run_name`, logging the full training config.
    Returns whether it started.

    Called before trainer.train(): transformers' own wandb integration then finds wandb.run
    already active and reuses it instead of re-initializing with less context.
    The run name is also the wandb run id, so `--resume` continues the same wandb charts.
    """
    if resolve_report_to(cfg) != "wandb":
        return False
    import wandb
    wandb.init(project=cfg["training"]["wandb"]["project"], name=run_name, id=run_name, resume="allow",
               config=cfg["training"])
    return True


def log(step: int, metrics: dict) -> None:
    """Log metrics to the active wandb run, if any. Safe to call whether or not wandb ran."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
