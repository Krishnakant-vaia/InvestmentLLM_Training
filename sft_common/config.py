"""Config loader shared by every step."""
import os
from pathlib import Path

import yaml
from dotenv import dotenv_values

# The step folder that owns config.yaml, .env and outputs/. Set by that step's step_root.py
# locally; on a Daytona sandbox the package sits inside the step's code dir, so the parent is right.
ROOT = Path(os.environ.get("SFT_STEP_ROOT") or Path(__file__).resolve().parent.parent)


def load_env(path: str | Path | None = None) -> None:
    """Load .env into os.environ. Non-empty .env values override stale shell/machine vars;
    blank lines are ignored so an unfilled template never wipes existing credentials."""
    for k, v in dotenv_values(path or ROOT / ".env").items():
        if v and v.strip():
            os.environ[k] = v.strip()


def load_config(path: str | Path | None = None) -> dict:
    """Load .env, then config.yaml (repo root by default)."""
    load_env()
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
