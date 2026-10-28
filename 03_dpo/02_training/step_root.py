"""Imported first by every script in this step: makes the shared sft_common package (at the repo
root) importable and tells it this folder is the root for config.yaml, .env and outputs/."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SFT_STEP_ROOT", str(HERE))
if not (HERE / "sft_common").is_dir():          # locally; on a Daytona sandbox it is unpacked here
    sys.path.insert(1, str(HERE.parent.parent))
