# Each step keeps its scripts flat in its own folder; put those folders on the
# import path so every test can import its module by name.
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("data_prep", "training", "wandb_tracking"):
    sys.path.insert(0, str(HERE / sub))
