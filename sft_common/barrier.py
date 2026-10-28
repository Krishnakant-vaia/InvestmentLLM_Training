"""The eval barrier on a training machine: no eval code, eval workbooks or eval outputs may be present.

Run:  python3 -m sft_common.barrier [paths...]     (default: /root)   exit 1 if anything is found.
The Daytona launcher runs this as the first step of every training-type job, so a training run cannot
start on a machine where any trace of the evals exists. Standard library only (runs before any install).
"""
import sys
from pathlib import Path

EVAL_DIRS = {"04_evaluation", "eval_input", "eval_bank"}            # folder names that only the eval side uses
EVAL_NAME_PARTS = ("eval_bank", "VAIA_Eval", "train_overlap")        # file-name pieces of eval code / inputs / outputs
EVAL_SUFFIXES = {".xlsx"}                                            # the eval banks are workbooks; training never reads one
SKIP_DIRS = {"venv", "venv-eval", ".cache", "site-packages", "__pycache__", ".git"}


def find_eval_material(roots) -> list[str]:
    """Every path under `roots` that belongs to the eval side (empty list = clean machine)."""
    found = []
    for root in map(Path, roots):
        if not root.exists():
            continue
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            for p in entries:
                if p.is_dir():
                    if p.name in EVAL_DIRS:
                        found.append(str(p))
                    elif p.name not in SKIP_DIRS and not p.is_symlink():
                        stack.append(p)
                elif p.suffix.lower() in EVAL_SUFFIXES or any(part in p.name for part in EVAL_NAME_PARTS):
                    found.append(str(p))
    return sorted(found)


def main(argv: list[str]) -> int:
    roots = argv or ["/root"]
    found = find_eval_material(roots)
    if found:
        print(f"[barrier] eval material found on the training machine, refusing to train: {found[:10]}", flush=True)
        return 1
    print(f"[barrier] clean: no eval code, eval workbooks or eval outputs under {', '.join(roots)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
