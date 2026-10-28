"""Rebuild a run's wandb charts from its S3 checkpoint (e.g. after switching wandb accounts).

Run:  python 02_training/wandb_backfill.py --run NAME [--checkpoint N]   # default: newest S3 checkpoint

A checkpoint's trainer_state.json holds the Trainer's whole log history up to that step (train loss,
grad norm, learning rate every logging_steps; val loss every eval_steps), across every resume. This
replays it into wandb (the account of WANDB_API_KEY in .env, project training.wandb.project) under
run id = run name, so a later `sft_train.py --resume` from the SAME checkpoint continues these charts.
Backfill up to exactly the checkpoint training resumes from, or the charts overlap / leave a gap.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.config import load_config  # noqa: E402
from sft_common.runs import Run  # noqa: E402

# trainer_state keys -> the names transformers' wandb integration uses, so the charts line up.
TRAIN_KEYS = {"loss": "train/loss", "grad_norm": "train/grad_norm", "learning_rate": "train/learning_rate"}
EVAL_KEYS = {"eval_loss": "eval/loss", "eval_runtime": "eval/runtime",
             "eval_samples_per_second": "eval/samples_per_second", "eval_steps_per_second": "eval/steps_per_second"}


def to_wandb(entry: dict) -> dict:
    """One trainer_state log entry as a wandb row (train or eval metrics, plus step and epoch)."""
    row = {"train/global_step": entry["step"], "train/epoch": entry.get("epoch")}
    for src, dst in {**TRAIN_KEYS, **EVAL_KEYS}.items():
        if src in entry:
            row[dst] = float(entry[src])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", type=int, help="checkpoint step to replay up to (default: newest on S3)")
    a = ap.parse_args()

    cfg = load_config()
    run = Run(cfg, a.run)
    step = a.checkpoint or run.store.checkpoint_steps()[-1]
    with tempfile.TemporaryDirectory() as tmp:
        n = run.store.download_dir(f"checkpoints/checkpoint-{step}", tmp, keep=lambda f: f == "trainer_state.json")
        if not n:
            raise SystemExit(f"no trainer_state.json in {run.store.uri(f'checkpoints/checkpoint-{step}')}")
        history = json.loads((Path(tmp) / "trainer_state.json").read_text(encoding="utf-8"))["log_history"]

    import wandb
    wandb.init(project=cfg["training"]["wandb"]["project"], name=run.name, id=run.name, resume="allow",
               config={**cfg["training"], "backfilled_from": run.store.uri(f"checkpoints/checkpoint-{step}")})
    wandb.define_metric("train/global_step")
    wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
    rows = [to_wandb(e) for e in history if "step" in e and (e.keys() & {**TRAIN_KEYS, **EVAL_KEYS}.keys())]
    for r in rows:
        wandb.log(r, step=r["train/global_step"])
    evals = [r for r in rows if "eval/loss" in r]
    print(f"[backfill] {len(rows)} rows up to step {step} ({len(evals)} val-loss points) -> {wandb.run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
