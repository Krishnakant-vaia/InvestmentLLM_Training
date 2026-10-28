"""Training callbacks: fit monitoring and S3 checkpoint sync."""
import json
from pathlib import Path

import numpy as np
from transformers import TrainerCallback

from . import tracking


class FitAnalysisCallback(TrainerCallback):
    """At each eval, compare train/val loss trends and print an over/underfit verdict.

    History is persisted to `log_file` so it survives resumed runs.
    """

    def __init__(self, log_file: str, window: int = 3):
        self.log_file, self.window = Path(log_file), window
        self.train, self.val = [], []
        if self.log_file.exists():
            try:
                h = json.loads(self.log_file.read_text())
                self.train, self.val = h["train"], h["val"]
            except (json.JSONDecodeError, KeyError):
                pass

    def _slope(self, xs):
        return float(np.polyfit(range(len(xs)), xs, 1)[0]) if len(xs) > 1 else 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        val = (metrics or {}).get("eval_loss")
        train = next((l["loss"] for l in reversed(state.log_history) if "loss" in l), None)
        if val is None or train is None:
            return
        self.train.append(train)
        self.val.append(val)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text(json.dumps({"train": self.train, "val": self.val}))
        if len(self.val) < self.window:
            return

        t_slope, v_slope = self._slope(self.train[-self.window:]), self._slope(self.val[-self.window:])
        gap, prev_gap = val - train, self.val[-2] - self.train[-2]
        over = (2 if v_slope > 1e-5 and t_slope < -1e-5 else 0) \
            + (1 if gap > max(prev_gap * 1.4, 0.05) else 0) + (1 if gap > val * 0.2 else 0)
        stalled = abs(v_slope) < val * 0.01 and v_slope > -1e-5

        verdict = ("OVERFITTING (high)" if over >= 2 else "overfitting (moderate)" if over == 1
                   else "STALLED (val plateau)" if stalled else "healthy")
        print(f"[fit] step {state.global_step}: train={train:.3f} val={val:.3f} gap={gap:.3f} -> {verdict}")
        tracking.log(state.global_step, {"fit/train_loss": train, "fit/val_loss": val, "fit/gap": gap,
                                         "fit/overfit_signal": over, "fit/verdict": verdict})


class S3SyncCallback(TrainerCallback):
    """Uploads each checkpoint to S3 as soon as the Trainer saves it, plus the training logs.

    Locally the Trainer keeps only the newest `save_total_limit` checkpoints; S3 keeps all of
    them. Uploads are synchronous (~1 GB every `save_steps`), so a checkpoint is never
    deleted locally before it has reached S3. A failed upload is retried by `sync_all`.
    """

    def __init__(self, run):
        self.run = run

    def _sync_logs(self, state) -> None:
        """Write the Trainer's log history (loss, lr, grad norm per step) and upload the logs folder."""
        self.run.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.run.logs_dir / "log_history.json").write_text(json.dumps(state.log_history, indent=1))
        self.run.store.upload_dir(self.run.logs_dir, "logs")

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero:           # only one process uploads
            return
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ok = self.run.store.upload_checkpoint(ckpt)
        print(f"[s3] {ckpt.name} -> {self.run.store.uri('checkpoints/' + ckpt.name)}: "
              f"{'uploaded' if ok else 'FAILED, will retry at the end'}")
        self._sync_logs(state)

    def sync_all(self, state) -> None:
        """Final pass: re-upload any local checkpoint S3 does not have complete, then the logs."""
        done = set(self.run.store.checkpoint_steps())
        for ckpt in sorted(self.run.train_dir.glob("checkpoint-*")):
            if ckpt.name[11:].isdigit() and int(ckpt.name[11:]) not in done:
                self.run.store.upload_checkpoint(ckpt)
        self._sync_logs(state)
