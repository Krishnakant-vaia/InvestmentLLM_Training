"""Make a LoRA adapter portable across machines."""
import json
from pathlib import Path


def point_adapter_at_base(adapter_dir: str | Path, base_dir: str | Path) -> None:
    """Rewrite adapter_config.json so the adapter's base model is `base_dir` on THIS machine.

    Training records the local base path (e.g. models/pretuned), which is wrong on another
    machine. Idempotent; the first original value is kept under `original_base_model_name_or_path`.
    """
    f = Path(adapter_dir) / "adapter_config.json"
    cfg = json.loads(f.read_text(encoding="utf-8"))
    new = str(Path(base_dir).resolve())
    if cfg.get("base_model_name_or_path") == new:
        return
    cfg.setdefault("original_base_model_name_or_path", cfg.get("base_model_name_or_path"))
    cfg["base_model_name_or_path"] = new
    f.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
