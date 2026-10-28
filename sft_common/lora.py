"""Where do the LoRA adapters actually attach? A wrong module name is silent: PEFT just skips it.

Qwen3.5 mixes two kinds of layers, and they name their projections differently:
    full attention (8 of 32 layers):     q_proj, k_proj, v_proj, o_proj
    linear attention (24 of 32 layers):  in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
    MLP (all 32 layers):                 gate_proj, up_proj, down_proj
With only the first and third lists as targets, the attention part of 24 layers would stay frozen
without any error. `adapters_per_target` counts what really got an adapter so it can be printed
and checked before training.
"""
from collections import Counter

ADAPTER_SUFFIX = ".lora_A.default"          # PEFT names each adapted layer's first matrix like this


def adapters_per_target(module_names, targets: list[str]) -> dict[str, int]:
    """How many layers received an adapter, per configured target name (0 = never matched)."""
    kinds = Counter(n[: -len(ADAPTER_SUFFIX)].rsplit(".", 1)[-1] for n in module_names if n.endswith(ADAPTER_SUFFIX))
    return {t: kinds.get(t, 0) for t in targets}


def unmatched(counts: dict[str, int]) -> list[str]:
    """Configured targets that attached to no layer at all."""
    return [t for t, n in counts.items() if n == 0]
