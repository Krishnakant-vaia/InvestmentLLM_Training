"""Guard against Qwen3.5 silently running on its slow, memory-hungry fallback kernels.

24 of Qwen3.5's 32 layers are linear-attention (Gated DeltaNet). transformers uses the fast Triton
kernels from the `flash-linear-attention` package when it is installed and otherwise quietly falls
back to a pure-PyTorch version that needs tens of GB more memory (one 25k-token row ran an 80 GB
H100 out of memory) and is far slower. The only sign is one warning line in the log, so we check
for it explicitly and stop early instead.
"""
import importlib

# The two flash-linear-attention kernels the linear-attention layers call. (causal-conv1d, the other
# library in transformers' warning, only speeds up a small convolution and needs a CUDA build: not required.)
REQUIRED = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")

FIX = "pip install flash-linear-attention==0.5.2   (it is pinned in requirements/training.txt)"


def missing_fast_kernels(module=None) -> list[str]:
    """Names of fast kernels that transformers' Qwen3.5 code could not import (empty list = all good).

    `module` is injectable for tests; by default it is the real transformers Qwen3.5 module.
    A transformers version without Qwen3.5 has nothing to check, so it reports nothing missing.
    """
    if module is None:
        try:
            module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        except ImportError:
            return []
    return [name for name in REQUIRED if getattr(module, name, None) is None]


def require_fast_kernels(module=None) -> None:
    """Stop with a clear message if the fast kernels are missing."""
    missing = missing_fast_kernels(module)
    if missing:
        raise SystemExit(
            f"Qwen3.5 fast kernels missing ({', '.join(missing)}): its 24 linear-attention layers would run a slow "
            f"PyTorch fallback needing tens of GB more memory.\nFix: {FIX}")
