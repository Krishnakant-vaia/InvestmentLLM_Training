"""Memory probe: how much GPU memory does one training step need, as a function of sequence length?

Run:  python 02_training/memory_probe.py [--lengths 2000,4000,8000,12000]
No dataset and no S3: it feeds random token ids to the real model (same LoRA setup as training), so it
needs only the model. For each way of computing the loss it runs a real forward + backward at
increasing lengths and prints peak memory, MB per token and speed:

  as trained   the loss the Trainer uses today (scores every token over the whole vocabulary)
  lean loss    scores only the answer tokens, in chunks (sft_common/lean_loss.py); same maths

It also prints facts that explain the numbers: which attention kernel is used, whether gradient
checkpointing is really on, and whether the linear-attention layers run the fast or the slow kernel.
The last quarter of each sequence counts as the answer, like a typical domain row.
Results go to the console and outputs/probe.json.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation near the limit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_train import load_model  # noqa: E402  (imports Unsloth first, as it must be)

import torch  # noqa: E402

from sft_common.config import ROOT, load_config  # noqa: E402
from sft_common.kernels import require_fast_kernels  # noqa: E402
from sft_common.lean_loss import IGNORE, lean_loss  # noqa: E402

ANSWER_SHARE = 0.25          # last quarter of each probe sequence is the "answer" (gets a label)


def describe(model) -> dict:
    """Facts about the loaded model that explain its memory use."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = list(inner.model.layers)
    delta = next((l.linear_attn for l in layers if hasattr(l, "linear_attn")), None)
    kernel = getattr(getattr(delta, "chunk_gated_delta_rule", None), "__module__", "?") if delta is not None else "no linear-attn layer"
    facts = {
        "attention_implementation": str(getattr(inner.config, "_attn_implementation", "?")),
        "layers": len(layers),
        "layers_with_gradient_checkpointing": sum(bool(getattr(l, "gradient_checkpointing", False)) for l in layers),
        "linear_attention_kernel_module": kernel,          # 'fla...' = fast, 'transformers...' = slow torch fallback
        "trainable_params_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6),
        "weights_GB": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9, 1),
        "training_mode": bool(model.training),
    }
    return facts


def make_batch(length: int, vocab: int, device):
    ids = torch.randint(0, vocab, (1, length), device=device)
    labels = ids.clone()
    labels[:, : int(length * (1 - ANSWER_SHARE))] = IGNORE          # context is not a target
    return ids, labels


def loss_as_trained(model, ids, labels):
    return model(input_ids=ids, labels=labels).loss


def loss_lean(model, ids, labels):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(input_ids=ids).last_hidden_state
    return lean_loss(hidden, inner.lm_head.weight, labels)[0]


VARIANTS = {"as trained": loss_as_trained, "lean loss": loss_lean}


def loss_agreement(model, vocab: int) -> dict:
    """Do the two losses agree on the same input? They must: the lean one is only a cheaper way to compute it."""
    torch.manual_seed(0)
    ids, labels = make_batch(2000, vocab, model.device)
    with torch.no_grad():
        a, b = float(loss_as_trained(model, ids, labels)), float(loss_lean(model, ids, labels))
    return {"as_trained": round(a, 4), "lean": round(b, 4), "relative_difference": round(abs(a - b) / max(abs(a), 1e-9), 5)}


def run_one(model, fn, length: int, vocab: int) -> dict:
    """One forward + backward at `length` tokens: peak memory above the resting weights, and speed."""
    gc.collect()
    torch.cuda.empty_cache()
    ids, labels = make_batch(length, vocab, model.device)
    torch.cuda.synchronize()
    resting = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fn(model, ids, labels).backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    model.zero_grad(set_to_none=True)
    return {"length": length, "status": "ok", "peak_GB": round(peak / 1e9, 1),
            "step_GB": round((peak - resting) / 1e9, 1), "MB_per_token": round((peak - resting) / 1e6 / length, 2),
            "seconds": round(seconds, 2), "tokens_per_s": round(length / seconds)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="2000,4000,8000,12000", help="sequence lengths to try, in order")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]

    require_fast_kernels()
    cfg = load_config()
    model, _ = load_model(cfg)
    model.train()
    facts = describe(model)
    gpu = torch.cuda.get_device_properties(0)
    print(f"[probe] {gpu.name}, {gpu.total_memory / 1e9:.0f} GB")
    print("[probe] model facts:", json.dumps(facts, indent=1))
    vocab = model.get_base_model().config.vocab_size if hasattr(model, "get_base_model") else model.config.vocab_size

    results = {"gpu": gpu.name, "facts": facts, "runs": {}, "loss_agreement": loss_agreement(model, vocab)}
    print("[probe] loss agreement:", results["loss_agreement"], flush=True)
    for name, fn in VARIANTS.items():
        rows = results["runs"][name] = []
        try:
            run_one(model, fn, 500, vocab)                     # warm-up: kernel compilation, allocator
        except Exception as e:  # noqa: BLE001
            rows.append({"length": 500, "status": f"error: {type(e).__name__}: {str(e)[:300]}"})
            print(f"[probe] {name}: warm-up failed: {rows[-1]['status']}", flush=True)
            continue
        for length in lengths:
            try:
                row = run_one(model, fn, length, vocab)
            except Exception as e:  # noqa: BLE001 - out of memory is a result here
                oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                row = {"length": length, "status": "OOM" if oom else f"error: {type(e).__name__}: {str(e)[:300]}"}
                model.zero_grad(set_to_none=True)
            rows.append(row)
            print(f"[probe] {name:11} {row}", flush=True)
            if row["status"] != "ok":
                break                                          # longer ones would fail too
    out = ROOT / "outputs" / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[probe] saved {out}")


if __name__ == "__main__":
    main()
