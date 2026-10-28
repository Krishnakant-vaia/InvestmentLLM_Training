"""Throughput, MFU and run-time model for gpt-oss-120b on one Hopper GPU, and the server settings that follow from it.

    python capacity.py                                   # table for H100 / H200 for the real job size
    python capacity.py --calls 2900 --prompt-tokens 9.1e6 --out-per-call 6000 --mbu 0.5

Model numbers are from the model's config.json (36 layers, 128 experts, top-4, 8 KV heads, head_dim 64, alternating
sliding-window 128 / full attention, MXFP4 experts, bf16 attention, router, embeddings and lm_head). GPU numbers are the
NVIDIA SXM spec sheet (HBM bandwidth, dense BF16 FLOPS). Everything else is an ASSUMPTION and a parameter:

  * Decode is memory-bound. One step reads the non-expert weights, every expert any sequence routes to, and each sequence's KV.
    MFU (achieved FLOPs / peak FLOPs) is therefore only a few percent and is NOT the number to optimise; MBU (achieved bytes/s
    over peak bandwidth) is. `mbu` = 0.5 is a typical MoE-decode figure; the pilot measures the real one.
  * Experts are hit uniformly at random (real routing is skewed, which only reduces traffic).
  * KV is bf16; capacity (how many sequences fit) uses the allocation vLLM actually reported; `step_overhead_s` is scheduler/launch time.
  * Prefill is compute-bound at `prefill_mfu`.
"""
import argparse
import shutil
import subprocess

# ---- the model (config.json of openai/gpt-oss-120b) ----------------------------------------------------
LAYERS, HIDDEN, EXPERT_FF, EXPERTS, TOP_K = 36, 2880, 2880, 128, 4
HEADS, KV_HEADS, HEAD_DIM, WINDOW, FULL_LAYERS, VOCAB = 64, 8, 64, 128, 18, 201_088

EXPERT_PARAMS_EACH = 3 * HIDDEN * EXPERT_FF                                  # gate_up (2x) + down
EXPERT_PARAMS = LAYERS * EXPERTS * EXPERT_PARAMS_EACH                        # ~114.7B
ATTN_PARAMS = LAYERS * (HIDDEN * HEADS * HEAD_DIM + 2 * HIDDEN * KV_HEADS * HEAD_DIM + HEADS * HEAD_DIM * HIDDEN)
EMBED_PARAMS = LM_HEAD_PARAMS = VOCAB * HIDDEN
ACTIVE_PARAMS = ATTN_PARAMS + LAYERS * TOP_K * EXPERT_PARAMS_EACH + LM_HEAD_PARAMS    # ~5.1B per generated token
EXPERT_BYTES = EXPERT_PARAMS * 4.25 / 8                                      # MXFP4: 4 bits + one 8-bit scale per 32 values
DENSE_STREAMED_BYTES = (ATTN_PARAMS + LM_HEAD_PARAMS) * 2                    # read every step; embedding lookups are negligible
WEIGHT_BYTES = EXPERT_BYTES + (ATTN_PARAMS + EMBED_PARAMS + LM_HEAD_PARAMS) * 2
FLOPS_PER_TOKEN = 2 * ACTIVE_PARAMS

GPUS = {"H100": {"mem_bytes": 85.0e9, "bw": 3.35e12, "flops": 989e12},          # SXM, HBM3, dense BF16 ("80 GB" = 85.0e9 bytes, as vLLM reports)
        "H200": {"mem_bytes": 150.0e9, "bw": 4.8e12, "flops": 989e12}}          # SXM, HBM3e ("141 GB")

GPU_UTIL, STEP_OVERHEAD_S = 0.92, 0.004
# Measured by vLLM 0.31 on an 80 GB H100 (--gpu-memory-utilization 0.92): weights 61.43 GiB, "Available KV cache memory: 9.34 GiB",
# "GPU KV cache size: 203,612 tokens". That is 49.3 KB of KV per token (vLLM keeps more than the full-attention layers need) and
# ~2.2 GB for CUDA graphs and activations. Capacity uses these; the decode traffic model below still uses the analytic bytes read.
WEIGHT_BYTES_MEASURED = 61.43 * 2**30
KV_ALLOC_BYTES_PER_TOKEN = 9.34 * 2**30 / 203_612
OVERHEAD_BYTES = 2.2e9
MAX_SEQS_CAP = 192                                                           # more concurrent sequences stop paying off


def kv_bytes(ctx: int) -> float:
    """KV cache of one sequence with `ctx` tokens: full-attention layers grow with ctx, sliding layers stop at WINDOW."""
    per_token_layer = 2 * KV_HEADS * HEAD_DIM * 2                           # K and V, bf16
    return per_token_layer * (FULL_LAYERS * ctx + (LAYERS - FULL_LAYERS) * min(ctx, WINDOW))


def kv_budget_bytes(gpu: str) -> float:
    return GPUS[gpu]["mem_bytes"] * GPU_UTIL - WEIGHT_BYTES_MEASURED - OVERHEAD_BYTES


def max_seqs(gpu: str, ctx: int) -> int:
    """Sequences of `ctx` tokens whose KV fits beside the weights (allocation as vLLM measured it)."""
    return max(1, int(kv_budget_bytes(gpu) // (KV_ALLOC_BYTES_PER_TOKEN * ctx)))


def step_seconds(gpu: str, batch: int, ctx: int, mbu: float = 0.5, step_overhead_s: float = STEP_OVERHEAD_S) -> float:
    """Time of one decode step for `batch` sequences, each `ctx` tokens long."""
    g = GPUS[gpu]
    touched = 1 - (1 - TOP_K / EXPERTS) ** batch                              # share of experts at least one sequence routes to
    traffic = touched * EXPERT_BYTES + DENSE_STREAMED_BYTES + batch * kv_bytes(ctx)
    return max(traffic / (g["bw"] * mbu), batch * FLOPS_PER_TOKEN / (g["flops"] * 0.4)) + step_overhead_s


def decode_tps(gpu: str, batch: int, ctx: int, mbu: float = 0.5) -> float:
    return batch / step_seconds(gpu, batch, ctx, mbu)


def mfu(gpu: str, tokens_per_s: float) -> float:
    return tokens_per_s * FLOPS_PER_TOKEN / GPUS[gpu]["flops"]


def implied_mbu(gpu: str, batch: int, ctx: int, tokens_per_s: float, step_overhead_s: float = STEP_OVERHEAD_S) -> float:
    """The bandwidth utilisation that would produce a measured throughput (the inverse of decode_tps)."""
    touched = 1 - (1 - TOP_K / EXPERTS) ** batch
    traffic = touched * EXPERT_BYTES + DENSE_STREAMED_BYTES + batch * kv_bytes(ctx)
    return traffic / (GPUS[gpu]["bw"] * max(batch / tokens_per_s - step_overhead_s, 1e-6))


def serve_seqs(gpu: str, ctx: int = 7000) -> int:
    """`--max-num-seqs` for the generation server."""
    return max(8, min(MAX_SEQS_CAP, max_seqs(gpu, ctx)))


def estimate(gpu: str, calls: int, prompt_tokens: float, out_per_call: float, judge_calls: int, judge_prompt_tokens: float,
             judge_out_per_call: float = 2500, gen_ctx: int = 7000, judge_ctx: int = 24000, mbu: float = 0.5,
             prefill_mfu: float = 0.35, load_minutes: float = 25) -> dict:
    """Hours per phase for one GPU: model download + load, generation, judging."""
    gen_batch = serve_seqs(gpu, gen_ctx)
    judge_batch = min(gen_batch, max_seqs(gpu, judge_ctx))
    prefill = lambda tokens: tokens * FLOPS_PER_TOKEN / (GPUS[gpu]["flops"] * prefill_mfu)  # noqa: E731
    gen_tps, judge_tps = decode_tps(gpu, gen_batch, gen_ctx, mbu), decode_tps(gpu, judge_batch, judge_ctx, mbu)
    gen_s = prefill(prompt_tokens) + calls * out_per_call / gen_tps
    judge_s = prefill(judge_prompt_tokens) + judge_calls * judge_out_per_call / judge_tps
    return {"gpu": gpu, "gen_concurrent": gen_batch, "judge_concurrent": judge_batch, "gen_tok_s": gen_tps,
            "gen_mfu": mfu(gpu, gen_tps), "load_h": load_minutes / 60, "gen_h": gen_s / 3600, "judge_h": judge_s / 3600,
            "total_h": load_minutes / 60 + (gen_s + judge_s) / 3600}


def detect_gpu() -> tuple[str, float] | None:
    """(GPUS key, memory in GB) of GPU 0 from nvidia-smi, or None when there is no GPU (a local test)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
        name, mem = [x.strip() for x in out.split(",")]
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None
    key = next((k for k in GPUS if k in name.upper()), None)
    return (key, float(mem) / 1000) if key else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=int, default=2900)
    ap.add_argument("--prompt-tokens", type=float, default=9.1e6, help="total generation prompt tokens")
    ap.add_argument("--out-per-call", type=float, default=6000, help="output tokens per call, reasoning included (ASSUMED)")
    ap.add_argument("--judge-calls", type=int, default=650)
    ap.add_argument("--judge-prompt-tokens", type=float, default=14e6)
    ap.add_argument("--mbu", type=float, default=0.5)
    a = ap.parse_args()
    print(f"weights {WEIGHT_BYTES / 1e9:.1f} GB (experts {EXPERT_BYTES / 1e9:.1f} GB MXFP4) | active params {ACTIVE_PARAMS / 1e9:.2f}B "
          f"| {FLOPS_PER_TOKEN / 1e9:.1f} GFLOP per token | KV {kv_bytes(7000) / 1e6:.0f} MB per 7k-token sequence")
    for gpu in GPUS:
        print(f"{gpu}: KV budget {kv_budget_bytes(gpu) / 1e9:.1f} GB -> {max_seqs(gpu, 7000)} concurrent 7k-token sequences "
              f"(server runs {serve_seqs(gpu)}), {max_seqs(gpu, 24000)} judge sequences of 24k")
    print(f"\n{'GPU':5} {'MBU':>5} {'concurrent':>10} {'tok/s':>7} {'MFU':>6} {'gen h':>7} {'judge h':>8} {'total h':>8}   (out {a.out_per_call:.0f} tokens/call)")
    for gpu in GPUS:
        for mbu in (0.35, a.mbu, 0.65):
            e = estimate(gpu, a.calls, a.prompt_tokens, a.out_per_call, a.judge_calls, a.judge_prompt_tokens, mbu=mbu)
            print(f"{gpu:5} {mbu:>5.2f} {e['gen_concurrent']:>10} {e['gen_tok_s']:>7.0f} {e['gen_mfu']:>6.1%} {e['gen_h']:>7.2f} "
                  f"{e['judge_h']:>8.2f} {e['total_h']:>8.2f}")
    print("\nsensitivity to output length (H200, MBU %.2f):" % a.mbu)
    for out in (4000, 6000, 8000):
        e = estimate("H200", a.calls, a.prompt_tokens, out, a.judge_calls, a.judge_prompt_tokens, mbu=a.mbu)
        print(f"  {out} tokens/call -> total {e['total_h']:.2f} h")


if __name__ == "__main__":
    main()
