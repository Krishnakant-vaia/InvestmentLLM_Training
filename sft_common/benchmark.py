"""Pure helpers for the micro-batch size benchmark (`sft_train.py --benchmark 1,2,4,8`).

The benchmark answers: how big can a micro-batch be before the GPU runs out of memory, how fast
does each size train, and how long would the whole run take? Size N means a micro-batch budget of
N x training.tokens_per_micro_batch tokens (with a 12,000 budget: 1 -> 12k, 2 -> 24k, 4 -> 48k, 8 -> 96k).
The GPU work lives in sft_train.py; everything here is plain Python and unit-tested.
"""
import random

from .batching import cost


def scaled_budget(base_tokens: int, base_rows: int, n: int) -> tuple[int, int]:
    """Token budget and row cap for size N. The row cap scales too, otherwise a big budget could
    never be filled with short rows."""
    return base_tokens * n, base_rows * n


def suggest_accum(base_accum: int, n: int) -> int:
    """Micro-batches per optimizer step that keep ~the same tokens per step as the base setup
    (never below 1). Bigger micro-batches need fewer of them per step."""
    return max(1, base_accum // n)


def pick_batches(lengths: list[int], batches: list[list[int]], count: int, mode: str, seed: int) -> list[list[int]]:
    """Micro-batches to run, most demanding first (so peak memory is hit during warm-up):
    the one holding the longest row, the one costing the most tokens, then random ones.
    No duplicates; at most `count`."""
    if not batches:
        return []
    order = [max(range(len(batches)), key=lambda k: max(lengths[i] for i in batches[k])),
             max(range(len(batches)), key=lambda k: cost(lengths, batches[k], mode))]
    chosen: list[int] = []
    for k in order:
        if k not in chosen:
            chosen.append(k)
    rest = [k for k in range(len(batches)) if k not in chosen]
    random.Random(seed).shuffle(rest)
    chosen += rest[:max(0, count - len(chosen))]
    return [batches[k] for k in chosen[:count]]


def smoke_schedule(max_steps: int) -> dict:
    """Trainer settings for a short smoke run: stop after `max_steps` optimizer steps, with two
    checkpoints and two evals inside it, so the S3 checkpoint upload and the val loss really run."""
    every = max(1, max_steps // 2)
    return {"max_steps": max_steps, "save_steps": every, "eval_steps": every, "logging_steps": 1}


def eta_hours(total_tokens: int, tokens_per_second: float | None) -> float | None:
    """Hours to train `total_tokens` at the measured speed (None if there is no usable speed)."""
    if not tokens_per_second or tokens_per_second <= 0:
        return None
    return total_tokens / tokens_per_second / 3600


def recommend(results: list[dict], max_pct: float = 90.0) -> dict | None:
    """The fastest size that ran OK and stayed within `max_pct` of GPU memory (headroom for the
    rare batch the benchmark did not sample). None if no size qualifies."""
    ok = [r for r in results if r.get("status") == "ok" and r.get("peak_pct") is not None
          and r["peak_pct"] <= max_pct and r.get("tokens_per_s")]
    return max(ok, key=lambda r: r["tokens_per_s"]) if ok else None


def format_table(results: list[dict]) -> str:
    """Plain-text results table."""
    def cell(v, spec="{}"):
        return "-" if v is None else spec.format(v)
    head = f"{'size':>4} {'budget':>8} {'accum':>5} {'peak GB':>8} {'% GPU':>6} {'tokens/s':>9} {'ETA (h)':>8}  status"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(f"{r['size']:>4} {r['budget']:>8} {r.get('accum', '-'):>5} {cell(r.get('peak_gb'), '{:.1f}'):>8} "
                     f"{cell(r.get('peak_pct'), '{:.0f}%'):>6} {cell(r.get('tokens_per_s'), '{:,.0f}'):>9} "
                     f"{cell(r.get('eta_hours'), '{:.1f}'):>8}  {r.get('status', '')}")
    return "\n".join(lines)
