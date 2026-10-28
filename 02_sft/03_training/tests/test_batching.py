"""Batching: both modes use every row exactly once, whole, within the budget, reproducibly."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_common.batching import (PackedSampler, PaddedSampler, batch_stats, cost, heaviest_batches,
                                 length_report, make_sampler, optimizer_steps)

BUDGET, MAX_ROWS = 32000, 64


def realistic_lengths(n=3000, seed=0):
    """Mostly ~1.7k domain rows, ~550 tool rows, and a few long ones up to 24.6k."""
    rng = random.Random(seed)
    out = [max(50, int(rng.gauss(1700, 500))) for _ in range(n)]
    out += [max(50, int(rng.gauss(550, 200))) for _ in range(n // 3)]
    out += [24604, 23019, 12400, 12000, 9000, 8500]
    return out


@pytest.mark.parametrize("mode", ["padded", "packed"])
def test_every_row_once_within_budget(mode):
    lengths = realistic_lengths()
    batches = make_sampler(mode, lengths, BUDGET, MAX_ROWS, seed=1).batches()
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))   # each row exactly once
    for b in batches:
        assert len(b) <= MAX_ROWS
        assert cost(lengths, b, mode) <= BUDGET or len(b) == 1                 # an oversize row goes alone


@pytest.mark.parametrize("mode", ["padded", "packed"])
def test_deterministic_and_reshuffled_per_epoch(mode):
    lengths = realistic_lengths()
    a = make_sampler(mode, lengths, BUDGET, MAX_ROWS, seed=7)
    b = make_sampler(mode, lengths, BUDGET, MAX_ROWS, seed=7)
    assert a.batches() == b.batches() and len(a) == len(a.batches())             # resume replays the same order
    b.set_epoch(1)
    assert b.batches() != a.batches()                                              # a new epoch reshuffles


def test_packed_fills_the_space_next_to_long_rows():
    """The user's scheme: the longest row first, short rows fill the space left; nothing alone."""
    lengths = [24600] + [1700] * 8
    batches = PackedSampler(lengths, max_tokens=BUDGET, max_rows=MAX_ROWS, seed=0).batches()
    first = next(b for b in batches if 0 in b)
    assert len(first) == 5 and cost(lengths, first, "packed") == 24600 + 4 * 1700   # 31.4k
    assert all(len(b) > 1 for b in batches)


def test_padded_long_row_alone():
    lengths = [24600] + [1700] * 8
    batches = PaddedSampler(lengths, max_tokens=BUDGET, max_rows=MAX_ROWS, seed=0).batches()
    assert [0] in batches                                                            # 2 x 24.6k > 32k


def test_packed_wastes_less_than_padded_on_realistic_data():
    lengths = realistic_lengths()
    p = batch_stats(lengths, make_sampler("padded", lengths, BUDGET, MAX_ROWS, 1).batches(), "padded", BUDGET)
    k = batch_stats(lengths, make_sampler("packed", lengths, BUDGET, MAX_ROWS, 1).batches(), "packed", BUDGET)
    assert k["padding_pct"] == 0 and k["micro_batches"] < p["micro_batches"]
    assert k["single_row_micro_batches"] <= p["single_row_micro_batches"]


def test_long_rows_spread_across_training():
    lengths = [1700] * 3000 + [20000] * 20
    batches = make_sampler("padded", lengths, BUDGET, MAX_ROWS, seed=3).batches()
    pos = [k for k, b in enumerate(batches) if any(lengths[i] > 8192 for i in b)]
    assert pos[0] < len(batches) // 2 < pos[-1]                                     # not grouped at the end


def test_eval_order_is_fixed_without_shuffle():
    lengths = realistic_lengths(300)
    a = make_sampler("packed", lengths, BUDGET, MAX_ROWS, seed=1, shuffle=False).batches()
    assert a == make_sampler("packed", lengths, BUDGET, MAX_ROWS, seed=99, shuffle=False).batches()


def test_stats_heaviest_steps_and_report():
    lengths = [100, 100, 9000, 40000]
    batches = [[0, 1], [2], [3]]
    s = batch_stats(lengths, batches, "padded", BUDGET)
    assert s["micro_batches"] == 3 and s["max_tokens_in_a_micro_batch"] == 40000 and s["single_row_micro_batches"] == 2
    h = heaviest_batches(lengths, batches, "packed")
    assert list(h.values()) == [[3], [0, 1]]
    assert optimizer_steps(10, 4, 1) == 3
    rep = length_report(["d", "d", "t"], [100, 40000, 50], limit=32000)
    assert rep["d"] == {"rows": 2, "dropped": 1, "max_tokens": 40000} and rep["t"]["dropped"] == 0


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        make_sampler("glued", [1, 2], BUDGET, MAX_ROWS, 0)


# ---- best fit, coverage proof, batch plan ---------------------------------------------

def _shortest_first(lengths, budget, max_rows):
    """The previous packed method (longest row + shortest rows), kept here only as a reference."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, lo, hi = [], 0, len(order) - 1
    while lo <= hi:
        cur, used = [order[lo]], lengths[order[lo]]
        lo += 1
        while lo <= hi and len(cur) < max_rows and used + lengths[order[hi]] <= budget:
            cur.append(order[hi])
            used += lengths[order[hi]]
            hi -= 1
        out.append(cur)
    return out


def test_best_fit_takes_the_largest_row_that_fits():
    lengths = [24600, 7200, 180, 400, 3000, 3000]
    batches = PackedSampler(lengths, max_tokens=BUDGET, max_rows=MAX_ROWS, seed=0).batches()
    first = next(b for b in batches if 0 in b)
    assert sorted(lengths[i] for i in first) == [180, 7200, 24600]   # gap 7.4k -> 7.2k -> 180 (not 400s/3000s)
    assert cost(lengths, first, "packed") == 31980


def test_best_fit_fills_at_least_as_well_as_shortest_first():
    lengths = realistic_lengths(6000, seed=5)
    new = PackedSampler(lengths, max_tokens=BUDGET, max_rows=MAX_ROWS, seed=0).batches()
    old = _shortest_first(lengths, BUDGET, MAX_ROWS)
    assert len(new) <= len(old)
    assert batch_stats(lengths, new, "packed", BUDGET)["avg_fill_pct"] >= batch_stats(lengths, old, "packed", BUDGET)["avg_fill_pct"]


def test_verify_coverage_catches_lost_and_repeated_rows():
    from sft_common.batching import verify_coverage
    verify_coverage(4, [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="lost 1"):
        verify_coverage(4, [[0, 1], [2]])
    with pytest.raises(ValueError, match="repeated 1"):
        verify_coverage(3, [[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        verify_coverage(2, [[0, 5]])


def test_batch_plan_names_the_train_jsonl_lines():
    from sft_common.batching import batch_plan
    plan = list(batch_plan([[1, 0]], [100, 200], row_ids=[10, 11], sources=["domain", "tool_calling"]))
    assert plan == [{"micro_batch": 0, "rows": [11, 10], "sources": ["tool_calling", "domain"],
                     "tokens": [200, 100], "total": 300}]


# ---- micro-batch size benchmark helpers (pure) ----------------------------------------

def test_benchmark_scaling_and_accum():
    from sft_common.benchmark import scaled_budget, suggest_accum
    assert scaled_budget(32000, 64, 4) == (128000, 256)
    assert [suggest_accum(4, n) for n in (1, 2, 4, 8)] == [4, 2, 1, 1]          # never below 1


def test_pick_batches_puts_the_hardest_first_without_duplicates():
    from sft_common.benchmark import pick_batches
    lengths = [24600, 1700, 1700, 400, 400, 400, 9000, 800]
    batches = [[1, 2], [0, 3], [6, 4], [5, 7]]
    picks = pick_batches(lengths, batches, count=3, mode="packed", seed=1)
    assert picks[0] == [0, 3]                                                    # holds the longest row
    assert picks[1] == [6, 4] or picks[1] == [1, 2]                              # then the costliest / random
    assert len(picks) == 3 and len({tuple(p) for p in picks}) == 3
    assert pick_batches(lengths, [], 3, "packed", 1) == []
    assert len(pick_batches(lengths, batches, 99, "packed", 1)) == 4             # capped at what exists


def test_eta_hours_and_recommendation():
    from sft_common.benchmark import eta_hours, recommend
    assert eta_hours(360_000_000, 10_000) == 10.0
    assert eta_hours(1, None) is None and eta_hours(1, 0) is None
    results = [
        {"size": 1, "status": "ok", "peak_pct": 40, "tokens_per_s": 9000},
        {"size": 2, "status": "ok", "peak_pct": 65, "tokens_per_s": 10500},
        {"size": 4, "status": "ok", "peak_pct": 95, "tokens_per_s": 11000},   # fastest but too close to the limit
        {"size": 8, "status": "OOM", "peak_pct": None, "tokens_per_s": None},
    ]
    assert recommend(results)["size"] == 2
    assert recommend(results, max_pct=99)["size"] == 4
    assert recommend([results[3]]) is None


def test_format_table_shows_failures_and_skips():
    from sft_common.benchmark import format_table
    table = format_table([
        {"size": 1, "budget": 32000, "accum": 4, "peak_gb": 41.2, "peak_pct": 29.2, "tokens_per_s": 9800.0,
         "eta_hours": 9.5, "status": "ok"},
        {"size": 8, "budget": 256000, "accum": 1, "status": "OOM"},
    ])
    assert "41.2" in table and "9,800" in table and "OOM" in table and table.count("\n") == 3


# ---- the configured length limit and micro-batch budget (12,000 / 12,000) ----------------

def test_config_budget_holds_the_longest_allowed_row_and_eval_context_is_separate():
    from sft_common.config import load_config
    c = load_config()
    limit, budget = c["model"]["max_seq_length"], c["training"]["tokens_per_micro_batch"]
    assert budget >= limit                                    # the longest allowed row must fit a micro-batch alone
    e = c["evaluation"]
    # eval must hold prompt + answer for val rows the training limit does not filter (longest val row: 16,437)
    assert e["max_model_len"] >= 16437 + max(e["max_new_tokens"].values())


@pytest.mark.parametrize("mode", ["padded", "packed"])
def test_every_row_up_to_the_limit_fits_the_budget_including_an_exact_limit_row(mode):
    limit = budget = 12000
    rng = random.Random(3)
    lengths = [rng.randint(300, 4300) for _ in range(2000)] + [limit, limit, limit - 1, 11000, 9000]
    batches = make_sampler(mode, lengths, budget, 64, seed=2).batches()
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(cost(lengths, b, mode) <= budget for b in batches)     # the 12,000-token rows go alone, never over


# ---- LoRA adapter placement + smoke schedule ------------------------------------------------

def test_adapters_per_target_counts_and_flags_unmatched_names():
    from sft_common.lora import adapters_per_target, unmatched
    names = [f"model.layers.{i}.mlp.gate_proj.lora_A.default" for i in range(3)] +             [f"model.layers.{i}.linear_attn.in_proj_qkv.lora_A.default" for i in range(2)] +             ["model.layers.0.mlp.gate_proj.lora_B.default", "model.layers.0.self_attn.q_proj.base_layer"]   # not counted
    counts = adapters_per_target(names, ["gate_proj", "in_proj_qkv", "q_proj", "out_proj"])
    assert counts == {"gate_proj": 3, "in_proj_qkv": 2, "q_proj": 0, "out_proj": 0}
    assert unmatched(counts) == ["q_proj", "out_proj"]


def test_config_targets_cover_full_attention_linear_attention_and_mlp():
    from sft_common.config import load_config
    targets = set(load_config()["training"]["lora"]["target_modules"])
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets              # 8 full-attention layers
    assert {"in_proj_qkv", "in_proj_z", "out_proj"} <= targets              # 24 linear-attention layers
    assert {"gate_proj", "up_proj", "down_proj"} <= targets                 # MLP in all 32 layers


def test_smoke_schedule_saves_and_evaluates_twice():
    from sft_common.benchmark import smoke_schedule
    assert smoke_schedule(20) == {"max_steps": 20, "save_steps": 10, "eval_steps": 10, "logging_steps": 1}
    assert smoke_schedule(1)["save_steps"] == 1
