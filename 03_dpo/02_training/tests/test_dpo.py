"""DPO: the chunked log-prob and loss maths, and the pair data (real Qwen3.5 tokenizer + chat template)."""
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_common.chat_format import ASSISTANT_PART, ChatFormat
from sft_common.dpo_data import IGNORE, collate_pairs, encode_pair, prepare, read_pairs
from sft_common.dpo_loss import completion_logps, dpo_loss

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "models" / "pretuned"
OPTS = {"max_pair_tokens": 400, "max_length_ratio": 2.0, "drop_think_mismatch": True,
        "val_fraction": 0.2, "val_cap_per_source": 50}


@pytest.fixture(scope="module")
def tok():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("pre-tuned tokenizer files not downloaded")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


# ---- maths ----------------------------------------------------------------------------------

def naive_logps(hidden, weight, labels):
    """Reference implementation: full vocabulary scores for every position."""
    logp = torch.log_softmax((hidden[:, :-1] @ weight.t()).float(), -1)
    tgt = labels[:, 1:]
    got = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (got * (tgt != IGNORE)).sum(1), (tgt != IGNORE).sum(1)


def test_chunked_logps_equal_the_naive_ones_and_their_gradients():
    torch.manual_seed(0)
    hidden = torch.randn(4, 17, 8, requires_grad=True)
    weight = torch.randn(50, 8, requires_grad=True)
    labels = torch.randint(0, 50, (4, 17))
    labels[:, :6] = IGNORE                      # prompt
    labels[2, 12:] = IGNORE                     # padding
    labels[3] = IGNORE                          # a row with no answer at all
    want, want_n = naive_logps(hidden, weight, labels)
    got, got_n = completion_logps(hidden, weight, labels, chunk=5)   # chunk smaller than the token count
    assert torch.allclose(got, want, atol=1e-4) and torch.equal(got_n, want_n)
    g1 = torch.autograd.grad(got.sum(), [hidden, weight])
    g2 = torch.autograd.grad(want.sum(), [hidden, weight])
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-4)


def test_no_answer_tokens_at_all_still_gives_a_graph():
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    sums, n = completion_logps(hidden, torch.randn(10, 4), torch.full((2, 5), IGNORE))
    assert n.sum() == 0 and sums.requires_grad and sums.abs().sum() == 0


def test_dpo_loss_matches_the_formula():
    pc, pr, rc, rr = (torch.tensor([-10.0, -20.0]), torch.tensor([-12.0, -15.0]),
                      torch.tensor([-11.0, -21.0]), torch.tensor([-11.5, -17.0]))
    beta = 0.1
    z = [beta * ((-10 + 12) - (-11 + 11.5)), beta * ((-20 + 15) - (-21 + 17))]
    want = sum(-math.log(1 / (1 + math.exp(-x))) for x in z) / 2
    loss, m = dpo_loss(pc, pr, rc, rr, beta)
    assert loss.item() == pytest.approx(want, rel=1e-5)
    assert m["rewards/accuracies"] == 0.5               # chosen reward 0.1 beats rejected -0.05 in pair 1; loses to +0.2 in pair 2
    assert m["ref/prefers_chosen"] == 0.5               # reference: -11 > -11.5 yes, -21 > -17 no


def test_identical_policy_and_reference_is_loss_log2_with_zero_margin():
    x = torch.tensor([-5.0, -9.0])
    loss, m = dpo_loss(x, x - 1, x, x - 1, beta=0.3)
    assert loss.item() == pytest.approx(math.log(2)) and m["rewards/margins"] == 0.0


def test_label_smoothing_blends_the_two_directions():
    pc, pr, rc, rr = torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-2.0])
    z = 0.5 * 2.0
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.5, label_smoothing=0.25)
    assert loss.item() == pytest.approx(-(0.75 * F.logsigmoid(torch.tensor(z)) + 0.25 * F.logsigmoid(torch.tensor(-z))).item())


# ---- data -----------------------------------------------------------------------------------

def row(i, prompt=None, chosen="A clear and correct answer about dividends.", rejected="A vague answer.", source="domain"):
    return {"prompt": prompt or f"Question number {i}: what is a dividend?", "preferred": chosen, "not_preferred": rejected,
            "source": source, "line": i}


def test_pair_is_rendered_like_sft_and_ends_on_im_end(tok):
    p = encode_pair(ChatFormat(tok), row(0))
    assert isinstance(p, dict)
    sft_text = ChatFormat(tok).render([{"role": "user", "content": row(0)["prompt"]},
                                       {"role": "assistant", "content": row(0)["preferred"], "reasoning": ""}])
    ids = p["prompt_ids"] + p["chosen_ids"]
    assert tok.decode(ids) == sft_text.removesuffix("\n")                    # same text SFT trained on
    assert tok.decode(p["prompt_ids"]).endswith(ASSISTANT_PART)
    assert tok.decode(p["chosen_ids"][-1:]) == "<|im_end|>"


def test_think_block_goes_through_the_template(tok):
    p = encode_pair(ChatFormat(tok), row(0, chosen="<think>why</think>\nBecause.", rejected="<think>meh</think>\nNo."))
    assert "<think>\nwhy\n</think>\n\nBecause.<|im_end|>" in tok.decode(p["chosen_ids"])


@pytest.mark.parametrize("bad, reason", [
    (dict(preferred="same", not_preferred="same"), "identical"),
    (dict(preferred="  ", not_preferred="x"), "empty"),
    (dict(prompt="   "), "empty"),
])
def test_invalid_pairs_come_back_as_a_reason(tok, bad, reason):
    out = encode_pair(ChatFormat(tok), row(0, **bad))
    assert isinstance(out, str) and reason in out


def test_prepare_drops_whole_and_counts_every_reason(tok):
    rows = [row(i) for i in range(30)]
    rows.append(row(100, chosen="<think>x</think>\nAnswer.", rejected="Answer without thinking, similar length."))   # think mismatch
    rows.append(row(101, chosen="word " * 100, rejected="word"))                                                     # length ratio
    rows.append(row(102, prompt="long " * 600))                                                                     # too long
    rows.append(row(103, chosen="same", rejected="same"))                                                           # identical
    out = prepare(rows, tok, OPTS, seed=1)
    assert sum(out.dropped.values()) == 4 and len(out.train) + sum(map(len, out.val.values())) == 30
    assert any("<think>" in why for why in out.dropped) and any("2.0x" in why for why in out.dropped)
    assert any("400 tokens" in why for why in out.dropped)
    assert all(side <= 400 for side in (len(p["prompt_ids"]) + len(p["chosen_ids"]) for p in out.train))


def test_split_is_by_prompt_never_leaks_and_is_reproducible(tok):
    rows = [row(i, prompt=f"Shared question {i % 20}", chosen=f"Good answer {i}.", rejected=f"Bad {i}.") for i in range(60)]   # 3 pairs per prompt
    a, b = prepare(rows, tok, OPTS, seed=7), prepare(rows, tok, OPTS, seed=7)
    train = {tuple(p["prompt_ids"]) for p in a.train}
    val = {tuple(p["prompt_ids"]) for v in a.val.values() for p in v}
    assert val and train and not (train & val)
    assert [p["line"] for p in a.train] == [p["line"] for p in b.train]
    assert "prompts in both: 0" in a.report[-1]


def test_length_bias_warning(tok):
    rows = [row(i, chosen="A much longer chosen answer with many more words in it.", rejected="Short one.") for i in range(20)]
    rep = prepare(rows, tok, {**OPTS, "max_length_ratio": 10.0}, seed=1).report
    assert any("WARNING" in l and "verbosity" in l for l in rep)


def test_read_pairs_accepts_source_type_alias(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text(json.dumps({"prompt": "q", "preferred": "a", "not_preferred": "b", "source_type": "esg"}) + "\n\n" +
                 json.dumps({"prompt": "q2", "preferred": "a", "not_preferred": "b"}) + "\n")
    got = read_pairs(f)
    assert [r["source"] for r in got] == ["esg", "unknown"] and [r["line"] for r in got] == [0, 2]


def test_collate_masks_prompt_and_padding_and_orders_chosen_then_rejected():
    a = {"prompt_ids": [1, 2, 3], "chosen_ids": [10, 11], "rejected_ids": [20]}
    b = {"prompt_ids": [4], "chosen_ids": [12], "rejected_ids": [21, 22, 23]}
    out = collate_pairs([a, b], pad_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3, 10, 11], [4, 12, 0, 0, 0], [1, 2, 3, 20, 0], [4, 21, 22, 23, 0]]
    assert out["labels"].tolist() == [[-100, -100, -100, 10, 11], [-100, 12, -100, -100, -100],
                                      [-100, -100, -100, 20, -100], [-100, 21, 22, 23, -100]]
    assert out["attention_mask"].tolist()[1] == [1, 1, 0, 0, 0]
