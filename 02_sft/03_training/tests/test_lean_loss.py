"""The lean loss must equal the standard language-model loss: same value, same gradients."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn.functional as F  # noqa: E402

from sft_common.lean_loss import IGNORE, chunked_ce_sum, lean_loss  # noqa: E402


def standard_loss(hidden, weight, labels):
    """What a normal causal-LM computes: all logits, shift, ignore -100, mean over targets."""
    logits = (hidden @ weight.t()).float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=IGNORE)


def make(B=2, T=37, H=16, V=101, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, T, H, generator=g, dtype=torch.float64).requires_grad_()
    weight = torch.randn(V, H, generator=g, dtype=torch.float64).requires_grad_()
    labels = torch.randint(0, V, (B, T), generator=g)
    labels[:, : T // 2] = IGNORE                          # first half is context, not a target
    labels[0, -5:] = IGNORE                               # and some padding at the end of row 0
    return hidden, weight, labels


@pytest.mark.parametrize("chunk", [1, 7, 2048])           # 1 and 7 cross chunk boundaries, 2048 is a single chunk
def test_same_loss_and_gradients_as_the_standard_loss(chunk):
    hidden, weight, labels = make()
    ref = standard_loss(hidden, weight, labels)
    g_h, g_w = torch.autograd.grad(ref, [hidden, weight])
    loss, n = lean_loss(hidden, weight, labels, chunk=chunk)
    l_h, l_w = torch.autograd.grad(loss, [hidden, weight])
    assert n == int((labels[:, 1:] != IGNORE).sum())
    assert torch.allclose(loss, ref, atol=1e-10)
    assert torch.allclose(l_h, g_h, atol=1e-10) and torch.allclose(l_w, g_w, atol=1e-10)


def test_num_items_normalises_by_the_whole_accumulation_window():
    hidden, weight, labels = make()
    loss_mean, n = lean_loss(hidden, weight, labels)
    loss_window, _ = lean_loss(hidden, weight, labels, num_items=4 * n)         # window holds 4x as many targets
    assert torch.allclose(loss_window * 4, loss_mean, atol=1e-12)


def test_no_targets_gives_zero_loss_that_can_still_backpropagate():
    hidden, weight, _ = make()
    labels = torch.full((2, 37), IGNORE)
    loss, n = lean_loss(hidden, weight, labels)
    assert n == 0 and float(loss) == 0.0
    loss.backward()                                                              # must not raise
    assert float(hidden.grad.abs().sum()) == 0.0


def test_chunked_sum_matches_the_plain_sum():
    hidden, weight, labels = make()
    h = hidden[:, :-1].reshape(-1, hidden.shape[-1]).detach()
    y = torch.where(labels[:, 1:].reshape(-1) == IGNORE, torch.zeros(1, dtype=torch.long), labels[:, 1:].reshape(-1))
    plain = F.cross_entropy((h @ weight.detach().t()).float(), y, reduction="sum")
    assert torch.allclose(chunked_ce_sum(h, weight.detach(), y, chunk=5), plain, atol=1e-9)
