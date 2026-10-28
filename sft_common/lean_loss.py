"""A memory-lean version of the language-model loss.

The standard loss first turns EVERY token's hidden state into scores over the whole vocabulary
(Qwen3.5: 248,320 words), then converts those scores to 32-bit numbers and keeps several copies for
the backward pass. For a 12,000-token row that is tens of GB, and 80% of a domain row is the
source document, which is context only and never scored.

This version:
  1. scores only the positions whose next token is a training target (labels != -100), and
  2. does that in chunks, recomputing each chunk's scores during the backward pass, so only one
     chunk of scores exists at a time.
The result is mathematically the same loss and the same gradients (see tests), at a fraction of
the memory.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE = -100          # label value meaning "not a target" (system prompt, question, document, tool results)


def _chunk_sum(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Summed cross-entropy of one chunk: hidden (n, H) x weight (V, H) against targets (n,)."""
    logits = (hidden @ weight.t()).float()             # 32-bit, like the standard loss
    return F.cross_entropy(logits, targets, reduction="sum")


def chunked_ce_sum(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Sum of cross-entropy over all rows, never holding more than `chunk` rows of vocabulary scores.

    Each chunk is wrapped in a checkpoint, so its scores are dropped after the forward pass and
    recomputed when the backward pass needs them.
    """
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[0], chunk):
        total = total + checkpoint(_chunk_sum, hidden[i:i + chunk], weight, targets[i:i + chunk], use_reentrant=False)
    return total


def lean_loss(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
              num_items: int | float | torch.Tensor | None = None, chunk: int = 2048) -> tuple[torch.Tensor, int]:
    """Next-token loss over the target positions only.

    hidden: (B, T, H) final hidden states.  weight: (V, H) output layer.  labels: (B, T), IGNORE = not a target.
    Position t predicts token t+1, so hidden[:, :-1] is scored against labels[:, 1:] (the standard shift).
    Returns (loss, number of target tokens). loss = summed cross-entropy / num_items when given
    (the whole accumulation window's target count), else the mean over this batch's targets.
    """
    shifted = labels[:, 1:]
    mask = shifted != IGNORE
    n = int(mask.sum())
    if n == 0:                                          # nothing to learn from here: zero loss that still has a graph
        return hidden.sum() * 0.0, 0
    total = chunked_ce_sum(hidden[:, :-1][mask], weight, shifted[mask], chunk)
    return total / (num_items if num_items is not None else n), n
