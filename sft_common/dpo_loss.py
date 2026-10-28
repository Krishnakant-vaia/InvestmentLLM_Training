"""The DPO maths, memory-lean (same idea as lean_loss.py).

DPO compares how likely the model finds the chosen vs the rejected answer, relative to a frozen
reference. That needs the log-probability of every answer token, and the standard way builds scores
over the whole vocabulary (Qwen3.5: 248,320 words) for EVERY token, in 32-bit, for both answers:
tens of GB. Here only the answer tokens are scored, a chunk at a time, and each chunk is recomputed
in the backward pass, so one chunk of vocabulary scores exists at any moment.

    completion_logps(hidden, weight, labels)  ->  (summed log-prob per row, answer-token count per row)
    dpo_loss(pol_c, pol_r, ref_c, ref_r, beta) ->  (loss, metrics)
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE = -100          # label value meaning "not part of the answer" (prompt, padding)


def _chunk_logps(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-probability of each target token: hidden (n, H) x weight (V, H) against targets (n,)."""
    return -F.cross_entropy((hidden @ weight.t()).float(), targets, reduction="none")


def completion_logps(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                     chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: the summed log-probability of its answer tokens, and how many there are.

    hidden: (B, T, H) final hidden states.  weight: (V, H) output layer.  labels: (B, T), IGNORE = not answer.
    Position t predicts token t+1, so hidden[:, :-1] is scored against labels[:, 1:] (the standard shift).
    """
    shifted = labels[:, 1:]
    mask = shifted != IGNORE
    rows = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(mask)[mask]
    picked, targets = hidden[:, :-1][mask], shifted[mask]
    parts = [checkpoint(_chunk_logps, picked[i:i + chunk], weight, targets[i:i + chunk], use_reentrant=False)
             for i in range(0, picked.shape[0], chunk)]
    token_logps = torch.cat(parts) if parts else hidden.new_zeros(0, dtype=torch.float32)
    sums = torch.zeros(labels.shape[0], dtype=torch.float32, device=labels.device).index_add(0, rows, token_logps)
    return sums + hidden.sum() * 0.0, mask.sum(1)      # "+ 0": keeps a graph even when no row has an answer


def dpo_loss(pol_c: torch.Tensor, pol_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor,
             beta: float, label_smoothing: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    """The DPO loss over a batch of pairs. Inputs are per-pair summed log-probs of the chosen / rejected
    answer under the policy (pol_*) and the frozen reference (ref_*), each of shape (B,).

        z    = beta * [(pol_c - pol_r) - (ref_c - ref_r)]      how much more the policy prefers chosen than the reference does
        loss = -log sigmoid(z)                                  (with label_smoothing e: -(1-e) log s(z) - e log s(-z))

    Metrics (floats, averaged over the pairs): implicit rewards beta*(pol - ref) of both answers, their margin,
    the share of pairs with chosen reward > rejected reward, the raw log-probs, and ref/prefers_chosen = the
    share of pairs the REFERENCE already ranks right (a data check: ~100% = nothing to learn, ~0% = the pairs
    disagree with the SFT model's own taste).
    """
    z = beta * ((pol_c - pol_r) - (ref_c - ref_r))
    loss = -(1.0 - label_smoothing) * F.logsigmoid(z) - label_smoothing * F.logsigmoid(-z)
    rc, rr = (beta * (pol_c - ref_c)).detach(), (beta * (pol_r - ref_r)).detach()
    metrics = {
        "rewards/chosen": rc.mean().item(),
        "rewards/rejected": rr.mean().item(),
        "rewards/margins": (rc - rr).mean().item(),
        "rewards/accuracies": (rc > rr).float().mean().item(),
        "logps/chosen": pol_c.detach().mean().item(),
        "logps/rejected": pol_r.detach().mean().item(),
        "ref/prefers_chosen": (ref_c > ref_r).float().mean().item(),
    }
    return loss.mean(), metrics
