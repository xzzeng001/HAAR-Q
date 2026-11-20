"""Shared path selection for autoregressive sampling without replacement."""

import torch


@torch.jit.script
def select_children(
    normalized_log_amplitudes: torch.Tensor,
    parent_log_prob: torch.Tensor,
    parent_score: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return surviving parent rows, local states, log probabilities and Gumbel scores.

    Conditioning makes the largest child score equal to its parent's score.
    The log-domain form also preserves branches with very small probabilities.
    """
    parents = parent_log_prob.size(0)
    log_prob = parent_log_prob[:, None] + 2 * normalized_log_amplitudes.reshape(parents, -1)
    scores = log_prob - (-torch.rand_like(log_prob).log()).log()
    maxima = scores.amax(dim=1, keepdim=True)
    # exp(-conditioned) = exp(-parent) + exp(-score) * (1 - exp(score - max)).
    correction = (-torch.expm1(scores - maxima)).log()
    conditioned = -torch.logaddexp(-parent_score[:, None], correction - scores)
    conditioned = torch.where(scores.isneginf(), -torch.inf, conditioned)

    # Keep child-major order so all wavefunctions use the same ordering of branches.
    flat = conditioned.transpose(0, 1).reshape(-1)
    selected = flat.argsort(descending=True)[:count]
    selected = selected[flat[selected].isfinite()]
    parent = selected % parents
    child = selected // parents
    return parent, child, log_prob[parent, child], flat[selected]
