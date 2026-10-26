"""Per-token attention over a bank of frozen block sums."""

import torch
import torch.nn.functional as F


def attention_residual(head, bank, norm, proj):
    """Score frozen block sums and the current block independently per token."""
    if not bank:
        return head
    rows = torch.stack([*bank, head], dim=1)
    # Keep the score and mixture reductions in FP32.
    xf = rows.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    scores = F.linear(normalized * norm.weight.float(), proj.weight.float())
    weights = scores.softmax(dim=1)
    return (weights * xf).sum(dim=1).to(head.dtype)
