"""Per-token attention over a bank of frozen block sums."""

import torch
import torch.nn.functional as F


def attention_residual(head, bank, norm, proj):
    """Score frozen block sums and the current block independently per token."""
    if not bank:
        return head
    rows = torch.stack([*bank, head], dim=1)
    if (
        head.is_cuda
        and head.dtype == norm.weight.dtype == proj.weight.dtype == torch.bfloat16
        and norm.weight.device == proj.weight.device == head.device
        and norm.weight.is_contiguous()
        and proj.weight.is_contiguous()
        and head.ndim == 2
        and proj.weight.shape == (1, head.shape[-1])
    ):
        from .attn_residual_kernel import attention_residual as fused_residual

        return fused_residual(rows, norm.weight, proj.weight, norm.eps)
    # Keep the score and mixture reductions in FP32.
    xf = rows.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    scores = F.linear(normalized * norm.weight.float(), proj.weight.float())
    weights = scores.softmax(dim=1)
    return (weights * xf).sum(dim=1).to(head.dtype)
