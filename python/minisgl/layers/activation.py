from __future__ import annotations

import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul", "situ_and_mul"]


def situ_and_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """SiTU: beta*tanh(gate/beta)*sigmoid(gate), optionally clip up."""
    if beta <= 0 or (linear_beta is not None and linear_beta <= 0):
        raise ValueError("SiTU beta and linear_beta must be positive")
    g, u = gate.float(), up.float()
    g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (g * u).to(gate.dtype)
