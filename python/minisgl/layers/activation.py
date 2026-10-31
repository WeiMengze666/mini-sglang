from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GatedActivation:
    """Configuration and eager implementation of a gated MLP activation."""

    name: str
    beta: float = 1.0
    linear_beta: float | None = None

    def __post_init__(self):
        name = self.name.lower()
        if name not in {"silu", "gelu", "situ"}:
            raise ValueError(f"Unsupported gated activation: {self.name}")
        if self.beta <= 0 or (self.linear_beta is not None and self.linear_beta <= 0):
            raise ValueError("Gated activation beta values must be positive")
        object.__setattr__(self, "name", name)

    @classmethod
    def from_config(cls, config) -> "GatedActivation":
        return cls(
            config.hidden_act,
            beta=getattr(config, "activation_situ_beta", None) or 1.0,
            linear_beta=getattr(config, "activation_situ_linear_beta", None),
        )

    def apply(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.name == "silu":
            return F.silu(gate) * up
        if self.name == "gelu":
            return F.gelu(gate) * up
        return situ_and_mul(gate, up, self.beta, self.linear_beta)


def resolve_gated_activation(
    name: str, *, beta: float = 1.0, linear_beta: float | None = None
) -> GatedActivation:
    """Resolve a model activation into one shared, validated contract."""
    return GatedActivation(name, beta=beta, linear_beta=linear_beta)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = [
    "GatedActivation",
    "resolve_gated_activation",
    "silu_and_mul",
    "gelu_and_mul",
    "situ_and_mul",
]


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
