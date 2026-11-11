from __future__ import annotations

from dataclasses import dataclass
from functools import cache

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

    def apply(
        self, gate: torch.Tensor, up: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Apply the activation, optionally selecting its output storage dtype."""
        if self.name == "situ":
            return situ_and_mul(gate, up, self.beta, self.linear_beta, dtype=dtype)
        output = F.silu(gate) * up if self.name == "silu" else F.gelu(gate) * up
        return output if dtype is None else output.to(dtype)


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
    "sigmoid_and_mul",
]


@cache
def _activation_vector_bytes(device: torch.device) -> int:
    """Match the SDK's architecture-dependent maximum vector width."""
    return 32 if torch.cuda.get_device_capability(device)[0] >= 10 else 16


def situ_and_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """SiTU in FP32 arithmetic, with an explicit optional output storage dtype."""
    if beta <= 0 or (linear_beta is not None and linear_beta <= 0):
        raise ValueError("SiTU beta and linear_beta must be positive")
    dtype = gate.dtype if dtype is None else dtype
    if (
        gate.is_cuda
        and dtype == torch.bfloat16
        and gate.ndim == up.ndim == 2
        and gate.shape == up.shape
        and gate.device == up.device
        and gate.dtype == up.dtype
        and gate.dtype in (torch.bfloat16, torch.float32)
        # SDK vectors use the wider input/output type.
        and gate.shape[-1] > 0
        and gate.shape[-1]
        % (_activation_vector_bytes(gate.device) // max(gate.element_size(), dtype.itemsize))
        == 0
    ):
        from sglang.kernels.ops.activation import situ_and_mul as fused_situ
        from tvm_ffi import use_torch_stream

        width = gate.shape[-1]
        adjacent = (
            gate.stride(-1) == up.stride(-1) == 1
            and gate.stride() == up.stride()
            and gate.untyped_storage()._cdata == up.untyped_storage()._cdata
            and up.storage_offset() == gate.storage_offset() + width
        )
        packed = (
            gate.as_strided((len(gate), 2 * width), gate.stride())
            if adjacent
            else torch.cat((gate, up), dim=-1)
        )
        output = gate.new_empty(gate.shape, dtype=dtype)
        with use_torch_stream():
            return fused_situ(packed, output, beta, linear_beta)
    g, u = gate.float(), up.float()
    g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (g * u).to(dtype)


@cache
def _sigmoid_mul_kernel():
    """Resolve an optional mature kernel without requiring the SDK for fallback."""
    from importlib import import_module

    name = "sglang.kernels.ops.attention.mla_output_gate"
    try:
        module = import_module(name)
    except ModuleNotFoundError as error:
        if error.name and (name == error.name or name.startswith(error.name + ".")):
            return None
        raise
    from tvm_ffi import use_torch_stream

    return module.covered, module.kimi_k3_mla_output_gate, use_torch_stream


def sigmoid_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Multiply by sigmoid in the input dtype, preserving both rounding stages."""
    if (
        gate.is_cuda
        and up.is_cuda
        and gate.device == up.device
        and gate.dtype == up.dtype == torch.bfloat16
        and not (gate.requires_grad or up.requires_grad)
        # The SDK flattens its element count into a uint32_t.
        and up.numel() < 2**32
        and gate.data_ptr() % 16 == up.data_ptr() % 16 == 0
    ):
        kernel = _sigmoid_mul_kernel()
        if kernel is not None:
            covered, run, stream = kernel
            if covered(up, gate):
                with stream():
                    return run(up, gate)
    return up * gate.sigmoid()
