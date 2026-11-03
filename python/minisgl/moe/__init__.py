from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch
from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend, TorchExpertExecutor
from .routing import RoutingPolicy

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_BACKENDS.register("marlin")
def create_marlin_moe_backend():
    from .marlin import MarlinMoeBackend

    return MarlinMoeBackend()


def select_moe_backend(quantization, *, dtype: torch.dtype, activation_precision: str) -> str:
    """Select expert compute from the checkpoint format and activation contract."""
    if (
        isinstance(quantization, Mapping)
        and quantization.get("format") == "mxfp4-pack-quantized"
        and dtype == torch.bfloat16
        and activation_precision == "bf16"
    ):
        return "marlin"
    return "fused"


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


def get_moe_backend() -> BaseMoeBackend:
    from minisgl.core import get_global_ctx_or_none

    backend = getattr(get_global_ctx_or_none(), "moe_backend", None)
    return backend if backend is not None else create_moe_backend("fused")


def create_expert_executor(
    experts,
    *,
    activation,
    beta=1.0,
    linear_beta=None,
    num_global_experts=None,
    routing_policy=None,
    backend_override=None,
):
    """Select local expert compute from weight format and execution policy.

    Routing and collectives stay with the caller. Unsupported combinations keep
    the portable per-expert path instead of changing state or activation precision.
    """
    from minisgl.layers.linear import Linear

    items = list(experts)
    portable = TorchExpertExecutor(items)
    projections = [op for _, expert in items for op in (expert.w1, expert.w3, expert.w2)]
    if not projections or not all(
        isinstance(op, Linear)
        and op.weight_format == "mxfp4"
        and op.weight_packed.is_cuda
        and op.weight_packed.dtype == torch.uint8
        and op.weight_scale.dtype == torch.uint8
        and op.weight_packed.is_contiguous()
        and op.weight_scale.is_contiguous()
        and op.weight_scale.device == op.weight_packed.device
        and op.bias is None
        for op in projections
    ):
        return portable
    backend, precision = projections[0]._backend, projections[0]._activation_precision
    if any((op._backend, op._activation_precision) != (backend, precision) for op in projections):
        return portable
    backend = backend_override or backend
    if backend == "marlin":
        if precision != "bf16":
            raise ValueError("Marlin MXFP4 requires BF16 activations")
        from .marlin import MarlinMXFP4Experts

        return MarlinMXFP4Experts(
            items,
            activation=activation,
            beta=beta,
            linear_beta=linear_beta,
            num_global_experts=num_global_experts,
            routing_policy=routing_policy,
        )
    if backend == "triton":
        from .mxfp4 import MXFP4Experts

        return MXFP4Experts(
            items,
            activation=activation,
            beta=beta,
            linear_beta=linear_beta,
            routing_policy=routing_policy,
        )
    if backend == "flashinfer":
        from .flashinfer import FlashInferMXFP4Experts

        return FlashInferMXFP4Experts(
            items,
            activation=activation,
            beta=beta,
            linear_beta=linear_beta,
            num_global_experts=num_global_experts,
            activation_precision=precision,
        )
    return portable


__all__ = [
    "BaseMoeBackend",
    "TorchExpertExecutor",
    "RoutingPolicy",
    "create_moe_backend",
    "select_moe_backend",
    "get_moe_backend",
    "create_expert_executor",
    "SUPPORTED_MOE_BACKENDS",
]
