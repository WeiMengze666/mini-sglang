from __future__ import annotations

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


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


def create_expert_executor(
    experts,
    *,
    activation,
    beta=1.0,
    linear_beta=None,
    num_global_experts=None,
    routing_policy=None,
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
    "create_expert_executor",
    "SUPPORTED_MOE_BACKENDS",
]
