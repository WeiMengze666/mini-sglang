from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import torch
from minisgl.distributed.info import DistributedInfo


@dataclass(frozen=True)
class DeferredMoEOutput:
    """Local unfinalized expert values and their native weighted-row mapping."""

    values: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class BaseMoeBackend(ABC):
    activation_precisions = ("bf16", "mxfp8")

    def validate_precision(self, precision: str) -> None:
        if precision not in self.activation_precisions:
            raise ValueError(
                f"{type(self).__name__} supports activation precision {self.activation_precisions}"
            )

    def route(self, logits, **options):
        """Return optimized (weights, IDs), or defer to portable routing."""
        return None

    def prepare_experts(self, experts, **options):
        from . import create_expert_executor

        return create_expert_executor(experts, **options)

    @property
    def requires_preparation(self) -> bool:
        return False

    def prepare(self, model, context):
        """Optional startup session; unsupported backends do not import FI."""

    def prepare_transport(self, plan, context):
        """Optional persistent resources after metadata agreement, before cache sizing."""

    def close(self) -> None:
        """Release backend resources after graphs and before borrowed groups."""

    def prepare_local(self, executor, routing_builder, context, *, dry_run=False):
        return None

    def supports_out(self, executor, dtype: torch.dtype) -> bool:
        """Caller-owned expert output currently requires unscattered plain TP."""
        from minisgl.distributed.layout import get_parallel_layout

        supports = getattr(executor, "supports_out", None)
        return get_parallel_layout().ep_size == 1 and callable(supports) and supports(dtype)

    def supports_deferred_output(self, executor, dtype: torch.dtype) -> bool:
        supports = getattr(executor, "supports_deferred_output", None)
        return self.supports_out(executor, dtype) and callable(supports) and supports(dtype)

    def forward_experts(self, executor, x, routing, *, out=None, deferred: bool = False):
        """Let an executor fuse routing with compute, then use ordinary routes."""
        if out is not None and not self.supports_out(executor, out.dtype):
            raise NotImplementedError(
                "Caller-owned expert output requires plain TP and executor support"
            )
        if deferred:
            if out is None or not self.supports_deferred_output(executor, out.dtype):
                raise NotImplementedError("Deferred expert output requires a plain-TP executor")
            return executor.forward_deferred(self, x, routing, out=out)
        options = {} if out is None else {"out": out}
        fused = getattr(executor, "try_forward_routed", None)
        output = fused(x, routing, **options) if fused is not None else None
        if output is not None:
            return output
        weights, ids = routing.select(self)
        return executor.forward(x, weights, ids, **options)

    @cached_property
    def epilogue(self):
        from .epilogue import MoEEpilogue

        return MoEEpilogue(self.reduce_outputs)

    def reduce_outputs(self, parallel, *outputs, **options):
        from .epilogue import reduce_outputs

        return reduce_outputs(parallel, *outputs, **options)

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
        expert_partition: DistributedInfo = DistributedInfo(0, 1),
        num_global_experts: int | None = None,
    ) -> torch.Tensor: ...
