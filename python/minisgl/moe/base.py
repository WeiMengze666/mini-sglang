from abc import ABC, abstractmethod

import torch
from minisgl.distributed.info import DistributedInfo


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

    def prepare(self, model, context):
        """Optional startup session; unsupported backends do not import FI."""

    def prepare_local(self, executor, routing_builder, context, *, dry_run=False):
        return None

    def forward_experts(self, executor, x, routing):
        """Let an executor fuse routing with compute, then use ordinary routes."""
        fused = getattr(executor, "try_forward_routed", None)
        output = fused(x, routing) if fused is not None else None
        if output is not None:
            return output
        weights, ids = routing.select(self)
        return executor.forward(x, weights, ids)

    def reduce_outputs(self, parallel, *outputs):
        """Pack expert/shared partials for the selected communication implementation."""
        if not outputs:
            raise ValueError("MoE reduction needs at least one output")
        if len(outputs) == 1:
            return (parallel.all_reduce(outputs[0]),)
        sizes = [part.numel() for part in outputs]
        packed = torch.cat([part.to(outputs[0].dtype).flatten() for part in outputs])
        reduced = parallel.all_reduce(packed)
        return tuple(
            part.view(original.shape) for part, original in zip(reduced.split(sizes), outputs)
        )

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


class TorchExpertExecutor:
    """Portable local expert mixture for arbitrary IDs and projection formats."""

    supports_cuda_graph = False
    input_dtype = None

    def __init__(self, experts):
        self.experts = dict(experts)

    def forward(self, x, weights, ids):
        mixed = torch.zeros_like(x, dtype=torch.float32)
        for expert_id in ids.unique(sorted=True).tolist():
            if expert_id not in self.experts:
                continue
            rows, slots = torch.where(ids == expert_id)
            if rows.numel():
                values = self.experts[expert_id].forward(x[rows]).float()
                mixed.index_add_(0, rows, values * weights[rows, slots, None])
        return mixed
