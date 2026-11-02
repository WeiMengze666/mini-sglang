from abc import ABC, abstractmethod

import torch
from minisgl.distributed.info import DistributedInfo


class BaseMoeBackend(ABC):
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
