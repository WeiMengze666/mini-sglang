from abc import ABC, abstractmethod

import torch
from minisgl.distributed.partition import Partition


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
        expert_partition: Partition = Partition(0, 1),
        num_global_experts: int | None = None,
    ) -> torch.Tensor: ...
