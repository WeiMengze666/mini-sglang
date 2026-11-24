import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.distributed.layout import get_parallel_layout
from minisgl.layers.activation import resolve_gated_activation
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()
        self._executor = None

        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = resolve_gated_activation(activation)
        if self.activation.name not in ("silu", "gelu"):
            raise NotImplementedError("The fused MoE backend supports SiLU and GELU only")
        self.apply_router_weight_on_input = apply_router_weight_on_input
        layout = get_parallel_layout()
        self._expert_partition = layout.experts
        self._expert_ids = layout.expert_range(num_experts)
        intermediate_size_per_partition = div_even(intermediate_size, layout.expert_tensor.size)
        self.gate_up_proj = torch.empty(
            len(self._expert_ids),
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        self.down_proj = torch.empty(
            len(self._expert_ids),
            hidden_size,
            intermediate_size_per_partition,
        )

    def after_load(self):
        from minisgl.moe import get_moe_backend

        self._executor = get_moe_backend().prepare_stacked(
            self.gate_up_proj,
            self.down_proj,
            activation=self.activation.name,
            weight_on_input=self.apply_router_weight_on_input,
            partition=self._expert_partition,
            global_experts=self.num_experts,
            top_k=self.top_k,
        )

    def prepare_runtime(self, context, *, dry_run=False):
        return get_global_ctx().moe_backend.prepare_local(
            self._executor, None, context, dry_run=dry_run
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        from minisgl.moe.routing import SoftmaxTopK

        ctx = get_global_ctx()
        if self._executor is None:
            self.after_load()
        routing = SoftmaxTopK(hidden_states, router_logits, self.top_k, self.renormalize)
        final_hidden_states = ctx.moe_backend.forward_experts(
            self._executor, hidden_states, routing
        )
        if self.tp_size > 1:
            (final_hidden_states,) = ctx.moe_backend.reduce_outputs(self._comm, final_hidden_states)
        return final_hidden_states
