"""Grouped routing, shared experts and an optional normalized latent mixture."""

from __future__ import annotations

import torch
from minisgl.distributed.layout import get_parallel_layout
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.activation import GatedActivation, situ_and_mul
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.linear import Linear, linear_bf16, linear_fp32
from minisgl.layers.norm import TorchRMSNorm
from minisgl.moe import TorchExpertExecutor, create_expert_executor
from minisgl.moe.routing import RoutingPolicy

__all__ = ["GatedMLP", "GroupedRouter", "LatentMoE", "situ_and_mul"]


def _activation(gate: torch.Tensor, up: torch.Tensor, config) -> torch.Tensor:
    return GatedActivation.from_config(config).apply(gate, up)


class GatedMLP(BaseOP):
    """Dense/shared MLP: HF gate_proj, up_proj, down_proj names."""

    def __init__(self, config, intermediate_size: int | None = None):
        self._config = config
        self._activation = GatedActivation.from_config(config)
        self._tp = TensorParallel()
        width = intermediate_size if intermediate_size is not None else config.intermediate_size
        width = local_size(width, self._tp.size, "MLP intermediate_size")
        self.gate_proj = Linear(config.hidden_size, width)
        self.up_proj = Linear(config.hidden_size, width)
        self.down_proj = Linear(width, config.hidden_size)
        self._gate_up_weight = None

    def after_load(self):
        self._gate_up_weight = None
        if (
            isinstance(self.gate_proj, Linear)
            and isinstance(self.up_proj, Linear)
            and self.gate_proj.weight_format == self.up_proj.weight_format == "dense"
            and self.gate_proj.weight.is_cuda
            and self.gate_proj.weight.dtype == torch.bfloat16
            and self.up_proj.weight.dtype == torch.bfloat16
            and self.gate_proj.bias is None
            and self.up_proj.bias is None
            and self.gate_proj.weight.is_contiguous()
            and self.up_proj.weight.is_contiguous()
            and self.gate_proj.weight.shape == self.up_proj.weight.shape
        ):
            fused = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0)
            width = self.gate_proj.weight.shape[0]
            self.gate_proj.weight = fused[:width]
            self.up_proj.weight = fused[width:]
            self._gate_up_weight = fused

    def forward_local(self, x: torch.Tensor) -> torch.Tensor:
        if self._gate_up_weight is not None and x.is_cuda:
            gate_up = linear_bf16(x, self._gate_up_weight, None)
            width = self.gate_proj.weight.shape[0]
            gate, up = gate_up.split(width, dim=-1)
        else:
            gate, up = self.gate_proj.forward(x), self.up_proj.forward(x)
        return self.down_proj.forward(self._activation.apply(gate, up))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._tp.all_reduce(self.forward_local(x))


class _Expert(BaseOP):
    """Checkpoint expert w1=gate, w2=down, w3=up."""

    def __init__(self, hidden_size: int, intermediate_size: int, config):
        self._config = config
        self._activation = GatedActivation.from_config(config)
        intermediate_size = local_size(
            intermediate_size, get_parallel_layout().expert_tensor.size, "expert intermediate_size"
        )
        self.w1 = Linear(hidden_size, intermediate_size)
        self.w2 = Linear(intermediate_size, hidden_size)
        self.w3 = Linear(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(self._activation.apply(self.w1.forward(x), self.w3.forward(x)))


class GroupedRouter(BaseOP):
    def __init__(self, config):
        self._config = config
        self.weight = torch.empty(config.num_experts, config.hidden_size)
        self.e_score_correction_bias = (
            torch.empty(config.num_experts, dtype=torch.float32)
            if getattr(config, "topk_method", "noaux_tc") == "noaux_tc"
            else None
        )
        groups = config.num_expert_group
        if groups < 1 or config.num_experts % groups:
            raise ValueError("num_experts must be divisible by num_expert_group")
        if not 1 <= config.topk_group <= groups:
            raise ValueError("topk_group must be in [1, num_expert_group]")
        if (
            not 1
            <= config.num_experts_per_token
            <= config.topk_group * config.num_experts // groups
        ):
            raise ValueError("top-k exceeds selected expert group capacity")
        if self.e_score_correction_bias is not None and config.num_experts // groups < 2:
            raise ValueError("Bias-corrected group routing needs at least two experts per group")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self._config
        logits = linear_fp32(x, self.weight)
        if c.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif c.moe_router_activation_func == "softmax" and self.e_score_correction_bias is None:
            scores = logits.softmax(-1)
        else:
            raise ValueError("Bias-corrected routing requires sigmoid scores")
        choices = scores
        if self.e_score_correction_bias is not None:
            choices = choices + self.e_score_correction_bias.float()
        grouped = choices.reshape(x.shape[0], c.num_expert_group, -1)
        group_scores = (
            grouped.topk(2, dim=-1).values.sum(-1)
            if self.e_score_correction_bias is not None
            else grouped.amax(-1)
        )
        group_ids = group_scores.topk(c.topk_group, dim=-1, sorted=False).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_ids, True)
        mask = mask.unsqueeze(-1).expand_as(grouped).reshape_as(choices)
        ids = (
            choices.masked_fill(~mask, -torch.inf)
            .topk(c.num_experts_per_token, dim=-1, sorted=False)
            .indices
        )
        # Correction bias affects selection only, never the mixing weights.
        weights = scores.gather(1, ids)
        if c.moe_renormalize:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * c.routed_scaling_factor, ids


class LatentMoE(BaseOP):
    def __init__(self, config):
        self._config = config
        self._tp = TensorParallel()
        self.gate = GroupedRouter(config)
        latent = getattr(config, "routed_expert_hidden_size", None)
        width = latent or config.hidden_size
        expert_ids = get_parallel_layout().expert_range(config.num_experts)
        self.experts = OPList(
            [_Expert(width, config.moe_intermediate_size, config) for _ in expert_ids],
            indices=expert_ids,
        )
        shared = getattr(config, "num_shared_experts", None) or 0
        self.shared_experts = (
            GatedMLP(config, config.moe_intermediate_size * shared) if shared else None
        )
        self.routed_expert_down_proj = Linear(config.hidden_size, width) if latent else None
        self.routed_expert_up_proj = Linear(width, config.hidden_size) if latent else None
        self.routed_expert_norm = (
            TorchRMSNorm(width, config.rms_norm_eps)
            if latent and config.latent_moe_use_norm
            else None
        )
        self._expert_executor = TorchExpertExecutor(self.experts.items())

    def after_load(self):
        self._expert_executor = create_expert_executor(
            self.experts.items(),
            activation=self._config.hidden_act,
            beta=getattr(self._config, "activation_situ_beta", None) or 1.0,
            linear_beta=getattr(self._config, "activation_situ_linear_beta", None),
            num_global_experts=self._config.num_experts,
            routing_policy=RoutingPolicy.from_config(self._config),
        )

    def validate_cuda_graph(self):
        if not self._expert_executor.supports_cuda_graph:
            raise ValueError("CUDA graphs require a device-side grouped expert executor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if not x.shape[0]:
            return x.reshape(shape)
        weights, ids = self.gate.forward(x)
        routed = self.routed_expert_down_proj.forward(x) if self.routed_expert_down_proj else x
        # Accumulate mixture in FP32. RMSNorm is AFTER expert reduction, per token.
        mixed = self._expert_executor.forward(routed, weights, ids)
        # Routed and shared partials use the same rank world and token rows.
        # Pack one FP32 reduction before applying the nonlinear latent norm/up.
        shared = self.shared_experts.forward_local(x) if self.shared_experts else None
        if shared is not None:
            widths = (mixed.shape[-1], shared.shape[-1])
            mixed, shared = self._tp.all_reduce(torch.cat((mixed, shared.float()), -1)).split(
                widths, -1
            )
        else:
            mixed = self._tp.all_reduce(mixed)
        output = mixed.to(x.dtype)
        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm.forward(output)
        if self.routed_expert_up_proj is not None:
            output = self.routed_expert_up_proj.forward(output)
        if shared is not None:
            output = output + shared.to(x.dtype)
        return output.reshape(shape)
