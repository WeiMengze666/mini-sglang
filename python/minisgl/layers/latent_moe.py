"""Grouped routing, shared experts and an optional normalized latent mixture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.activation import situ_and_mul
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.linear import Linear
from minisgl.layers.norm import TorchRMSNorm


def _activation(gate: torch.Tensor, up: torch.Tensor, config) -> torch.Tensor:
    if config.hidden_act == "situ":
        return situ_and_mul(
            gate,
            up,
            getattr(config, "activation_situ_beta", None) or 1.0,
            getattr(config, "activation_situ_linear_beta", None),
        )
    if config.hidden_act == "silu":
        return F.silu(gate) * up
    raise ValueError(f"Unsupported activation: {config.hidden_act}")


class GatedMLP(BaseOP):
    """Dense/shared MLP: HF gate_proj, up_proj, down_proj names."""

    def __init__(self, config, intermediate_size: int | None = None):
        self._config = config
        self._tp = TensorParallel()
        width = intermediate_size if intermediate_size is not None else config.intermediate_size
        width = local_size(width, self._tp.size, "MLP intermediate_size")
        self.gate_proj = Linear(config.hidden_size, width)
        self.up_proj = Linear(config.hidden_size, width)
        self.down_proj = Linear(width, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._tp.all_reduce(
            self.down_proj.forward(
                _activation(self.gate_proj.forward(x), self.up_proj.forward(x), self._config)
            )
        )


class _Expert(BaseOP):
    """Checkpoint expert w1=gate, w2=down, w3=up."""

    def __init__(self, hidden_size: int, intermediate_size: int, config):
        self._config = config
        intermediate_size = local_size(
            intermediate_size, TensorParallel().size, "expert intermediate_size"
        )
        self.w1 = Linear(hidden_size, intermediate_size)
        self.w2 = Linear(intermediate_size, hidden_size)
        self.w3 = Linear(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(_activation(self.w1.forward(x), self.w3.forward(x), self._config))


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
        logits = F.linear(x.float(), self.weight.float())
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
        self.experts = OPList(
            [
                _Expert(width, config.moe_intermediate_size, config)
                for _ in range(config.num_experts)
            ]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if not x.shape[0]:
            return x.reshape(shape)
        weights, ids = self.gate.forward(x)
        routed = self.routed_expert_down_proj.forward(x) if self.routed_expert_down_proj else x
        # Accumulate mixture in FP32. RMSNorm is AFTER expert reduction, per token.
        mixed = torch.zeros_like(routed, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts.op_list):
            rows, slots = torch.where(ids == expert_id)
            if rows.numel():
                values = expert.forward(routed[rows]).float() * weights[rows, slots, None]
                mixed.index_add_(0, rows, values)
        # Reduce the full routed mixture once, before nonlinear latent norm/up.
        # Individual experts must never issue collectives: routing is sparse.
        output = self._tp.all_reduce(mixed).to(x.dtype)
        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm.forward(output)
        if self.routed_expert_up_proj is not None:
            output = self.routed_expert_up_proj.forward(output)
        if self.shared_experts is not None:
            output = output + self.shared_experts.forward(x)
        return output.reshape(shape)
