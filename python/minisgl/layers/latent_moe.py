"""Grouped routing, shared experts and an optional normalized latent mixture."""

from __future__ import annotations

import torch
from minisgl.distributed.layout import get_parallel_layout
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.activation import GatedActivation, situ_and_mul
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.linear import Linear, MergedLinear, linear_bf16, linear_fp32
from minisgl.layers.norm import TorchRMSNorm
from minisgl.moe import TorchExpertExecutor, get_moe_backend
from minisgl.moe.routing import GroupedRouting, RoutingPolicy
from minisgl.precision import use_strict_bf16

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
        if self.gate_proj.weight.is_meta:
            MergedLinear.try_create({"gate": self.gate_proj, "up": self.up_proj})

    def after_load(self):
        self._gate_up_weight = None
        if (
            getattr(self.gate_proj, "weight", None) is not None
            and self.gate_proj.weight.is_cuda
            and self.gate_proj.weight.dtype == torch.bfloat16
        ):
            plan = MergedLinear.try_create({"gate": self.gate_proj, "up": self.up_proj})
            if plan is not None:
                self._gate_up_weight = plan.weight

    def forward_from_projections(self, gate, up, *, dtype):
        activated = self._activation.apply(gate, up, dtype=dtype)
        return self.down_proj.forward(activated)

    def forward_local(
        self, x: torch.Tensor, *, projection_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Optionally retain dense gate/up accumulators through the activation."""
        if projection_dtype not in (None, torch.float32):
            raise ValueError("Gate/up projection dtype must be None or float32")
        if projection_dtype is not None and any(
            op.weight_format != "dense" or op.bias is not None
            for op in (self.gate_proj, self.up_proj)
        ):
            raise ValueError("FP32 gate/up projection requires dense, bias-free weights")
        if self._gate_up_weight is not None and x.is_cuda:
            project = linear_fp32 if projection_dtype is not None else linear_bf16
            gate_up = project(x, self._gate_up_weight)
            width = self.gate_proj.weight.shape[0]
            gate, up = gate_up.split(width, dim=-1)
        elif projection_dtype is not None:
            gate = linear_fp32(x, self.gate_proj.weight)
            up = linear_fp32(x, self.up_proj.weight)
        else:
            gate, up = self.gate_proj.forward(x), self.up_proj.forward(x)
        return self.forward_from_projections(gate, up, dtype=x.dtype)

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
        self._backend = get_moe_backend()
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
        return self.route(linear_fp32(x, self.weight))

    def routing(self, logits: torch.Tensor) -> GroupedRouting:
        """Keep scores available for an executor that fuses routing and quantization."""
        c = self._config
        if logits.ndim != 2 or logits.shape[1] != c.num_experts:
            raise ValueError("Router logits must have one column per expert")
        return GroupedRouting(
            logits=logits,
            bias=self.e_score_correction_bias,
            topk=c.num_experts_per_token,
            num_groups=c.num_expert_group,
            topk_groups=c.topk_group,
            scoring_func=c.moe_router_activation_func,
            renormalize=c.moe_renormalize,
            scale=c.routed_scaling_factor,
        )

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routing(logits).select(self._backend)


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
        self._backend = get_moe_backend()
        self._expert_executor = TorchExpertExecutor(self.experts.items())
        self._front = None
        if self.gate.weight.is_meta:
            MergedLinear.try_create(self._front_projections())

    def _front_projections(self):
        if self.shared_experts is None or self.routed_expert_down_proj is None:
            return {}
        return {
            "gate": self.shared_experts.gate_proj,
            "up": self.shared_experts.up_proj,
            "router": self.gate,
            "latent": self.routed_expert_down_proj,
        }

    def after_load(self):
        self._front = None
        if self.gate.weight.is_cuda and self.gate.weight.dtype == torch.bfloat16:
            self._front = MergedLinear.try_create(self._front_projections())
            if self._front is not None:
                # The child plan shares the same allocation after parent fusion.
                width = self.shared_experts.gate_proj.weight.shape[0] * 2
                self.shared_experts._gate_up_weight = self._front.weight[:width]
        self._expert_executor = self._backend.prepare_experts(
            self.experts.items(),
            activation=self._config.hidden_act,
            beta=getattr(self._config, "activation_situ_beta", None) or 1.0,
            linear_beta=getattr(self._config, "activation_situ_linear_beta", None),
            num_global_experts=self._config.num_experts,
            routing_policy=RoutingPolicy.from_config(self._config),
        )

    def prepare_runtime(self, context, *, dry_run=False):
        return self._backend.prepare_local(
            self._expert_executor, self.gate.routing, context, dry_run=dry_run
        )

    def validate_cuda_graph(self):
        if not self._expert_executor.supports_cuda_graph:
            raise ValueError("CUDA graphs require a device-side grouped expert executor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if not x.shape[0]:
            return x.reshape(shape)
        front = self._front.forward(x) if self._front is not None else None
        logits = linear_fp32(x, self.gate.weight) if front is None else front["router"]
        routing = self.gate.routing(logits)
        input_dtype = getattr(self._expert_executor, "input_dtype", None) or x.dtype
        down = self.routed_expert_down_proj
        if down is None:
            routed = x.to(input_dtype)
        elif front is not None:
            routed = front["latent"].to(input_dtype)
        elif input_dtype == torch.float32:
            if down.weight_format != "dense" or down.bias is not None:
                raise ValueError("FP32 expert inputs require a dense, bias-free latent projection")
            routed = linear_fp32(x, down.weight)
        else:
            routed = down.forward(x).to(input_dtype)
        # The executor owns mixture precision; normalization follows expert reduction.
        mixed = self._backend.forward_experts(self._expert_executor, routed, routing)
        strict_bf16 = x.dtype == torch.bfloat16 and use_strict_bf16()
        if strict_bf16:
            mixed = mixed.to(x.dtype)
        shared = None
        if self.shared_experts is not None:
            dense = all(
                op.weight_format == "dense"
                for op in (self.shared_experts.gate_proj, self.shared_experts.up_proj)
            )
            shared = (
                self.shared_experts.forward_from_projections(
                    front["gate"], front["up"], dtype=x.dtype
                )
                if front is not None
                else self.shared_experts.forward_local(
                    x, projection_dtype=torch.float32 if strict_bf16 and dense else None
                )
            )
        # Reduce partials before nonlinear latent normalization/up-projection.
        # Strict BF16 rounds each local mixture before the collective.
        if shared is not None:
            mixed, shared = self._backend.reduce_outputs(self._tp, mixed, shared)
        else:
            (mixed,) = self._backend.reduce_outputs(self._tp, mixed)
        output = mixed.to(x.dtype)
        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm.forward(output)
        if self.routed_expert_up_proj is not None:
            output = self.routed_expert_up_proj.forward(output)
        if shared is not None:
            output = output + shared.to(x.dtype)
        return output.reshape(shape)
