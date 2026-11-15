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

    def supports_out(self, executor, dtype: torch.dtype) -> bool:
        """Caller-owned expert output currently requires unscattered plain TP."""
        from minisgl.distributed.layout import get_parallel_layout

        supports = getattr(executor, "supports_out", None)
        return get_parallel_layout().ep_size == 1 and callable(supports) and supports(dtype)

    def forward_experts(self, executor, x, routing, *, out=None):
        """Let an executor fuse routing with compute, then use ordinary routes."""
        if out is not None and not self.supports_out(executor, out.dtype):
            raise NotImplementedError(
                "Caller-owned expert output requires plain TP and executor support"
            )
        options = {} if out is None else {"out": out}
        fused = getattr(executor, "try_forward_routed", None)
        output = fused(x, routing, **options) if fused is not None else None
        if output is not None:
            return output
        weights, ids = routing.select(self)
        return executor.forward(x, weights, ids, **options)

    def reduce_outputs(
        self, parallel, *outputs, packed=None, output_key=None, norm=None, output_dtype=None
    ):
        """Reduce local partials, then cast and optionally normalize the mixture.

        A prepared same-dtype RMS epilogue can normalize the latent prefix in
        the selected communication backend. Other paths keep the original
        reduce -> mixture cast -> normalizer order, including FP32 partials.
        """
        from minisgl.layers.norm import RMSNormEpilogue, TorchRMSNorm

        if not outputs:
            raise ValueError("MoE reduction needs at least one output")
        sizes = [part.numel() for part in outputs]
        epilogue = None
        if packed is not None:
            if packed.ndim != 1 or not packed.is_contiguous() or packed.numel() != sum(sizes):
                raise ValueError(
                    "Caller-owned packed MoE output must be an exact contiguous flat tensor"
                )
            offset = 0
            for part in outputs:
                if (
                    part.dtype != packed.dtype
                    or part.device != packed.device
                    or not part.is_contiguous()
                    or part.data_ptr() != packed.data_ptr() + offset * packed.element_size()
                ):
                    raise ValueError(
                        "MoE partials must be contiguous consecutive blocks of packed output"
                    )
                offset += part.numel()
            if (
                isinstance(norm, TorchRMSNorm)
                and packed.dtype == output_dtype == norm.weight.dtype == torch.bfloat16
                and outputs[0].ndim == 2
                and outputs[0].shape[1] == norm.weight.numel()
            ):
                epilogue = RMSNormEpilogue(norm, outputs[0].shape[0])
            options = {} if epilogue is None else {"norm": epilogue}
            reduced = parallel.all_reduce_residual(packed, output_key=output_key, **options)
            parts = tuple(
                part.view(original.shape) for part, original in zip(reduced.split(sizes), outputs)
            )
        elif len(outputs) == 1:
            parts = (parallel.all_reduce(outputs[0]),)
        else:
            value = torch.cat([part.to(outputs[0].dtype).flatten() for part in outputs])
            reduced = parallel.all_reduce(value)
            parts = tuple(
                part.view(original.shape) for part, original in zip(reduced.split(sizes), outputs)
            )
        if epilogue is not None:
            return parts  # Native or callable fallback already normalized the prefix once.
        mixed, *tail = parts
        if output_dtype is not None:
            mixed = mixed.to(output_dtype)
        if norm is not None:
            mixed = norm.forward(mixed)
        return (mixed, *tail)

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
