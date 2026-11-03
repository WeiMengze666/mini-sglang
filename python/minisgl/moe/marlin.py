"""Tensor-layout adapter for mature Marlin MXFP4 expert kernels.

The backend owns packing and integer route metadata. GEMMs, activations and
top-k reduction are supplied by the optional SGLang kernel SDK.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from minisgl.layers.activation import resolve_gated_activation
from minisgl.precision import use_strict_bf16

from .fused import FusedMoe
from .routing import RoutingPolicy, align_routes


class MarlinMoeBackend(FusedMoe):
    activation_precisions = ("bf16",)


    def prepare_experts(self, experts, **options):
        from . import create_expert_executor

        return create_expert_executor(experts, backend_override="marlin", **options)


class MarlinMXFP4Experts:
    supports_cuda_graph = True
    input_dtype = torch.bfloat16

    def __init__(
        self,
        experts,
        *,
        activation,
        beta=1.0,
        linear_beta=None,
        num_global_experts=None,
        routing_policy=None,
    ):
        items = list(experts)
        if not items:
            raise ValueError("Marlin requires at least one local expert")
        self._activation = resolve_gated_activation(activation, beta=beta, linear_beta=linear_beta)
        if self._activation.name not in ("silu", "situ"):
            raise ValueError("Marlin supports gated SiLU or SiTU experts")
        self._start = items[0][0]
        self._count = len(items)
        self._global = num_global_experts or self._start + self._count
        if self._start < 0 or self._start + self._count > self._global:
            raise ValueError("Local experts must fit the global expert range")
        first = items[0][1]
        self._hidden = first.w1._input_size
        intermediate = first.w1._output_size
        if self._hidden <= 0 or self._hidden % 32:
            raise ValueError("Marlin MXFP4 requires hidden size divisible by 32")
        for offset, (expert_id, expert) in enumerate(items):
            if expert_id != self._start + offset:
                raise ValueError("Marlin local expert IDs must be contiguous")
            expected = ((intermediate, self._hidden),) * 2 + ((self._hidden, intermediate),)
            for op, shape in zip((expert.w1, expert.w3, expert.w2), expected):
                if (op._output_size, op._input_size) != shape or op.bias is not None:
                    raise ValueError("Marlin expert projections must be homogeneous and bias-free")
                if not op.weight_packed.is_cuda or op.weight_packed.dtype != torch.uint8:
                    raise ValueError("Marlin requires CUDA uint8 packed weights")
                if (
                    op.weight_scale.dtype != torch.uint8
                    or op.weight_scale.device != op.weight_packed.device
                ):
                    raise ValueError("Marlin requires raw E8M0 scale bytes on the weight device")
        self._stable = use_strict_bf16() or (routing_policy or RoutingPolicy()).stable
        self._ops = self._load_kernels()
        self._params = self._prepare_weights(items)
        self._intermediate = self._params.w2_weight.shape[1] * 16
        self._kernel_hidden = self._params.w13_weight.shape[1] * 16

    @staticmethod
    def _load_kernels():
        try:
            from sgl_kernel.scalar_type import scalar_types
            from sglang.kernels.ops.activation.activation import silu_and_mul
            from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
            from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
            from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import situ_and_mul
            from sglang.srt.layers.quantization.marlin_utils_fp4 import (
                prepare_moe_mxfp4_layer_for_marlin,
            )
            from tvm_ffi import use_torch_stream
        except ImportError as exc:
            raise RuntimeError("Marlin MXFP4 requires the optional SGLang kernel SDK") from exc
        return SimpleNamespace(
            stream=use_torch_stream,
            align=moe_align_block_size,
            gemm=moe_wna16_marlin_gemm,
            silu=silu_and_mul,
            situ=situ_and_mul,
            sum=moe_topk_sum,
            prepare=prepare_moe_mxfp4_layer_for_marlin,
            qtype=scalar_types.float4_e2m1f,
        )

    @staticmethod
    def _pad_hidden(w13, w2, s13, s2):
        # The mature FP4 GEMM2 portfolio needs a 128-aligned output width.
        # Packed zero weights and unit E8M0 scales add no model features.
        padding = (-(w13.shape[-1] * 2)) % 128
        if not padding:
            return w13, w2, s13, s2
        return (
            F.pad(w13, (0, padding // 2)),
            F.pad(w2, (0, 0, 0, padding)),
            F.pad(s13, (0, padding // 32), value=127),
            F.pad(s2, (0, 0, 0, padding), value=127),
        )

    def _prepare_weights(self, items):
        device = items[0][1].w1.weight_packed.device
        # Preserve checkpoint bytes on CPU while replacing raw layer allocations.
        # Engine's backend storage contract keeps them out of a global raw arena.
        sources = []
        for _, expert in items:
            ops = (expert.w1, expert.w3, expert.w2)
            packed = tuple(op.weight_packed.detach().cpu() for op in ops)
            scales = tuple(op.weight_scale.detach().cpu() for op in ops)
            for op in ops:
                op.weight_packed = torch.empty(0, dtype=torch.uint8, device=device)
                op.weight_scale = torch.empty(0, dtype=torch.uint8, device=device)
            sources.append((packed, scales))
        w13, w2, s13, s2 = self._pad_hidden(
            torch.stack([torch.cat((w[0], w[1])) for w, _ in sources]),
            torch.stack([w[2] for w, _ in sources]),
            torch.stack([torch.cat((s[0], s[1])) for _, s in sources]),
            torch.stack([s[2] for _, s in sources]),
        )

        def parameter(value, dtype):
            return torch.nn.Parameter(value.to(device).view(dtype), False)

        p = SimpleNamespace(
            w13_weight=parameter(w13, torch.int8),
            w2_weight=parameter(w2, torch.int8),
            w13_weight_scale=parameter(s13, torch.float8_e8m0fnu),
            w2_weight_scale=parameter(s2, torch.float8_e8m0fnu),
            orig_dtype=torch.bfloat16,
        )
        with self._ops.stream():
            self._ops.prepare(p)
        return p

    def forward(self, x, weights, ids):
        if (
            x.ndim != 2
            or x.shape[1] != self._hidden
            or x.dtype != self.input_dtype
            or not x.is_cuda
        ):
            raise ValueError("Marlin expects CUDA BF16 [tokens, hidden] inputs")
        if ids.ndim != 2 or weights.shape != ids.shape or ids.shape[0] != len(x):
            raise ValueError("Marlin routes and weights must match [tokens, topk]")
        if not len(x):
            return x.new_empty(x.shape)
        x = (
            F.pad(x, (0, self._kernel_hidden - self._hidden))
            if self._kernel_hidden != self._hidden
            else x.contiguous()
        )
        weights = weights.float().contiguous()
        local = ids - self._start
        local = local.masked_fill((local < 0) | (local >= self._count), -1).int().contiguous()
        tokens, topk = ids.shape
        block = next((b for b in (8, 16, 32, 48, 64) if tokens * topk / self._count / b < 0.9), 64)
        if self._stable:
            sorted_ids, expert_ids, length = align_routes(local, block, self._global)
        else:
            with self._ops.stream():
                sorted_ids, expert_ids, length = self._ops.align(local, block, self._global)
        p = self._params
        routes, intermediate = tokens * topk, self._intermediate
        projected = x.new_zeros((routes, 2 * intermediate))
        activated = x.new_empty((routes, intermediate))
        partials = x.new_zeros((routes, self._kernel_hidden))

        def gemm(a, output, packed, scale, m, n, k, route_topk, weighted):
            return self._ops.gemm(
                a,
                output,
                packed,
                None,
                scale,
                None,
                None,
                None,
                None,
                p.workspace,
                sorted_ids,
                expert_ids,
                length,
                weights,
                moe_block_size=block,
                top_k=route_topk,
                mul_topk_weights=weighted,
                is_ep=True,
                b_q_type=self._ops.qtype,
                size_m=m,
                size_n=n,
                size_k=k,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
            )

        with self._ops.stream():
            gemm(
                x,
                projected,
                p.w13_weight,
                p.w13_weight_scale,
                tokens,
                2 * intermediate,
                self._kernel_hidden,
                topk,
                False,
            )
            if self._activation.name == "situ":
                self._ops.situ(
                    activated, projected, self._activation.beta, self._activation.linear_beta
                )
            else:
                self._ops.silu(projected, activated)
            gemm(
                activated,
                partials,
                p.w2_weight,
                p.w2_weight_scale,
                routes,
                self._kernel_hidden,
                intermediate,
                1,
                True,
            )
            output = x.new_empty((tokens, self._kernel_hidden))
            self._ops.sum(partials.view(tokens, topk, self._kernel_hidden), output)
        return (
            output
            if self._kernel_hidden == self._hidden
            else output[:, : self._hidden].contiguous()
        )
