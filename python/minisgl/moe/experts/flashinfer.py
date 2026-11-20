"""Opt-in FlashInfer TRTLLM FP4 grouped experts."""

from __future__ import annotations

import os
from typing import Iterable

import torch
from minisgl.layers.activation import resolve_gated_activation


class FlashInferMXFP4Experts:
    """Grouped MXFP4 experts backed by FlashInfer's SM100 TRTLLM kernel.

    Weight preparation is intentionally eager and immutable after construction;
    forward only prepares routing/activation metadata and launches the fused op.
    """

    supports_cuda_graph = True

    def __init__(
        self,
        experts: Iterable[tuple[int, object]],
        *,
        activation: str,
        beta: float,
        linear_beta: float | None,
        num_global_experts: int,
        activation_precision: str,
    ):
        activation_spec = resolve_gated_activation(activation, beta=beta, linear_beta=linear_beta)
        if activation_spec.name not in ("silu", "situ"):
            raise ValueError("FlashInfer FP4 supports gated silu or situ")
        if activation_precision not in ("bf16", "mxfp8"):
            raise ValueError("FlashInfer activation precision must be bf16 or mxfp8")
        if activation_spec.name == "situ" and activation_precision != "mxfp8":
            raise ValueError("FlashInfer SiTU requires mxfp8 activation precision")
        if activation_spec.name == "situ" and activation_spec.linear_beta is None:
            raise ValueError("FlashInfer SiTU requires linear_beta")
        items = list(experts)
        if not items:
            raise ValueError("FlashInfer FP4 requires at least one local expert")
        if num_global_experts < 1:
            raise ValueError("num_global_experts must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("FlashInfer FP4 grouped experts require CUDA")
        if items[0][0] < 0 or items[-1][0] >= num_global_experts:
            raise ValueError("FlashInfer expert IDs must fit the global expert range")
        self._activation = activation_spec.name
        self._precision = activation_precision
        self._start = items[0][0]
        self._global_experts = num_global_experts
        self._experts = len(items)
        first = items[0][1]
        self._hidden = first.w1._input_size
        self._intermediate = first.w1._output_size
        if self._hidden % 256 or self._intermediate % 128:
            raise ValueError(
                "FlashInfer FP4 requires hidden_size divisible by 256 and "
                "intermediate_size divisible by 128"
            )
        self._beta = activation_spec.beta
        self._linear_beta = activation_spec.linear_beta
        self._prepare_weights(items)

    def _prepare_weights(self, items):
        from flashinfer.fp4_quantization import block_scale_interleave
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        cache = {}
        w13_weights, w13_scales, w2_weights, w2_scales = [], [], [], []
        sources = []
        for index, expert in items:
            if index != self._start + len(sources):
                raise ValueError("FlashInfer expert IDs must be contiguous")
            ops = (expert.w1, expert.w3, expert.w2)
            expected = (
                (self._intermediate, self._hidden),
                (self._intermediate, self._hidden),
                (self._hidden, self._intermediate),
            )
            for op, (output_size, input_size) in zip(ops, expected):
                if (op._output_size, op._input_size) != (
                    output_size,
                    input_size,
                ) or op.bias is not None:
                    raise ValueError(
                        "FlashInfer expert projections must be homogeneous and bias-free"
                    )
            packed = tuple(op.weight_packed.detach().cpu() for op in ops)
            scales = tuple(op.weight_scale.detach().cpu() for op in ops)
            if any(op.weight_packed.device.type != "cuda" for op in ops):
                raise RuntimeError("FlashInfer FP4 expert weights must be on CUDA")
            if any(t.dtype != torch.uint8 for t in (*packed, *scales)):
                raise ValueError("FlashInfer FP4 expects uint8 packed weights and scales")
            # The model loader has already materialized every layer. Release this
            # layer's raw GPU bytes before allocating its transformed stacked copy.
            for op in ops:
                op.weight_packed = torch.empty(0, dtype=torch.uint8, device="cuda")
                op.weight_scale = torch.empty(0, dtype=torch.uint8, device="cuda")
            sources.append((packed, scales))
        torch.cuda.empty_cache()
        for packed, scales in sources:
            packed = tuple(t.cuda() for t in packed)
            scales = tuple(t.cuda() for t in scales)
            w13 = torch.cat((packed[1], packed[0]), dim=0)
            # Serialized scale bytes are already UE8M0; keep them bit-preserving.
            w13_scale = torch.cat((scales[1], scales[0]), dim=0).view(torch.float8_e8m0fnu)
            w2 = packed[2]
            w2_scale = scales[2].view(torch.float8_e8m0fnu)
            w13_perm = _maybe_get_cached_w3_w1_permute_indices(cache, w13, 128)
            w13_scale_perm = _maybe_get_cached_w3_w1_permute_indices(
                cache, w13_scale.view(torch.uint8), 128, num_elts_per_sf=16
            )
            w2_perm = get_w2_permute_indices_with_cache(cache, w2, 128)
            w2_scale_perm = get_w2_permute_indices_with_cache(
                cache, w2_scale.view(torch.uint8), 128, num_elts_per_sf=16
            )
            w13_weights.append(w13.view(torch.uint8)[w13_perm].contiguous())
            w13_scales.append(
                block_scale_interleave(
                    w13_scale.view(torch.uint8)[w13_scale_perm].contiguous()
                ).view(torch.float8_e4m3fn)
            )
            w2_weights.append(w2.view(torch.uint8)[w2_perm].contiguous())
            w2_scales.append(
                block_scale_interleave(w2_scale.view(torch.uint8)[w2_scale_perm].contiguous()).view(
                    torch.float8_e4m3fn
                )
            )
        self._w13 = torch.stack(w13_weights).contiguous()
        self._w13_scale = (
            torch.stack(w13_scales).reshape(self._experts, 2 * self._intermediate, -1).contiguous()
        )
        self._w2 = torch.stack(w2_weights).contiguous()
        self._w2_scale = (
            torch.stack(w2_scales).reshape(self._experts, self._hidden, -1).contiguous()
        )
        if self._activation == "situ":
            self._alpha = torch.full(
                (self._experts,), self._beta, dtype=torch.float32, device=self._w13.device
            )
            self._linear = torch.full(
                (self._experts,), self._linear_beta, dtype=torch.float32, device=self._w13.device
            )
        else:
            self._alpha = self._linear = None

    @staticmethod
    def _tune_tokens(tokens: int) -> int:
        return max(1, 1 << (tokens - 1).bit_length())

    @property
    def input_dtype(self) -> torch.dtype:
        # Preserve projection accumulators until MXFP8 quantization.
        return torch.float32 if self._precision == "mxfp8" else torch.bfloat16

    def supports_out(self, dtype: torch.dtype) -> bool:
        """The finalized FI executor writes a contiguous BF16 [tokens, hidden]."""
        return (
            dtype == torch.bfloat16
            and self._w2 is not None
            and self._w2.is_cuda
            and not self._w2.requires_grad
        )

    def supports_deferred_output(self, dtype: torch.dtype) -> bool:
        return (
            self._activation == "situ" and self._precision == "mxfp8" and self.supports_out(dtype)
        )

    def forward_deferred(self, backend, x, routing, *, out):
        if not self.supports_deferred_output(out.dtype):
            raise NotImplementedError("Deferred FI output requires SiTU/MXFP8 and BF16 output")
        self._validate_output(x, out)
        return self._run(*self.prepare_inputs(backend, x, routing), out=out, do_finalize=False)

    def _validate_output(self, x, out, *extra_sources) -> None:
        if out is None:
            return
        if not self.supports_out(out.dtype):
            raise NotImplementedError(
                "FlashInfer caller-owned output requires loaded CUDA weights and BF16"
            )
        if x.ndim != 2 or out.shape != (x.shape[0], self._hidden):
            raise ValueError("FlashInfer output must have the exact [tokens, hidden] shape")
        if out.device != x.device or out.device != self._w2.device or not out.is_contiguous():
            raise ValueError("FlashInfer output must be contiguous on the input and weight device")
        if x.requires_grad or out.requires_grad:
            raise ValueError("FlashInfer caller-owned output is inference-only")
        sources = (
            x,
            self._w13,
            self._w13_scale,
            self._w2,
            self._w2_scale,
            self._alpha,
            self._linear,
            *extra_sources,
        )
        if any(torch._C._overlaps(out, source) for source in sources if source is not None):
            raise ValueError(
                "FlashInfer output must use separate storage from inputs and parameters"
            )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[-1] != self._hidden:
            raise ValueError("FlashInfer FP4 expects [tokens, hidden] input")
        allowed = (
            (torch.bfloat16, torch.float32) if self._precision == "mxfp8" else (torch.bfloat16,)
        )
        if x.dtype not in allowed or not x.is_cuda:
            raise ValueError(f"FlashInfer FP4 requires CUDA inputs with dtype in {allowed}")

    def _prepare_routed_inputs_with_ids(self, x, routing):
        if self._precision != "mxfp8":
            return None
        self._validate_input(x)
        from ..routing import fused_route_quant

        prepared = fused_route_quant(x, routing)
        if prepared is None:
            return None
        _, ids, packed_routes, x_input, x_scale = prepared
        x_scale = x_scale.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
        return ids, (x_input, x_scale, packed_routes, routing.topk)

    def _prepare_routed_inputs(self, x, routing):
        prepared = self._prepare_routed_inputs_with_ids(x, routing)
        return None if prepared is None else prepared[1]

    def _prepare_inputs(self, x, weights, ids):
        self._validate_input(x)
        if ids.ndim != 2 or weights.shape != ids.shape:
            raise ValueError(
                "FlashInfer route IDs and weights must have the same [tokens, topk] shape"
            )
        if weights.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("FlashInfer route weights must be bfloat16 or float32")

        if self._precision == "mxfp8":
            from minisgl.kernel.experts.group_quant import per_token_group_quant

            x_input, x_scale = per_token_group_quant(x, group_size=32, scale_ue8m0=True)
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
        else:
            x_input, x_scale = x, None
        return (
            x_input,
            x_scale,
            (ids.to(torch.int32).contiguous(), weights.contiguous()),
            ids.shape[1],
        )

    def prepare_inputs(self, backend, x, routing):
        prepared = self._prepare_routed_inputs(x, routing)
        return (
            prepared if prepared is not None else self._prepare_inputs(x, *routing.select(backend))
        )

    def _validate_dispatch_payloads(self, ids, payloads):
        if self._precision != "mxfp8":
            raise NotImplementedError("Prepared token dispatch requires MXFP8 activations")
        if ids.ndim not in (2, 3) or ids.dtype != torch.int32 or ids.shape[-1] < 1:
            raise ValueError("Dispatch IDs must be int32 [tokens, topk] or [EP, tokens, topk]")
        if len(payloads) != 3:
            raise ValueError(
                "Prepared dispatch requires activation, linear scale and route payloads"
            )
        x, scale, carrier = payloads
        expected = (*ids.shape[:-1], self._hidden)
        if x.shape != expected or scale.shape != (*ids.shape[:-1], self._hidden // 32):
            raise ValueError("Prepared dispatch requires per-token linear UE8M0 scale bytes")
        if x.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e4m3fn:
            raise NotImplementedError("Dispatch requires MXFP8 and its unchanged UE8M0 byte view")
        if carrier.shape != ids.shape or carrier.dtype not in (
            torch.int32,
            torch.bfloat16,
            torch.float32,
        ):
            raise NotImplementedError("Dispatch requires packed int32 or original BF16/FP32 routes")
        if carrier.dtype == torch.int32 and self._global_experts > 32768:
            raise NotImplementedError("Packed dispatch expert IDs must fit signed 16 bits")
        for value in (ids, *payloads):
            if not value.is_cuda or value.device != x.device or not value.is_contiguous():
                raise ValueError("Dispatch payloads must be contiguous on the same CUDA device")
            if value.requires_grad:
                raise ValueError("Prepared token dispatch is inference-only")

    dispatch_output_dtype = torch.bfloat16

    @property
    def global_experts(self):
        return self._global_experts

    def dispatch_layouts(self, top_k):
        from ..dispatch import DispatchLayout

        if self._precision != "mxfp8":
            return ()
        # The small-batch fused router packs IDs and weights. Other routes
        # carry their original weights alongside the separately exchanged IDs.
        return tuple(
            DispatchLayout(
                self.global_experts,
                top_k,
                (
                    (torch.float8_e4m3fn, self._hidden),
                    (torch.float8_e4m3fn, self._hidden // 32),
                    (carrier, top_k),
                ),
                (self.dispatch_output_dtype, self._hidden),
            )
            for carrier in (torch.int32, torch.bfloat16, torch.float32)
        )

    def prepare_local(self, backend, routing_builder, context, *, dry_run=False):
        from .flashinfer_tuning import prepare_local

        return prepare_local(backend, self, routing_builder, context, dry_run=dry_run)

    def prepare_dispatch_inputs(self, backend, x, routing):
        """Prepare the full batch before any source-owner row slicing.

        Return independent global IDs and three per-token payloads. The route
        payload is either the original packed BF16 carrier or the original
        unpacked BF16/FP32 weights; its dtype identifies the existing native ABI.
        The scale is an opaque UE8M0 byte view, not a numeric FP8 conversion.
        Unsupported formats fail here, before a caller can begin dispatch.
        """
        if self._precision != "mxfp8":
            raise NotImplementedError("Prepared token dispatch requires MXFP8 activations")
        if x.shape[0] == 0:
            raise ValueError("An empty global batch must skip dispatch on every rank")
        routed = self._prepare_routed_inputs_with_ids(x, routing)
        if routed is None:
            prepared = self._prepare_inputs(x, *routing.select(backend))
            x_input, scale, (ids, carrier), _ = prepared
        else:
            ids, (x_input, scale, carrier, _) = routed
        payloads = (x_input, scale, carrier)
        self._validate_dispatch_payloads(ids, payloads)
        return ids, payloads

    def forward_dispatched(self, ids, payloads, *, out=None):
        """Compute local experts from fixed receive views without re-preparation.

        The dispatcher only sanitizes padding token slots in independent IDs.
        Preserve every real carrier, including foreign expert routes; the native
        local-expert range filter handles those. Combine must not weight again.
        """
        self._validate_dispatch_payloads(ids, payloads)
        if ids.ndim != 3 or ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError("Received dispatch IDs require nonempty [EP, tokens, topk] geometry")
        x, scale, carrier = payloads
        x = x.view(-1, self._hidden)
        scale = scale.view(-1, self._hidden // 32)
        ids = ids.view(-1, ids.shape[-1])
        carrier = carrier.view_as(ids)
        self._validate_output(x, out, scale, ids, carrier)
        if carrier.dtype == torch.int32:
            # Same padding encoding as the mature router: ID -1 and BF16 +0.
            routes = torch.where(ids == -1, -65536, carrier)
        else:
            # Keep the original route-weight dtype and bits on every real slot.
            routes = (ids, torch.where(ids == -1, 0, carrier))
        return self._run(x, scale, routes, ids.shape[-1], out=out)

    def try_forward_routed(self, x, routing, *, out=None):
        self._validate_output(x, out)
        prepared = self._prepare_routed_inputs(x, routing)
        return None if prepared is None else self._run(*prepared, out=out)

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, *, out=None
    ) -> torch.Tensor:
        self._validate_output(x, out, weights, ids)
        return self._run(*self._prepare_inputs(x, weights, ids), out=out)

    def _run(self, x_input, x_scale, routes, topk, *, out=None, do_finalize: bool = True):
        route_inputs = routes if isinstance(routes, tuple) else (routes,)
        self._validate_output(x_input, out, x_scale, *route_inputs)
        from flashinfer import trtllm_fp4_block_scale_routed_moe
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer.tllm_enums import RoutingMethodType, SfLayout

        output = (
            out
            if out is not None
            else torch.empty(
                (x_input.shape[0], self._hidden), dtype=torch.bfloat16, device=x_input.device
            )
        )
        result = trtllm_fp4_block_scale_routed_moe(
            # The SDK consumes this dtype directly, including in finalization.
            topk_ids=routes,
            routing_bias=None,
            hidden_states=x_input,
            hidden_states_scale=x_scale,
            hidden_states_scale_layout=(SfLayout.layout_linear if x_scale is not None else None),
            gemm1_weights=self._w13,
            gemm1_weights_scale=self._w13_scale,
            gemm1_bias=None,
            gemm1_alpha=self._alpha,
            gemm1_beta=self._linear,
            gemm1_clamp_limit=None,
            gemm2_weights=self._w2,
            gemm2_weights_scale=self._w2_scale,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=self._global_experts,
            top_k=topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self._intermediate,
            local_expert_offset=self._start,
            local_num_experts=self._experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.TopK.value,
            activation_type=(
                ActivationType.Situ.value
                if self._activation == "situ"
                else ActivationType.Swiglu.value
            ),
            tune_max_num_tokens=self._tune_tokens(x_input.shape[0]),
            output=output,
            do_finalize=do_finalize,
            # Official TRTLLM MoE policy (runner:50-56, EnvInt default8192).
            # FI additionally masks this hint by the actual device capability.
            enable_pdl=x_input.shape[0]
            <= int(os.getenv("SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS", "8192")),
        )
        if not do_finalize:
            from ..base import DeferredMoEOutput

            # FI returns (values, weights, expanded-to-permuted indices).
            # Keep native BF16/F32 routing weights and the explicit [N,H] output,
            # matching the official wrapper's finalized/deferred profile shapes.
            values, weights, indices = result
            return DeferredMoEOutput(values, indices, weights)
        # The finalized kernel publishes its result through the explicit destination.
        # Keep BF16 partials through the tensor-parallel reduction.
        return output
