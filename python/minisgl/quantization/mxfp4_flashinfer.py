"""Opt-in FlashInfer TRTLLM FP4 grouped experts."""

from __future__ import annotations

from typing import Iterable

import torch


class FlashInferMXFP4Experts:
    """Grouped MXFP4 experts backed by FlashInfer's SM100 TRTLLM kernel.

    Weight preparation is intentionally eager and immutable after construction;
    forward only prepares routing/activation metadata and launches the fused op.
    """

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
        if activation not in ("silu", "situ"):
            raise ValueError("FlashInfer FP4 supports gated silu or situ")
        if activation_precision not in ("bf16", "mxfp8"):
            raise ValueError("FlashInfer activation precision must be bf16 or mxfp8")
        if activation == "situ" and activation_precision != "mxfp8":
            raise ValueError("FlashInfer SiTU requires mxfp8 activation precision")
        if not torch.cuda.is_available():
            raise RuntimeError("FlashInfer FP4 grouped experts require CUDA")
        items = list(experts)
        if not items:
            raise ValueError("FlashInfer FP4 requires at least one local expert")
        if num_global_experts < 1:
            raise ValueError("num_global_experts must be positive")
        self._activation = activation
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
        self._beta = float(beta)
        self._linear_beta = None if linear_beta is None else float(linear_beta)
        self._prepare_weights(items)

    def _prepare_weights(self, items):
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a
        from flashinfer.fp4_quantization import block_scale_interleave
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        cache = {}
        w13_weights, w13_scales, w2_weights, w2_scales = [], [], [], []
        for index, expert in items:
            if index != self._start + len(w13_weights):
                raise ValueError("FlashInfer expert IDs must be contiguous")
            packed = (expert.w1.weight_packed, expert.w3.weight_packed, expert.w2.weight_packed)
            scales = (expert.w1.weight_scale, expert.w3.weight_scale, expert.w2.weight_scale)
            if any(t.device.type != "cuda" for t in (*packed, *scales)):
                raise RuntimeError("FlashInfer FP4 expert weights must be on CUDA")
            if any(t.dtype != torch.uint8 for t in (*packed, *scales)):
                raise ValueError("FlashInfer FP4 expects uint8 packed weights and scales")
            w13 = torch.cat((packed[1], packed[0]), dim=0)
            w13_scale = torch.exp2(torch.cat((scales[1], scales[0]), dim=0).float() - 127)
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
            w2 = packed[2]
            w2_scale = torch.exp2(scales[2].float() - 127).to(torch.float8_e8m0fnu)
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
                block_scale_interleave(w13_scale.view(torch.uint8)[w13_scale_perm].contiguous())
                .view(torch.float8_e4m3fn)
            )
            w2_weights.append(w2.view(torch.uint8)[w2_perm].contiguous())
            w2_scales.append(
                block_scale_interleave(w2_scale.view(torch.uint8)[w2_scale_perm].contiguous())
                .view(torch.float8_e4m3fn)
            )
        self._w13 = torch.stack(w13_weights).contiguous()
        self._w13_scale = torch.stack(w13_scales).reshape(self._experts, 2 * self._intermediate, -1).contiguous()
        self._w2 = torch.stack(w2_weights).contiguous()
        self._w2_scale = torch.stack(w2_scales).reshape(self._experts, self._hidden, -1).contiguous()
        if self._activation == "situ":
            self._alpha = torch.full((self._experts,), self._beta, dtype=torch.float32, device=self._w13.device)
            self._linear = torch.full(
                (self._experts,), self._linear_beta, dtype=torch.float32, device=self._w13.device
            )
        else:
            self._alpha = self._linear = None

    @staticmethod
    def _tune_tokens(tokens: int) -> int:
        return max(1, 1 << (tokens - 1).bit_length())

    def forward(self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self._hidden:
            raise ValueError("FlashInfer FP4 expects [tokens, hidden] input")
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError("FlashInfer FP4 currently requires CUDA BF16 inputs")
        if ids.ndim != 2 or weights.shape != ids.shape:
            raise ValueError("FlashInfer route IDs and weights must have the same [tokens, topk] shape")
        from flashinfer import mxfp8_quantize, trtllm_fp4_block_scale_routed_moe
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer.tllm_enums import RoutingMethodType, SfLayout

        if self._precision == "mxfp8":
            x_input, x_scale = mxfp8_quantize(x, False, alignment=self._hidden, backend="cute-dsl")
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
        else:
            x_input, x_scale = x, None
        output = torch.empty((x.shape[0], self._hidden), dtype=torch.bfloat16, device=x.device)
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids.to(torch.int32).contiguous(), weights.float().contiguous()),
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
            top_k=ids.shape[1],
            n_group=None,
            topk_group=None,
            intermediate_size=self._intermediate,
            local_expert_offset=self._start,
            local_num_experts=self._experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.TopK.value,
            activation_type=(
                ActivationType.Situ.value if self._activation == "situ" else ActivationType.Swiglu.value
            ),
            tune_max_num_tokens=self._tune_tokens(x.shape[0]),
            output=output,
            do_finalize=True,
            enable_pdl=True,
        )
        return result[0].float() if isinstance(result, (tuple, list)) else result.float()
