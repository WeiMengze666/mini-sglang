# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    """Number of output-dtype elements needed to store one fp32 LSE value."""
    return torch.finfo(torch.float32).bits // torch.finfo(output_dtype).bits


@triton.jit
def _dcp_pack_a2a_send_kernel(
    out_ptr,
    lse_ptr,
    dst_o_ptr,
    dst_lse_ptr,
    out_stride_B,
    out_stride_H,
    lse_stride_B,
    lse_stride_H,
    dst_o_stride_N,
    dst_o_stride_B,
    dst_o_stride_H,
    dst_lse_stride_N,
    dst_lse_stride_B,
    dst_lse_stride_H,
    H_PER_RANK: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter one (batch, head) partial into its peer's send slot."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    peer = h // H_PER_RANK
    h_local = h % H_PER_RANK

    src = out_ptr + b * out_stride_B + h * out_stride_H
    dst = dst_o_ptr + peer * dst_o_stride_N + b * dst_o_stride_B + h_local * dst_o_stride_H

    for start in tl.range(0, WORDS, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < WORDS
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    tl.store(
        dst_lse_ptr + peer * dst_lse_stride_N + b * dst_lse_stride_B + h_local * dst_lse_stride_H,
        tl.load(lse_ptr + b * lse_stride_B + h * lse_stride_H),
    )


def dcp_pack_a2a_send(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    dst_o: torch.Tensor,
    dst_lse: torch.Tensor,
) -> None:
    """Scatter ``[B, H, D]`` partials + ``[B, H]`` LSE into a transport's send slots.

    ``dst_o`` is ``[N, B_max, H // N, D]`` and ``dst_lse`` ``[N, B_max, H // N]``,
    in any stride order. Rows beyond ``B`` are left untouched.
    """
    B, H, D = cp_attn_out.shape
    N, B_max, H_per_rank = dst_lse.shape
    lpd = _lse_pack_dim(cp_attn_out.dtype)

    if cp_attn_lse.dtype != torch.float32 or dst_lse.dtype != torch.float32:
        raise ValueError("LSE tensors must be float32")
    if D % lpd:
        raise ValueError(f"head dim {D} must be a multiple of the LSE pack dim {lpd}")
    if dst_o.shape != (N, B_max, H_per_rank, D) or H_per_rank * N != H or B > B_max:
        raise ValueError(
            f"destination {tuple(dst_o.shape)} / {tuple(dst_lse.shape)} does not "
            f"match out {tuple(cp_attn_out.shape)}"
        )

    out_words = cp_attn_out.view(torch.float32)
    dst_o_words = dst_o.view(torch.float32)
    words = D // lpd

    _dcp_pack_a2a_send_kernel[(B, H)](
        out_words,
        cp_attn_lse,
        dst_o_words,
        dst_lse,
        out_words.stride(0),
        out_words.stride(1),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        dst_o_words.stride(0),
        dst_o_words.stride(1),
        dst_o_words.stride(2),
        dst_lse.stride(0),
        dst_lse.stride(1),
        dst_lse.stride(2),
        H_PER_RANK=H_per_rank,
        WORDS=words,
        BLOCK=min(1024, triton.next_power_of_2(words)),
    )


@triton.jit
def _dcp_lse_combine_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    recv_output_stride_N,
    recv_output_stride_B,
    recv_output_stride_H,
    recv_output_stride_D,
    recv_lse_stride_N,
    recv_lse_stride_B,
    recv_lse_stride_H,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Combine N partial attention outputs weighted by their LSE values.

    Grid: (B, H_local).
    Each program handles one (batch, head) position across all N shards.

    Two-pass approach:
    Pass 1: find max LSE and weight sum across shards
    Pass 2: accumulate weighted outputs
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_base = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H

    # Pass 1: find max LSE across N shards
    lse_max = tl.load(recv_lse_ptr + lse_base).to(tl.float32)
    lse_max = tl.where((lse_max != lse_max) | (lse_max == float("inf")), -float("inf"), lse_max)
    for i in tl.static_range(1, N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where((lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i)
        lse_max = tl.where(lse_i > lse_max, lse_i, lse_max)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Pass 2: accumulate weighted outputs
    weight_sum = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in tl.static_range(N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where((lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i)
        centered = lse_i - lse_max
        if IS_BASE_E:
            w = tl.exp(centered)
        else:
            w = tl.exp2(centered)
        weight_sum += w

        o_offsets = (
            i * recv_output_stride_N
            + batch_idx * recv_output_stride_B
            + head_idx * recv_output_stride_H
            + d_offsets * recv_output_stride_D
        )
        partial_out = tl.load(recv_output_ptr + o_offsets).to(tl.float32)
        acc += partial_out * w

    acc = acc / tl.maximum(weight_sum, 1.0)

    out_offsets = batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty))

    if RETURN_LSE:
        if IS_BASE_E:
            global_lse = tl.log(weight_sum) + lse_max
        else:
            global_lse = tl.log2(weight_sum) + lse_max
        out_lse_offset = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    is_lse_base_on_e: bool = True,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the Triton LSE-combine kernel.

    Args:
        recv_output: [N, B, H_local, D] partial outputs from each DCP rank.
        recv_lse:    [N, B, H_local]    log-sum-exp from each DCP rank.
        is_lse_base_on_e: True if LSE uses base-e (FlashAttention),
                          False if base-2 (FlashInfer).
        return_lse: If True, also return the combined global LSE.

    Returns:
        (combined_output [B, H_local, D], combined_lse [B, H_local] or None)
    """
    N, B, H_local, D = recv_output.shape
    out = torch.empty((B, H_local, D), device=recv_output.device, dtype=recv_output.dtype)
    out_lse = (
        torch.empty((B, H_local), device=recv_lse.device, dtype=recv_lse.dtype)
        if return_lse
        else recv_lse.new_empty(0)
    )

    grid = (B, H_local)
    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        recv_output.stride(0),
        recv_output.stride(1),
        recv_output.stride(2),
        recv_output.stride(3),
        recv_lse.stride(0),
        recv_lse.stride(1),
        recv_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )
    return out, (out_lse if return_lse else None)
