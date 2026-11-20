from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm_ffi import Module

from functools import cache

import torch
from minisgl.kernel.native import load_jit, make_cpp_args
from minisgl.utils.arch import is_sm90_supported as is_arch_support_pdl


def _row_aligned(t: torch.Tensor, align: int) -> bool:
    """Base pointer and all non-unit strides aligned to ``align`` bytes."""
    if t.data_ptr() % align != 0:
        return False
    esize = t.element_size()
    return all(s * esize % align == 0 for s in t.stride()[:-1])


def _pick_num_warps(total_items: int) -> int:
    # Same GB300 heuristic as set_mla_kv_buffer: small grids favour more CTAs.
    return 4 if total_items <= 768 else 8


@cache
def set_mla_kv_concat_q_fp8_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "set_mla_kv_concat_q_fp8",
        *args,
        cuda_files=["elementwise/set_mla_kv_concat_q.cuh"],
        cuda_wrappers=[
            ("set_mla_kv_concat_q_fp8", f"SetMlaKVConcatQFp8Kernel<{args}>::run"),
        ],
    )


@cache
def can_use_set_mla_kv_concat_q_fp8() -> bool:
    """SM90+ (TMA bulk store) and the module compiles. Row widths are fixed
    at 512/64 (the MLA absorb layout) inside the kernel."""
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        set_mla_kv_concat_q_fp8_module(is_arch_support_pdl())
        return True
    except Exception:  # pragma: no cover - compile-time only
        return False


def covered_fp8(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
) -> bool:
    """Select the fused FP8 path for supported dtypes, widths and alignment.

    Expects flattened views: kv_buffer
    [pages, 576] fp8/uint8, k halves [B, 512/64] bf16, q halves
    [B, H, 512/64] bf16, loc [B]. The C++ launcher validates matching shapes.
    """
    if not (
        kv_buffer.dtype in (torch.float8_e4m3fn, torch.uint8)
        and k_nope.dtype == torch.bfloat16
        and k_rope.dtype == torch.bfloat16
        and q_nope.dtype == torch.bfloat16
        and q_rope.dtype == torch.bfloat16
    ):
        return False
    if loc.dtype not in (torch.int32, torch.int64):
        return False
    if not loc.is_contiguous():
        return False
    if k_nope.shape[-1] != 512 or k_rope.shape[-1] != 64:
        return False
    if not can_use_set_mla_kv_concat_q_fp8():
        return False
    if any(t.stride(-1) != 1 for t in (kv_buffer, k_nope, k_rope, q_nope, q_rope)):
        return False
    if kv_buffer.data_ptr() % 16 != 0 or kv_buffer.stride(0) % 16 != 0:
        return False
    return (
        _row_aligned(k_nope, 16)
        and _row_aligned(k_rope, 4)
        and _row_aligned(q_nope, 16)
        and _row_aligned(q_rope, 4)
    )


def set_mla_kv_concat_q_fp8(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    num_warps: int = 0,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
) -> torch.Tensor:
    """Quantize bf16 [k_nope | k_rope] rows to fp8-e4m3 and scatter them into
    ``kv_buffer`` at ``loc``, and return the fp8 concatenated query
    [q_nope | q_rope], all in one kernel launch (replaces concat + three
    aten fp8 casts + the KV-row write on the fp8 decode path).

    Under DCP, ``loc`` is VIRTUAL: the physical row is ``loc //
    dcp_world_size`` and only the owner rank (``loc % dcp_world_size ==
    dcp_rank``) writes its KV row (query conversion still runs for every
    token). world=1/rank=0 is the non-DCP identity.

    Shapes (leading singleton dims on the k sources are flattened away):
        kv_buffer:    [num_pages, 576] fp8_e4m3/uint8 (or [num_pages, 1, 576])
        loc:          [n_loc]
        cache_k_nope: [n_loc, 512] bf16   cache_k_rope: [n_loc, 64] bf16
        q_nope:       [n_loc, H, 512] bf16   q_rope: [n_loc, H, 64] bf16
    Returns:
        query: [n_loc, H, 576] float8_e4m3fn (new contiguous tensor)
    """
    n_loc = loc.shape[0]
    src_nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    src_rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer

    q_out = torch.empty(
        (q_nope.shape[0], q_nope.shape[1], 576),
        dtype=torch.float8_e4m3fn,
        device=q_nope.device,
    )
    if num_warps <= 0:
        num_warps = _pick_num_warps(n_loc + q_nope.shape[0] * q_nope.shape[1])

    module = set_mla_kv_concat_q_fp8_module(is_arch_support_pdl())
    module.set_mla_kv_concat_q_fp8(
        buf,
        loc,
        src_nope,
        src_rope,
        q_nope,
        q_rope,
        q_out,
        num_warps,
        dcp_world_size,
        dcp_rank,
    )
    return q_out
