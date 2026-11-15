"""Map generic paged storage to the mature MLA cache-write kernels."""

from functools import cache

import torch


@cache
def _kernels():
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.kvcache import mla_buffer
    from tvm_ffi import use_torch_stream

    return mla_buffer, is_arch_support_pdl(), use_torch_stream


def store_mla(latent, key, locations, pages, storage, *, page_size, cp_size, cp_rank):
    if not len(locations):
        return
    if storage.stride(-1) != 1:
        raise ValueError("MLA storage must have a contiguous payload dimension")
    owned = (locations >= 0) & (locations % cp_size == cp_rank)
    logical_pages = locations.clamp_min(0) // page_size
    physical_pages = logical_pages if pages is None else pages[logical_pages]
    # Physical pages may contain other layers or alignment padding. The SDK's
    # pointer/stride API accepts element offsets, so no layout-specific compute
    # kernel or contiguous staging cache is needed.
    offsets = physical_pages * storage.stride(0)
    offsets = offsets + ((locations % page_size) // cp_size) * storage.stride(1)
    offsets = torch.where(owned, offsets, -1).to(torch.int64)
    latent = latent.contiguous() if latent.stride(-1) != 1 else latent
    key = key.contiguous() if key.stride(-1) != 1 else key
    sdk, use_pdl, stream = _kernels()
    width = latent.shape[-1] + key.shape[-1]
    launch = {"USE_GDC": True, "launch_pdl": True} if use_pdl else {}
    arguments = (
        storage,
        latent,
        key,
        offsets,
        -1,
        1,
        latent.stride(0),
        key.stride(0),
        latent.shape[-1],
        key.shape[-1],
    )
    with stream():
        if storage.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            sdk.set_mla_kv_buffer_fp8_quant_kernel[(len(locations), (width + 127) // 128)](
                *arguments, BLOCK=128, **launch
            )
        else:
            sdk.set_mla_kv_buffer_kernel[(len(locations), 1)](
                *arguments,
                BLOCK=1 << (width - 1).bit_length(),
                DCP_RANK=0,
                DCP_WORLD_SIZE=1,
                **launch,
            )


@cache
def _concat_query_kernels():
    from sglang.kernels.ops.attention import set_mla_kv_concat_q
    from tvm_ffi import use_torch_stream

    return set_mla_kv_concat_q, use_torch_stream


def store_mla_and_concat_query(storage, locations, latent, key, query, suffix, *, cp_size, cp_rank):
    """One official FP8 KV-write/query-concat launch, or a pre-write decline.

    The SDK handles virtual cyclic locations and SATFINITE E4M3 conversion.
    No location remapping, temporary cache, or post-launch fallback is involved.
    """
    tensors = (storage, locations, latent, key, query, suffix)
    if (
        not all(value.is_cuda and value.device == storage.device for value in tensors)
        or storage.dtype != torch.float8_e4m3fn
        or storage.ndim != 2
        or storage.shape[1] != 576
        or locations.ndim != 1
        or query.ndim != 3
        or suffix.shape != (*query.shape[:2], 64)
        or query.shape != (len(locations), query.shape[1], 512)
    ):
        return None
    sdk, stream = _concat_query_kernels()
    if not sdk.covered_fp8(storage, locations, latent, key, query, suffix):
        return None
    with stream():
        return sdk.set_mla_kv_concat_q_fp8(
            storage,
            locations,
            latent,
            key,
            query,
            suffix,
            dcp_world_size=cp_size,
            dcp_rank=cp_rank,
        )
