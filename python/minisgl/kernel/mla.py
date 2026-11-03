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
