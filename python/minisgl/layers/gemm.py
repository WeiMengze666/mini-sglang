"""Optional mature GEMM dispatch, independent of model and projection layout."""

from functools import cache

import torch
from minisgl.precision import use_strict_bf16


@cache
def _native_bf16_kernels():
    try:
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            cutedsl_bf16_gemm_out,
            use_cutedsl_bf16_gemm,
        )
    except ImportError:
        return None
    return cutedsl_bf16_gemm, cutedsl_bf16_gemm_out, use_cutedsl_bf16_gemm


def try_native_bf16_gemm(x, weight, bias=None, *, out=None):
    """Use the SDK's measured shape policy, or let the caller use PyTorch.

    Strict arithmetic keeps its existing reference path. Default arithmetic can
    use native SM10x kernels when the optional SDK and tensor layouts support it.
    The caller owns any BF16/FP32 output buffer; input storage is never repacked.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or x.ndim != 2
        or weight.ndim != 2
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.device != weight.device
        or x.shape[1] != weight.shape[1]
        or x.data_ptr() % 16
        or weight.data_ptr() % 16
        or use_strict_bf16()
        or torch.cuda.get_device_capability(x.device)[0] != 10
    ):
        return None
    shape = (x.shape[0], weight.shape[0])
    if bias is not None and (
        bias.device != x.device
        or bias.dtype != x.dtype
        or bias.shape != (shape[1],)
        or not bias.is_contiguous()
    ):
        return None
    if out is not None and (
        out.device != x.device
        or out.dtype not in (torch.bfloat16, torch.float32)
        or out.shape != shape
        or not out.is_contiguous()
    ):
        return None
    kernels = _native_bf16_kernels()
    if kernels is None:
        return None
    gemm, gemm_out, covered = kernels
    if not covered(x.shape[0], weight.shape[0], weight.shape[1]):
        return None
    return gemm(x, weight, bias) if out is None else gemm_out(x, weight, out, bias)
