"""Adapt the mature DCP combine kernel without changing producer precision."""

from functools import cache

import torch


@cache
def _native_lse_combine():
    try:
        from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    except ImportError:
        return None
    return dcp_lse_combine_triton


def combine_partial_attention(output: torch.Tensor, lse: torch.Tensor, *, is_lse_base_on_e=True):
    """Combine neutral partials in the input dtype, or decline before selection.

    Empty shards must supply zero O and -inf LSE. The SDK accumulates in FP32,
    treats NaN/+inf LSE as -inf, and still multiplies every O by its weight.
    """
    if (
        not output.is_cuda
        or not lse.is_cuda
        or output.device != lse.device
        or output.layout != torch.strided
        or lse.layout != torch.strided
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or output.ndim != 4
        or lse.shape != output.shape[:-1]
        or lse.dtype != torch.float32
        or min(output.shape) < 1
        or output.shape[-1] > 1024
        or output.shape[-1] & (output.shape[-1] - 1)
        or any(value.requires_grad or value.is_conj() or value.is_neg() for value in (output, lse))
    ):
        return None
    combine = _native_lse_combine()
    if combine is None:
        return None
    # The SDK receives all four O strides, including the channel stride.
    combined, _ = combine(output, lse, is_lse_base_on_e=is_lse_base_on_e)
    return combined
