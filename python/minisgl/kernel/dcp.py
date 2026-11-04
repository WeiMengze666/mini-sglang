"""Adapt the native DCP softmax combine kernel; no local numerical kernel."""

from functools import cache

import torch


@cache
def _native_lse_combine():
    try:
        from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    except ImportError:
        return None
    return dcp_lse_combine_triton


def combine_partial_attention(output: torch.Tensor, lse: torch.Tensor):
    """Try native FP32 combination while preserving empty/corrupt-row semantics."""
    if (
        not output.is_cuda
        or output.ndim != 4
        or lse.shape != output.shape[:-1]
        or lse.dtype != torch.float32
        or output.shape[-1] < 1
        or output.shape[-1] > 1024
        or output.shape[-1] & (output.shape[-1] - 1)
    ):
        return None
    combine = _native_lse_combine()
    if combine is None:
        return None
    # Empty shards can contain undefined outputs. The native kernel consumes
    # those values even at zero weight, so supply zeros for empty rows.
    safe = output.float().masked_fill(torch.isneginf(lse)[..., None], 0)
    combined, _ = combine(safe, lse, is_lse_base_on_e=True)
    # Native sanitization must not hide a corrupt active producer from callers.
    invalid = (torch.isnan(lse) | torch.isposinf(lse)).any(0)
    return combined.masked_fill(invalid[..., None], torch.nan)
