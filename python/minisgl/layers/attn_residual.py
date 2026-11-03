"""Per-token attention over a bank of frozen block sums."""

from functools import cache
from types import SimpleNamespace

import torch
import torch.nn.functional as F


class _NormView:
    def __init__(self, norm):
        self.weight = norm.weight
        self.variance_epsilon = norm.eps
        self._forward = norm.forward

    def __call__(self, x):
        return self._forward(x)


@cache
def _native_kernels():
    from sglang.srt.layers.attn_residual import _aggregate, aggregate_stream
    from tvm_ffi import use_torch_stream

    return _aggregate, aggregate_stream, use_torch_stream


def _read_native(head, rows, norm, proj, output_norm):
    aggregate, mix, stream = _native_kernels()
    # SDK metadata stays on a parameter view, without changing Mini's layer
    # interfaces. Weight replacement invalidates all derived score weights.
    cached = getattr(proj, "_residual_kernel_weights", None)
    if cached is None or cached[0] is not proj.weight or cached[1] is not norm.weight:
        cached = (proj.weight, norm.weight, SimpleNamespace(weight=proj.weight))
        proj._residual_kernel_weights = cached
    score = cached[2]
    with stream():
        head = head.contiguous()
        if output_norm is None:
            return mix(head, rows, rows.shape[1] - 1, score, _NormView(norm))
        return aggregate(
            head, rows, rows.shape[1] - 1, score, _NormView(norm), _NormView(output_norm)
        )


def attention_residual(head, bank, norm, proj, output_norm=None):
    """Read a residual bank, optionally retaining FP32 through output RMSNorm.

    The normalized path folds the scoring weights in the input dtype, then
    keeps scores and the mixture in FP32 until the final output conversion.
    """
    if not bank:
        return head if output_norm is None else output_norm.forward(head)
    rows = torch.stack([*bank, head], dim=1)
    if (
        head.is_cuda
        and head.dtype == norm.weight.dtype == proj.weight.dtype == torch.bfloat16
        and norm.weight.device == proj.weight.device == head.device
        and norm.weight.is_contiguous()
        and proj.weight.is_contiguous()
        and head.ndim == 2
        and head.shape[-1] % 1024 == 0
        and proj.weight.shape == (1, head.shape[-1])
        and (
            output_norm is None
            or (
                output_norm.weight.device == head.device
                and output_norm.weight.dtype == head.dtype
                and output_norm.weight.shape == (head.shape[-1],)
                and output_norm.weight.is_contiguous()
                and output_norm.eps == norm.eps
            )
        )
    ):
        return _read_native(head, rows, norm, proj, output_norm)
    # Keep the score and mixture reductions in FP32.
    xf = rows.float()
    inverse = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    if output_norm is None:
        scores = F.linear((xf * inverse) * norm.weight.float(), proj.weight.float())
    else:
        query = (norm.weight.float() * proj.weight.float()).to(head.dtype).float()
        scores = (xf * query).sum(-1, keepdim=True) * inverse
    mixed = (scores.softmax(dim=1) * xf).sum(dim=1)
    if output_norm is not None:
        mixed = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + output_norm.eps)
        mixed = mixed * output_norm.weight.float()
    return mixed.to(head.dtype)
