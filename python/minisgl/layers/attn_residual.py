"""Per-token attention over a bank of frozen block sums."""

from functools import cache
from types import SimpleNamespace

import torch
import torch.nn.functional as F


class ResidualBank:
    """Own frozen block sums in the row layout consumed by residual kernels.

    One bank belongs to one forward. Only the appended prefix is valid; the
    unused capacity is never read. Appending copies the raw sum, so subsequent
    updates of the caller's stream cannot mutate an earlier block.
    """

    def __init__(self, prototype: torch.Tensor, capacity: int):
        if prototype.ndim != 2 or capacity < 1:
            raise ValueError("Residual bank requires [tokens, hidden] and positive capacity")
        self._storage = prototype.new_empty((len(prototype), capacity, prototype.shape[-1]))
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        for index in range(self._size):
            yield self._storage[:, index]

    def append(self, value: torch.Tensor) -> None:
        if self._size == self._storage.shape[1]:
            raise ValueError("Residual bank capacity exhausted")
        if (
            value.shape != (self._storage.shape[0], self._storage.shape[2])
            or value.dtype != self._storage.dtype
            or value.device != self._storage.device
        ):
            raise ValueError("Residual snapshot must preserve shape, dtype and device")
        self._storage[:, self._size].copy_(value)
        self._size += 1


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


def _read_native(head, rows, norm, proj, output_norm, *, valid_blocks=None):
    aggregate, mix, stream = _native_kernels()
    count = rows.shape[1] - 1 if valid_blocks is None else valid_blocks
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
            return mix(head, rows, count, score, _NormView(norm))
        return aggregate(head, rows, count, score, _NormView(norm), _NormView(output_norm))


def attention_residual(head, bank, norm, proj, output_norm=None):
    """Read a residual bank, optionally retaining FP32 through output RMSNorm.

    The normalized path folds the scoring weights in the input dtype, then
    keeps scores and the mixture in FP32 until the final output conversion.
    """
    if not bank:
        return head if output_norm is None else output_norm.forward(head)
    # The SDK score buffers and fused kernel cover up to eight frozen rows.
    if (
        head.is_cuda
        and len(bank) <= 8
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
        rows = (
            bank._storage if isinstance(bank, ResidualBank) else torch.stack([*bank, head], dim=1)
        )
        return _read_native(head, rows, norm, proj, output_norm, valid_blocks=len(bank))
    rows = torch.stack([*bank, head], dim=1)
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
