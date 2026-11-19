from dataclasses import dataclass
from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class TorchRMSNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.weight = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == self.weight.dtype == torch.bfloat16
            and self.weight.device == x.device
            and self.weight.is_contiguous()
            and x.shape[-1] > 0
        ):
            from .norm_kernel import rms_norm

            # Native BF16 normalization retains FP32 through the scale multiply
            # and rounds once at the output.
            return rms_norm(x, self.weight, self.eps, round_before_weight=False)
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        if x.dtype == self.weight.dtype == torch.bfloat16:
            # Match the native BF16 kernel: round after applying the scale.
            return (normalized * self.weight.float()).to(x.dtype)
        return normalized.to(x.dtype) * self.weight


@dataclass(frozen=True)
class RMSNormEpilogue:
    """Normalize only the leading rows of borrowed packed reduction storage.

    The original normalizer owns the fallback arithmetic. This same-dtype
    in-place epilogue leaves every trailing element unchanged; casts required
    by an allocating/FP32 expert path stay with the MoE backend.
    """

    normalizer: TorchRMSNorm
    num_norm_rows: int

    @property
    def weight(self) -> torch.Tensor:
        return self.normalizer.weight

    @property
    def eps(self) -> float:
        return self.normalizer.eps

    @property
    def width(self) -> int:
        return self.weight.numel()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if type(self.num_norm_rows) is not int or self.num_norm_rows < 0 or self.width < 1:
            raise ValueError("RMSNorm epilogue requires nonnegative rows and positive width")
        count = self.num_norm_rows * self.width
        if not x.is_contiguous() or count > x.numel():
            raise ValueError("RMSNorm prefix must fit contiguous packed storage")
        if x.dtype != self.weight.dtype:
            raise ValueError("Packed RMSNorm epilogue requires matching storage and weight dtype")
        if torch._C._overlaps(x, self.weight):
            raise ValueError("Packed RMSNorm output must not overlap normalization weights")
        if count:
            prefix = x.view(-1)[:count].view(self.num_norm_rows, self.width)
            prefix.copy_(self.normalizer.forward(prefix))
        return x
