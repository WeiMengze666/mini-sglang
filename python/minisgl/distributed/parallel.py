"""Tensor parallel operations shared by eager layers and reference execution."""

from __future__ import annotations

import torch

from .impl import DistributedCommunicator, _all_reduce_options
from .info import DistributedInfo, local_size, try_get_tp_info


def tp_info() -> tuple[int, int]:
    """Standalone CPU/reference use defaults to one rank."""
    info = try_get_tp_info()
    return (info.rank, info.size) if info is not None else (0, 1)


class TensorParallel:
    """Capture the model's TP layout and use mini's selected TP communicator."""

    def __init__(self):
        self.group = try_get_tp_info() or DistributedInfo(0, 1)
        self._comm = DistributedCommunicator()

    @property
    def rank(self):
        return self.group.rank

    @property
    def size(self):
        return self.group.size

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._comm.all_reduce(x) if self.size > 1 else x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self._comm.all_gather(x) if self.size > 1 else x

    def supports_linear_all_gather(self, x, weight, addend) -> bool:
        return self.size > 1 and self._comm.supports_linear_all_gather(x, weight, addend)

    def linear_all_gather(self, x, weight, addend) -> torch.Tensor:
        """Column GEMM + gather with full replicated operands; returns owned output."""
        if self.size == 1:
            raise NotImplementedError("Singleton linear keeps the ordinary Linear path")
        return self._comm.linear_all_gather(x, weight, addend)

    def reserve_output(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, *, enabled: bool = True
    ):
        """Prepare provider-owned output storage for multi-rank execution."""
        return (
            self._comm.reserve_output(key, shape, dtype, enabled=enabled) if self.size > 1 else None
        )

    def all_reduce_residual(
        self,
        x: torch.Tensor,
        residual=None,
        *,
        norm=None,
        output_key: str | None = None,
        overlap: bool = False,
    ) -> torch.Tensor:
        options = _all_reduce_options(norm, output_key, overlap)
        if self.size == 1:
            result = x
            if residual is not None:
                result = result + residual
            return norm(result) if norm is not None else result
        return self._comm.all_reduce_residual(x, residual, **options)

    def supports_weighted_all_reduce(self, out, *, norm, topk: int) -> bool:
        return self.size > 1 and self._comm.supports_weighted_all_reduce(out, norm=norm, topk=topk)

    def weighted_all_reduce(self, values, indices, weights, *, out, norm):
        if self.size == 1:
            raise NotImplementedError("Single-rank weighted finalization is not implemented")
        return self._comm.weighted_all_reduce(values, indices, weights, out=out, norm=norm)


__all__ = ["TensorParallel", "local_size", "tp_info"]
