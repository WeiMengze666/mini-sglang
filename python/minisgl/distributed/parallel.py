"""Tensor parallel operations shared by eager layers and reference execution."""

from __future__ import annotations

import torch

from .impl import DistributedCommunicator
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

    def all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> torch.Tensor:
        if self.size == 1:
            result = x
            if residual is not None:
                result = result + residual
            return norm(result) if norm is not None else result
        return self._comm.all_reduce_residual(x, residual, norm=norm)


__all__ = ["TensorParallel", "local_size", "tp_info"]
