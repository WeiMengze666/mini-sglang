"""Tensor parallel operations shared by eager layers and reference execution."""

from __future__ import annotations

import torch

from . import DistributedCommunicator, try_get_tp_info


def tp_info() -> tuple[int, int]:
    """Standalone CPU/reference use defaults to one rank."""
    info = try_get_tp_info()
    return (info.rank, info.size) if info is not None else (0, 1)


def local_size(size: int, tp_size: int, name: str) -> int:
    if tp_size < 1 or size % tp_size:
        raise ValueError(f"{name} ({size}) must be divisible by TP size ({tp_size})")
    return size // tp_size


class TensorParallel:
    """Capture the model's TP layout and use mini's selected TP communicator."""

    def __init__(self):
        self.rank, self.size = tp_info()
        self._comm = DistributedCommunicator()

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._comm.all_reduce(x) if self.size > 1 else x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self._comm.all_gather(x) if self.size > 1 else x
