"""Logical rank groups and contiguous tensor/expert partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .shard import ShardSpec

if TYPE_CHECKING:
    import torch


def local_size(size: int, group_size: int, name: str) -> int:
    if group_size < 1 or size % group_size:
        raise ValueError(f"{name} ({size}) must be divisible by group size ({group_size})")
    return size // group_size


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    size: int

    def __post_init__(self):
        if not 0 <= self.rank < self.size:
            raise ValueError("Distributed rank must belong to its group")

    def is_primary(self) -> bool:
        return self.rank == 0

    def range(self, size: int, name="partition"):
        width = local_size(size, self.size, name)
        return range(self.rank * width, (self.rank + 1) * width)

    def shard(self, axis: int, size: int, name="partition", block: int = 1):
        rows = self.range(size, name)
        return ShardSpec(axis, rows.start * block, rows.stop * block)

    def local_ids(self, ids: torch.Tensor, global_size: int):
        """Map global routed IDs to local IDs; -1 denotes a remote route."""
        owned = self.range(global_size)
        local = ids - owned.start
        return local.masked_fill((local < 0) | (local >= len(owned)), -1)


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info", "local_size"]
