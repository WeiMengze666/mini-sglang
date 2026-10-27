"""Contiguous logical partitions shared by layers and checkpoint readers."""

from dataclasses import dataclass

import torch

from .parallel import local_size


@dataclass(frozen=True)
class ShardSpec:
    axis: int
    start: int
    stop: int

    @property
    def length(self):
        return self.stop - self.start

    def slice(self, tensor: torch.Tensor):
        return tensor.narrow(self.axis, self.start, self.length)


@dataclass(frozen=True)
class Partition:
    rank: int
    size: int

    def __post_init__(self):
        if not 0 <= self.rank < self.size:
            raise ValueError("Partition rank must belong to its world")

    def range(self, size: int, name="partition"):
        width = local_size(size, self.size, name)
        return range(self.rank * width, (self.rank + 1) * width)

    def shard(self, axis: int, size: int, name="partition", block: int = 1):
        rows = self.range(size, name)
        return ShardSpec(axis, rows.start * block, rows.stop * block)

    def local_ids(self, ids: torch.Tensor, global_size: int):
        """Map global routing IDs to local IDs; the final ID denotes no work.

        Routing and normalization happen over all experts before this mapping.
        Fused kernels reserve one zero-output expert at the local expert count.
        """
        owned = self.range(global_size)
        local = ids - owned.start
        return local.masked_fill((local < 0) | (local >= len(owned)), len(owned))
