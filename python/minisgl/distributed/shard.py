"""Logical tensor slices, independent of rank-group topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


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
