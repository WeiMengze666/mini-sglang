"""Physical cache layout, independent of checkpoint names and model families."""

from dataclasses import dataclass

import torch
from minisgl.distributed.parallel import local_size, tp_info


@dataclass(frozen=True)
class HybridCacheConfig:
    num_layers: int
    mla_layers: tuple[int, ...]
    recurrent_layers: tuple[int, ...]
    latent_dim: int
    key_dim: int
    num_heads: int
    head_dim: int
    conv_kernel_size: int

    def __post_init__(self):
        if sorted(self.mla_layers + self.recurrent_layers) != list(range(self.num_layers)):
            raise ValueError("Cache layer IDs must partition all layers exactly once")
        if (
            min(self.latent_dim, self.num_heads, self.head_dim, self.conv_kernel_size) < 1
            or self.key_dim < 0
        ):
            raise ValueError("Invalid hybrid cache dimensions")

    def cache_bytes_per_token(self, dtype: torch.dtype) -> int:
        return len(self.mla_layers) * (self.latent_dim + self.key_dim) * dtype.itemsize

    def state_bytes_per_request(self, dtype: torch.dtype, tp_size: int | None = None) -> int:
        if not self.recurrent_layers:
            return 0
        heads = local_size(
            self.num_heads, tp_info()[1] if tp_size is None else tp_size, "recurrent heads"
        )
        dim = self.head_dim
        return len(self.recurrent_layers) * (
            heads * dim * dim * 4 + 3 * heads * dim * (self.conv_kernel_size - 1) * dtype.itemsize
        )
