"""Paged compressed MLA latent and shared-key storage."""

from __future__ import annotations

import torch
from minisgl.distributed.context import ContextParallel


class MLACache:
    def __init__(
        self, layer_ids, num_tokens, latent_dim, key_dim, dtype, device, arena=None, page_size=1
    ):
        self._full_layers = {layer: i for i, layer in enumerate(layer_ids)}
        self.num_tokens, self.latent_dim, self.rope_dim = num_tokens, latent_dim, key_dim
        self.arena, self.page_size = arena, page_size
        self._cp = ContextParallel()
        if page_size % self._cp.size:
            raise ValueError("MLA page size must be divisible by DCP size")
        self._local_page_size = page_size // self._cp.size
        self.dtype, self._device = dtype, torch.device(device)
        self._mla = (
            None
            if arena is not None
            else torch.zeros(
                (len(layer_ids), num_tokens // self._cp.size, latent_dim + key_dim),
                dtype=dtype,
                device=device,
            )
        )

    @property
    def device(self):
        return self._device

    def _paged_locations(self, locations):
        pages = self.arena.page_map[locations // self.page_size]
        if (pages < 0).any():
            raise RuntimeError("MLA page must be allocated before attention reads or writes it")
        width = self.latent_dim + self.rope_dim
        # Padding at the end of each physical byte page preserves alignment.
        storage = self.arena.page_storage(self.dtype)
        storage = storage[:, : len(self._full_layers) * self._local_page_size * width]
        storage = storage.view(
            len(self.arena.page_map), len(self._full_layers), self._local_page_size, width
        )
        return storage, pages, (locations % self.page_size) // self._cp.size

    def _locations(self, locations: torch.Tensor) -> torch.Tensor:
        if locations.ndim != 1 or locations.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cache locations must be a one-dimensional integer tensor")
        if locations.numel() and (
            locations.min().item() < 0 or locations.max().item() >= self.num_tokens
        ):
            raise IndexError("MLA cache location is out of range")
        return locations.to(device=self.device, dtype=torch.long)

    def store_mla(
        self, layer_id: int, latent: torch.Tensor, key_suffix: torch.Tensor, out_loc: torch.Tensor
    ) -> None:
        locations = self._locations(out_loc)
        if latent.shape != (len(locations), self.latent_dim) or key_suffix.shape != (
            len(locations),
            self.rope_dim,
        ):
            raise ValueError("MLA token/cache shapes differ; do not write padded tokens")
        owned = self._cp.owns(locations)
        locations, latent, key_suffix = locations[owned], latent[owned], key_suffix[owned]
        if self.arena is not None:
            storage, pages, offsets = self._paged_locations(locations)
            storage[pages, self._full_layers[layer_id], offsets] = torch.cat(
                (latent, key_suffix), -1
            ).to(self.dtype)
            return
        self._mla[self._full_layers[layer_id]].index_copy_(
            0, locations // self._cp.size, torch.cat((latent, key_suffix), dim=-1).to(self._mla)
        )

    def gather_mla(self, layer_id: int, locations: torch.Tensor):
        locations = self._locations(locations)
        if not self._cp.owns(locations).all():
            raise ValueError("MLA reads must select only this DCP rank's token positions")
        if self.arena is not None:
            storage, pages, offsets = self._paged_locations(locations)
            data = storage[pages, self._full_layers[layer_id], offsets]
        else:
            data = self._mla[self._full_layers[layer_id]].index_select(
                0, locations // self._cp.size
            )
        return data[:, : self.latent_dim], data[:, self.latent_dim :]
