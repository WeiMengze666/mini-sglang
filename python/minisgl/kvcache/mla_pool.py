"""Paged compressed MLA latent and shared-key storage."""

from __future__ import annotations

import torch
from minisgl.attention.mla import MLAPagedMetadata
from minisgl.distributed.context import ContextParallel


def _check(condition, message, error=ValueError):
    if condition.is_cuda:
        # Keep invariant checks on the forward stream, without a CPU readback.
        torch._assert_async(condition, message)
    elif not condition:
        raise error(message)


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
        _check(
            (pages >= 0).all(),
            "MLA page must be allocated before attention reads or writes it",
            RuntimeError,
        )
        return self._page_storage(), pages, (locations % self.page_size) // self._cp.size

    def _page_storage(self):
        width = self.latent_dim + self.rope_dim
        # Padding at the end of each physical byte page preserves alignment.
        storage = self.arena.page_storage(self.dtype)
        storage = storage[:, : len(self._full_layers) * self._local_page_size * width]
        storage = storage.view(
            len(self.arena.page_map), len(self._full_layers), self._local_page_size, width
        )
        return storage

    def paged_layer(self, layer_id):
        """A strided view over physical pages; never gather the token history."""
        layer = self._full_layers[layer_id]
        if self.arena is not None:
            data = self._page_storage()[:, layer]
        else:
            data = self._mla[layer].view(
                self.num_tokens // self.page_size,
                self._local_page_size,
                self.latent_dim + self.rope_dim,
            )
        return data[..., : self.latent_dim], data[..., self.latent_dim :]

    def decode_metadata(self, table_indices, sequence_lengths, page_table):
        """Translate logical pages once, after any arena allocation/compaction."""
        lengths = [
            max(0, (length - self._cp.rank + self._cp.size - 1) // self._cp.size)
            for length in sequence_lengths
        ]
        counts = [
            (length + self._local_page_size - 1) // self._local_page_size for length in lengths
        ]
        pages = (
            torch.cat(
                [
                    page_table[row, : count * self.page_size : self.page_size]
                    for row, count in zip(table_indices, counts)
                ]
            ).long()
            // self.page_size
        )
        if self.arena is not None:
            pages = self.arena.page_map[pages]
            _check((pages >= 0).all(), "MLA decode must reference allocated pages", RuntimeError)
        return MLAPagedMetadata(
            torch.tensor([0] + counts, dtype=torch.int32).cumsum(0, dtype=torch.int32),
            pages.to(torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            self._local_page_size,
            torch.tensor(lengths, device=self.device) > 0 if 0 in lengths else None,
        )

    def _locations(self, locations: torch.Tensor) -> torch.Tensor:
        if locations.ndim != 1 or locations.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cache locations must be a one-dimensional integer tensor")
        _check(
            ((locations >= 0) & (locations < self.num_tokens)).all(),
            "MLA cache location is out of range",
            IndexError,
        )
        return locations.to(device=self.device, dtype=torch.long)

    def store_mla(
        self,
        layer_id: int,
        latent: torch.Tensor,
        key_suffix: torch.Tensor,
        out_loc: torch.Tensor,
        *,
        implementation="torch",
    ) -> None:
        locations = self._locations(out_loc)
        if latent.shape != (len(locations), self.latent_dim) or key_suffix.shape != (
            len(locations),
            self.rope_dim,
        ):
            raise ValueError("MLA token/cache shapes differ; do not write padded tokens")
        if implementation == "triton" and latent.is_cuda:
            from .mla_kernel import store_mla

            if self.arena is not None:
                _check(
                    (self.arena.page_map[locations // self.page_size] >= 0).all(),
                    "MLA writes must reference allocated pages",
                    RuntimeError,
                )
            store_mla(
                latent,
                key_suffix,
                locations,
                self.arena.page_map if self.arena is not None else None,
                self.paged_layer(layer_id)[0],
                page_size=self.page_size,
                cp_size=self._cp.size,
                cp_rank=self._cp.rank,
            )
            return
        if self._cp.size > 1:
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
        _check(
            self._cp.owns(locations).all(),
            "MLA reads must select only this DCP rank's token positions",
        )
        if self.arena is not None:
            storage, pages, offsets = self._paged_locations(locations)
            data = storage[pages, self._full_layers[layer_id], offsets]
        else:
            data = self._mla[self._full_layers[layer_id]].index_select(
                0, locations // self._cp.size
            )
        return data[:, : self.latent_dim], data[:, self.latent_dim :]
