"""Paged compressed MLA latent and shared-key storage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.attention.mla import MLAPagedMetadata
from minisgl.distributed.context import ContextParallel

if TYPE_CHECKING:
    from .hybrid_pool import HybridCache


@dataclass(frozen=True)
class MLACacheWrite:
    """One forward's projected KV rows; an accepted decode consumes this write.

    A decoder returns None only before consuming it. The caller then owns the
    ordinary store. This value never persists in graph or cache metadata.
    """

    cache: "HybridCache"
    layer_id: int
    latent: torch.Tensor
    key: torch.Tensor
    locations: torch.Tensor
    implementation: str

    def store(self):
        cache = self.cache.mla
        cache._validate_write_locations(self.locations)
        cache._store_mla(
            self.layer_id,
            self.latent,
            self.key,
            self.locations,
            implementation=self.implementation,
        )

    def fused_query(self, query, suffix):
        if self.implementation != "sgl":
            return None
        cache = self.cache.mla
        return cache._store_and_concat_query(
            self.layer_id,
            self.latent,
            self.key,
            self.locations,
            query,
            suffix,
            validate=cache._validate_write_locations,
        )


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
        self._write_locations = None
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

    def paged_storage(self, layer_id):
        """Combined payload view; page gaps and layer offsets retain their strides."""
        layer = self._full_layers[layer_id]
        if self.arena is not None:
            data = self._page_storage()[:, layer]
        else:
            data = self._mla[layer].view(
                self.num_tokens // self.page_size,
                self._local_page_size,
                self.latent_dim + self.rope_dim,
            )
        return data

    def paged_layer(self, layer_id):
        """Library views over latent/key payloads, without gathering history."""
        data = self.paged_storage(layer_id)
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
        # Resolve the dense physical table once; both library layouts share it.
        # Dense decode kernels require a valid table width even on ranks with
        # no local tokens. Length zero prevents reads from this dummy entry.
        max_pages = max(1, max(counts, default=0))
        blocks = page_table[table_indices, : max_pages * self.page_size : self.page_size]
        blocks = blocks.long() // self.page_size
        valid = (
            torch.arange(max_pages, device=self.device)[None, :]
            < torch.tensor(counts, device=self.device)[:, None]
        )
        if self.arena is not None:
            blocks = self.arena.page_map[blocks]
            _check(
                ((blocks >= 0) | ~valid).all(),
                "MLA decode must reference allocated pages",
                RuntimeError,
            )
        blocks = blocks.masked_fill(~valid, 0).to(torch.int32)
        local_lengths = torch.tensor(lengths, device=self.device, dtype=torch.int32)
        return MLAPagedMetadata(
            torch.tensor([0] + counts, dtype=torch.int32).cumsum(0, dtype=torch.int32),
            torch.cat([blocks[row, :count] for row, count in enumerate(counts)]),
            torch.tensor(lengths, dtype=torch.int32),
            self._local_page_size,
            local_lengths > 0 if 0 in lengths else None,
            block_table=blocks,
            local_lengths=local_lengths,
            global_lengths=torch.tensor(sequence_lengths, device=self.device, dtype=torch.int32),
            max_seq_len=max(sequence_lengths, default=0),
            cp_size=self._cp.size,
            cp_rank=self._cp.rank,
        )

    def _validate_locations(self, locations: torch.Tensor) -> None:
        if locations.ndim != 1 or locations.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cache locations must be a one-dimensional integer tensor")
        _check(
            ((locations >= 0) & (locations < self.num_tokens)).all(),
            "MLA cache location is out of range",
            IndexError,
        )

    @contextmanager
    def _write_scope(self, locations):
        """Borrow read-only locations for one invocation, never across graph replay."""
        if self._write_locations is not None:
            raise RuntimeError("Nested MLA write scopes are not allowed")
        try:
            self._validate_locations(locations)
            self._write_locations = locations
            yield
        finally:
            self._write_locations = None

    def _validate_write_locations(self, locations):
        validated = self._write_locations
        if validated is not None and (
            not (locations.is_neg() or locations.is_conj())
            and not (validated.is_neg() or validated.is_conj())
            and locations.device == validated.device
            and locations.dtype == validated.dtype
            and locations.shape == validated.shape
            and locations.stride() == validated.stride()
            and locations.storage_offset() == validated.storage_offset()
            and locations.untyped_storage() is validated.untyped_storage()
        ):
            return
        self._validate_locations(locations)

    def _locations(self, locations: torch.Tensor) -> torch.Tensor:
        self._validate_locations(locations)
        return locations.to(device=self.device, dtype=torch.long)

    def store_and_concat_query(self, layer_id, latent, key, locations, query, suffix):
        return self._store_and_concat_query(
            layer_id, latent, key, locations, query, suffix, validate=self._validate_locations
        )

    def _store_and_concat_query(self, layer_id, latent, key, locations, query, suffix, *, validate):
        """Offer the mature FP8 writer for dense rows and virtual cyclic locations.

        Arena pages keep their existing stride-aware store. The fused SDK owns
        the E4M3 SATFINITE conversion; it is not an aten cast equivalence claim.
        """
        if (
            self.arena is not None
            or self.dtype != torch.float8_e4m3fn
            or (self.latent_dim, self.rope_dim) != (512, 64)
        ):
            return None
        from minisgl.kernel.mla import store_mla_and_concat_query

        validate(locations)
        if latent.shape != (len(locations), 512) or key.shape != (len(locations), 64):
            raise ValueError("MLA token/cache shapes differ; do not write padded tokens")
        return store_mla_and_concat_query(
            self._mla[self._full_layers[layer_id]],
            locations,
            latent,
            key,
            query,
            suffix,
            cp_size=self._cp.size,
            cp_rank=self._cp.rank,
        )

    def store_mla(
        self,
        layer_id: int,
        latent: torch.Tensor,
        key_suffix: torch.Tensor,
        out_loc: torch.Tensor,
        *,
        implementation="torch",
    ) -> None:
        self._validate_locations(out_loc)
        self._store_mla(layer_id, latent, key_suffix, out_loc, implementation=implementation)

    def _store_mla(self, layer_id, latent, key_suffix, out_loc, *, implementation="torch"):
        locations = out_loc.to(device=self.device, dtype=torch.long)
        if latent.shape != (len(locations), self.latent_dim) or key_suffix.shape != (
            len(locations),
            self.rope_dim,
        ):
            raise ValueError("MLA token/cache shapes differ; do not write padded tokens")
        if implementation == "sgl" and latent.is_cuda:
            from minisgl.kernel import store_mla

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
                self.paged_storage(layer_id),
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

    def gather_mla(self, layer_id: int, locations: torch.Tensor, *, out=None):
        locations = self._locations(locations)
        _check(
            self._cp.owns(locations).all(),
            "MLA reads must select only this DCP rank's token positions",
        )
        storage = self._mla if self.arena is None else self.arena.buffer
        if out is not None:
            if (
                out.shape != (len(locations), self.latent_dim + self.rope_dim)
                or out.device != self.device
                or not out.is_floating_point()
                or out.requires_grad
                or out.stride(-1) != 1
                or out.stride(0) < out.shape[1]
                or torch._C._overlaps(out, storage)
            ):
                raise ValueError("MLA read output requires a separate row-major floating buffer")
            if self.arena is None:
                from minisgl.kernel.mla import try_gather_mla

                if try_gather_mla(
                    self._mla[self._full_layers[layer_id]],
                    locations // self._cp.size,
                    out[:, : self.latent_dim],
                    out[:, self.latent_dim :],
                ):
                    return out[:, : self.latent_dim], out[:, self.latent_dim :]
        if self.arena is not None:
            storage, pages, offsets = self._paged_locations(locations)
            data = storage[pages, self._full_layers[layer_id], offsets]
        else:
            data = self._mla[self._full_layers[layer_id]].index_select(
                0, locations // self._cp.size
            )
        if out is not None:
            out.copy_(data)
            data = out
        return data[:, : self.latent_dim], data[:, self.latent_dim :]
