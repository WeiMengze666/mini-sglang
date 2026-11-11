"""Request-major ragged layouts and bounded cyclic-prefix exchange for MLA."""

from dataclasses import dataclass
from itertools import accumulate

import torch


@dataclass(frozen=True)
class RaggedLayout:
    lengths: torch.Tensor
    cu_lengths: torch.Tensor
    max_length: int

    @classmethod
    def from_lengths(cls, lengths, device, *, cu_lengths=None):
        return cls(
            torch.tensor(lengths, dtype=torch.int32, device=device),
            torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
            if cu_lengths is None
            else cu_lengths.to(dtype=torch.int32),
            max(lengths, default=0),
        )


@dataclass(frozen=True)
class PrefixChunk:
    layout: RaggedLayout
    locations: torch.Tensor
    # Per request: real rows, padded local rows, local rows, left padding.
    segments: tuple[tuple[int, int, int, int], ...]
    phase: int
    has_empty: bool

    def gather(self, cache, layer_id, cp, dtype):
        latent, key = cache.gather_mla(layer_id, self.locations)
        latent_dim = latent.shape[-1]
        local = torch.cat((latent, key), -1).to(dtype)
        if cp.size == 1:
            return local[:, :latent_dim], local[:, latent_dim:]
        padded_rows = sum(segment[1] for segment in self.segments)
        padded = local.new_empty((padded_rows, local.shape[-1]))
        local_offset = padded_offset = 0
        for _, rows, count, left in self.segments:
            padded[padded_offset + left : padded_offset + left + count].copy_(
                local[local_offset : local_offset + count]
            )
            local_offset += count
            padded_offset += rows
        joined = cp.gather_interleaved(padded, padded_rows * cp.size)
        pieces, offset = [], 0
        for length, rows, _, _ in self.segments:
            first = offset * cp.size + self.phase
            pieces.append(joined[first : first + length])
            offset += rows
        gathered = torch.cat(pieces)
        return gathered[:, :latent_dim], gathered[:, latent_dim:]


@dataclass(frozen=True)
class MLAPrefillMetadata:
    queries: RaggedLayout
    prefix_chunks: tuple[PrefixChunk, ...]

    @classmethod
    def from_batch(cls, batch, page_table, cp, capacity, *, cu_lengths=None):
        lengths = tuple(req.extend_len for req in batch.reqs)
        if not lengths or min(lengths) <= 0:
            raise ValueError("MLA prefill requires nonempty request queries")
        if capacity < len(lengths):
            raise ValueError("Prefix chunk capacity must fit one token per request")
        queries = RaggedLayout.from_lengths(lengths, page_table.device, cu_lengths=cu_lengths)
        chunk_length = capacity // len(lengths)
        prefixes = tuple(req.cached_len for req in batch.reqs)
        chunks = []
        for start in range(0, max(prefixes, default=0), chunk_length):
            chunk_lengths = tuple(min(max(prefix - start, 0), chunk_length) for prefix in prefixes)
            segments, locations = [], []
            phase = start % cp.size
            for req, length in zip(batch.reqs, chunk_lengths):
                first = start + (cp.rank - start) % cp.size
                indices = page_table[req.table_idx, first : start + length : cp.size]
                locations.append(indices)
                rows = (phase + length + cp.size - 1) // cp.size if length else 0
                segments.append((length, rows, len(indices), int(phase > cp.rank)))
            chunks.append(
                PrefixChunk(
                    RaggedLayout.from_lengths(chunk_lengths, page_table.device),
                    torch.cat(locations),
                    tuple(segments),
                    phase,
                    0 in chunk_lengths,
                )
            )
        return cls(queries, tuple(chunks))
