"""Batch execution for hybrid recurrent attention and paged MLA."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed.context import ContextParallel

from .base import BaseAttnBackend, BaseAttnMetadata
from .mla import CausalLayout, MLAPagedMetadata


@dataclass
class HybridMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor
    mla_decode: MLAPagedMetadata | None = None
    state_indices: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if not 0 <= bs < self.cu_seqlens_q.numel():
            raise ValueError("Requested batch size exceeds real Hybrid requests")
        return self.cu_seqlens_q[1 : bs + 1] - 1


class RecurrentAttentionBackend(BaseAttnBackend):
    # A padded row would mutate recurrent state. Use exact graph sizes and
    # retain eager execution for sizes that were not captured.
    supports_graph_padding = False

    def __init__(self, config, *, kda_implementation="torch"):
        self.config = config
        self.kda_implementation = kda_implementation
        from .mla import mla_attention

        self.mla_attention = mla_attention
        if kda_implementation == "flashinfer":
            from .mla_fi import FlashInferMLA

            self.mla_attention = FlashInferMLA()
        self.kvcache = get_global_ctx().kv_cache
        self._capture_metadata = {}
        self._capture_reqs = []
        self._capture_pages = []
        self._graph_sizes = set()

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        return forward_layer(
            layer,
            layer_id,
            x,
            batch,
            cache,
            page_table,
            kda_implementation=self.kda_implementation,
            mla_attention=self.mla_attention,
        )

    def prepare_metadata(self, batch) -> None:
        lengths = [req.extend_len for req in batch.reqs]
        if any(length <= 0 for length in lengths):
            raise ValueError("Hybrid requests must contain real new tokens")
        batch.attn_metadata = HybridMetadata(
            torch.tensor([0] + lengths, dtype=torch.int32, device=self.kvcache.device).cumsum(0)
        )
        if hasattr(self.mla_attention, "decode") and all(length == 1 for length in lengths):
            batch.attn_metadata.mla_decode = self.kvcache.mla.decode_metadata(
                [req.table_idx for req in batch.reqs],
                [req.device_len for req in batch.reqs],
                get_global_ctx().page_table,
            )

    def forward(self, q, k, v, layer_id, batch):
        raise NotImplementedError("Hybrid model uses KDA and MLA layers")

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        if not bs_list:
            return
        if not hasattr(self.mla_attention, "init_graphs"):
            raise ValueError("Recurrent CUDA graphs require a native paged MLA backend")
        layout = self.config.cache_config
        if (
            self.kvcache.dtype != torch.bfloat16
            or layout.head_dim != 128
            or layout.latent_dim != 512
            or layout.key_dim != 64
        ):
            raise ValueError("Native graph decode supports BF16 KDA128 and MLA512/64")
        page_size = self.kvcache.mla.page_size
        self.mla_attention.init_graphs(
            bs_list, (max_seq_len + page_size - 1) // page_size, self.kvcache.device
        )
        self._graph_sizes = set(bs_list)

    def prepare_for_capture(self, batch) -> None:
        if not self._graph_sizes:
            raise ValueError("No native recurrent CUDA graph sizes were initialized")
        if not self.kvcache.recurrent.is_idle:
            raise RuntimeError("Capture requires an idle recurrent pool before admission")
        # Borrow ordinary slots only during startup. Replay addresses them via
        # device indices, so no scratch state remains reserved after capture.
        batch.reqs = [replace(batch.reqs[0], table_idx=i, uid=-i - 1) for i in range(batch.size)]
        batch.padded_reqs = batch.reqs
        self._capture_reqs = batch.reqs
        batch.out_loc = torch.arange(batch.size, device=self.kvcache.device, dtype=torch.int32)
        if self.kvcache.arena is not None:
            page_size = self.kvcache.mla.page_size
            pages = list(range((batch.size + page_size - 1) // page_size))
            self.kvcache.arena.allocate_pages(pages)
            self._capture_pages = pages
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        # Warm a nonempty local attention tile on every rank. Replay supplies
        # the real lengths, including ranks with no visible keys.
        metadata.mla_decode = self.kvcache.mla.decode_metadata(
            [req.table_idx for req in batch.reqs],
            [ContextParallel().size] * batch.size,
            get_global_ctx().page_table,
        )
        metadata.mla_decode.capture_size = batch.size
        metadata.mla_decode.nonempty = metadata.mla_decode.lengths.to(self.kvcache.device) > 0
        self._capture_metadata[batch.size] = metadata

    def finish_capture(self, batch):
        for req in self._capture_reqs:
            self.kvcache.release_request(req.table_idx, req.uid)
        if self._capture_pages:
            self.kvcache.arena.free_pages_by_id(self._capture_pages)
        self._capture_reqs, self._capture_pages = [], []

    def prepare_for_replay(self, batch) -> None:
        if not self._graph_sizes:
            raise ValueError("No native recurrent CUDA graph sizes were initialized")
        metadata = batch.attn_metadata
        captured = self._capture_metadata[batch.size]
        captured.state_indices.copy_(
            self.kvcache.recurrent.batch_indices([req.table_idx for req in batch.reqs])
        )
        nonempty = metadata.mla_decode.nonempty
        if nonempty is None:
            captured.mla_decode.nonempty.fill_(True)
        else:
            captured.mla_decode.nonempty.copy_(nonempty)
        self.mla_attention.prepare_replay(metadata.mla_decode, batch.size)


def forward_layer(
    layer, layer_id, x, batch, cache, page_table, *, kda_implementation="torch", mla_attention=None
):
    """Apply a stateful attention layer to real request rows only.

    Prefix lengths, token-page addressing, and request state stay out of model
    definitions. This reference path deliberately does not process padding.
    """
    recurrent = layer.state_kind == "recurrent"
    positions = batch.positions[: len(x)]
    cp = ContextParallel()
    metadata = getattr(batch, "attn_metadata", None)
    if recurrent:
        projected = layer.project(x)
        if all(req.extend_len == 1 for req in batch.reqs):
            indices = metadata.state_indices if isinstance(metadata, HybridMetadata) else None
            if indices is None:
                indices = cache.recurrent.batch_indices([req.table_idx for req in batch.reqs])
                if isinstance(metadata, HybridMetadata):
                    metadata.state_indices = indices
            initial = cache.recurrent.get_kda_batch(layer_id, indices)
            output, final = layer.update(
                *(part.unsqueeze(1) for part in projected),
                initial,
                implementation=kda_implementation,
            )
            cache.recurrent.set_kda_batch(layer_id, indices, final)
            return layer.finish(x, output[:, 0])
    else:
        queries, latent, key = layer.project(x, positions)
        cache.store_mla(
            layer_id,
            latent,
            key,
            batch.out_loc[: len(x)],
            implementation="triton" if kda_implementation == "flashinfer" else "torch",
        )
        paged = getattr(metadata, "mla_decode", None)
        if paged is not None and hasattr(mla_attention, "decode"):
            partial = layer.attend_paged_partial(
                queries,
                cache.mla.paged_layer(layer_id),
                paged,
                attention=mla_attention,
            )
            if partial is not None:
                return layer.finish(x, partial)
    kwargs = {} if mla_attention is None else {"attention": mla_attention}
    outputs, offset = [], 0
    for req in batch.reqs:
        rows = slice(offset, offset + req.extend_len)
        if recurrent:
            state = cache.get_kda_state(layer_id, req.table_idx)
            out, state = layer.update(
                *(part[rows] for part in projected),
                state,
                implementation=kda_implementation,
            )
            cache.set_kda_state(layer_id, req.table_idx, state)
        else:
            history = torch.arange(cp.rank, max(cp.rank, req.device_len), cp.size, device=x.device)
            locations = page_table[req.table_idx, cp.rank : req.device_len : cp.size]
            latent, key = cache.gather_mla(layer_id, locations)
            out = layer.attend_partial(
                queries[rows],
                latent,
                key,
                positions[rows],
                history,
                causal_layout=CausalLayout(req.cached_len, cp.rank, cp.size),
                **kwargs,
            )
        outputs.append(out)
        offset += req.extend_len
    partial = (
        torch.cat(outputs)
        if recurrent or cp.size == 1
        else tuple(torch.cat(parts) for parts in zip(*outputs))
    )
    return layer.finish(x, partial)
