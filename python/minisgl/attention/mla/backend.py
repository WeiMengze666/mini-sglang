"""Paged MLA execution and graph plans, independent of recurrent attention."""

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed.context import ContextParallel

from ..base import BaseAttnBackend, SequenceAttnMetadata
from . import CausalLayout, MLAPagedMetadata, mla_attention


@dataclass
class MLAMetadata(SequenceAttnMetadata):
    decode: MLAPagedMetadata | None = None


class MLAAttentionBackend(BaseAttnBackend):
    supports_graph_padding = False
    cache_implementation = "torch"

    def __init__(self, config, *, attention=mla_attention, cache=None):
        self.config = config
        self.attention = attention
        self.kvcache = get_global_ctx().kv_cache if cache is None else cache
        self._capture_metadata = {}
        self._capture_pages = []
        self._graph_sizes = set()

    def prepare_metadata(self, batch):
        metadata = MLAMetadata.from_batch(batch, self.kvcache.device)
        if hasattr(self.attention, "decode") and all(req.extend_len == 1 for req in batch.reqs):
            metadata.decode = self.kvcache.mla.decode_metadata(
                [req.table_idx for req in batch.reqs],
                [req.device_len for req in batch.reqs],
                get_global_ctx().page_table,
            )
        batch.attn_metadata = metadata

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        positions = batch.positions[: len(x)]
        queries, latent, key = layer.project(x, positions)
        cache.store_mla(
            layer_id,
            latent,
            key,
            batch.out_loc[: len(x)],
            implementation=self.cache_implementation,
        )
        paged = batch.attn_metadata.decode
        if paged is not None:
            partial = layer.attend_paged_partial(
                queries, cache.mla.paged_layer(layer_id), paged, attention=self.attention
            )
            if partial is not None:
                return layer.finish(x, partial)
        cp = ContextParallel()
        outputs, offset = [], 0
        for req in batch.reqs:
            rows = slice(offset, offset + req.extend_len)
            history = torch.arange(cp.rank, max(cp.rank, req.device_len), cp.size, device=x.device)
            locations = page_table[req.table_idx, cp.rank : req.device_len : cp.size]
            latent, key = cache.gather_mla(layer_id, locations)
            outputs.append(
                layer.attend_partial(
                    queries[rows],
                    latent,
                    key,
                    positions[rows],
                    history,
                    causal_layout=CausalLayout(req.cached_len, cp.rank, cp.size),
                    attention=self.attention,
                )
            )
            offset += req.extend_len
        partial = (
            torch.cat(outputs)
            if cp.size == 1
            else tuple(torch.cat(parts) for parts in zip(*outputs))
        )
        return layer.finish(x, partial)

    def forward(self, q, k, v, layer_id, batch):
        raise NotImplementedError("MLA uses latent KV projections")

    def init_capture_graph(self, max_seq_len, bs_list):
        if not bs_list:
            return
        if not hasattr(self.attention, "init_graphs"):
            raise ValueError("MLA CUDA graphs require a native paged attention backend")
        layout = self.config.cache_config
        if self.kvcache.dtype != torch.bfloat16 or layout.latent_dim != 512 or layout.key_dim != 64:
            raise ValueError("Native MLA graph decode supports BF16 and MLA512/64")
        page_size = self.kvcache.mla.page_size
        self.attention.init_graphs(
            bs_list, (max_seq_len + page_size - 1) // page_size, self.kvcache.device
        )
        self._graph_sizes = set(bs_list)

    def prepare_for_capture(self, batch):
        if batch.size not in self._graph_sizes:
            raise ValueError("No MLA CUDA graph was initialized for this batch size")
        if self.kvcache.arena is not None:
            page_size = self.kvcache.mla.page_size
            self._capture_pages = list(range((batch.size + page_size - 1) // page_size))
            self.kvcache.arena.allocate_pages(self._capture_pages)
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        # Warm a nonempty local tile on every DCP rank. Replay supplies real lengths.
        metadata.decode = self.kvcache.mla.decode_metadata(
            [req.table_idx for req in batch.reqs],
            [ContextParallel().size] * batch.size,
            get_global_ctx().page_table,
        )
        metadata.decode.capture_size = batch.size
        metadata.decode.nonempty = metadata.decode.lengths.to(self.kvcache.device) > 0
        self._capture_metadata[batch.size] = metadata

    def finish_capture(self, batch):
        if self._capture_pages:
            self.kvcache.arena.free_pages_by_id(self._capture_pages)
        self._capture_pages = []

    def prepare_for_replay(self, batch):
        if batch.size not in self._graph_sizes:
            raise ValueError("No MLA CUDA graph was initialized for this batch size")
        metadata = batch.attn_metadata.decode
        captured = self._capture_metadata[batch.size].decode
        if metadata.nonempty is None:
            captured.nonempty.fill_(True)
        else:
            captured.nonempty.copy_(metadata.nonempty)
        self.attention.prepare_replay(metadata, batch.size)
