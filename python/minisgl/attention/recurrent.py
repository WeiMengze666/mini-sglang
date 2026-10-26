"""Eager-only metadata for the explicit Hybrid KDA/MLA model path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class HybridMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if not 0 <= bs < self.cu_seqlens_q.numel():
            raise ValueError("Requested batch size exceeds real Hybrid requests")
        return self.cu_seqlens_q[1 : bs + 1] - 1


class RecurrentAttentionBackend(BaseAttnBackend):
    def __init__(self, config):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        return forward_layer(layer, layer_id, x, batch, cache, page_table)

    def prepare_metadata(self, batch) -> None:
        lengths = [req.extend_len for req in batch.reqs]
        if any(length <= 0 for length in lengths):
            raise ValueError("Hybrid requests must contain real new tokens")
        batch.attn_metadata = HybridMetadata(
            torch.tensor([0] + lengths, dtype=torch.int32, device=self.kvcache.device).cumsum(0)
        )

    def forward(self, q, k, v, layer_id, batch):
        raise NotImplementedError("Hybrid model uses KDA and MLA layers")

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        if bs_list:
            raise ValueError("Hybrid attention is eager-only; disable CUDA graphs")

    def prepare_for_capture(self, batch) -> None:
        raise ValueError("Hybrid attention does not support CUDA graph capture")

    def prepare_for_replay(self, batch) -> None:
        raise ValueError("Hybrid attention does not support CUDA graph replay")


def forward_layer(layer, layer_id, x, batch, cache, page_table):
    """Apply a stateful attention layer to real request rows only.

    Prefix lengths, token-page addressing, and request state stay out of model
    definitions. This reference path deliberately does not process padding.
    """
    outputs, offset = [], 0
    for req in batch.reqs:
        rows = slice(offset, offset + req.extend_len)
        current, positions = x[rows], batch.positions[rows]
        if layer.state_kind == "recurrent":
            state = cache.get_kda_state(layer_id, req.table_idx)
            out, state = layer.forward(current, positions, state)
            cache.set_kda_state(layer_id, req.table_idx, state)
        else:
            q, latent, key = layer.project(current, positions)
            cache.store_mla(layer_id, latent, key, batch.out_loc[rows])
            latent, key = cache.gather_mla(layer_id, page_table[req.table_idx, : req.device_len])
            history = torch.arange(req.device_len, device=x.device)
            out = layer.attend(current, q, latent, key, positions, history)
        outputs.append(out)
        offset += req.extend_len
    return torch.cat(outputs)
