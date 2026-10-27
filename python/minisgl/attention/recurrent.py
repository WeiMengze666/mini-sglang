"""Eager-only metadata for the explicit Hybrid KDA/MLA model path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed.context import ContextParallel

from .base import BaseAttnBackend, BaseAttnMetadata
from .kda import KDAState
from .mla import CausalLayout


@dataclass
class HybridMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if not 0 <= bs < self.cu_seqlens_q.numel():
            raise ValueError("Requested batch size exceeds real Hybrid requests")
        return self.cu_seqlens_q[1 : bs + 1] - 1


class RecurrentAttentionBackend(BaseAttnBackend):
    def __init__(self, config, *, kda_implementation="torch"):
        self.config = config
        self.kda_implementation = kda_implementation
        from .mla import mla_attention

        self.mla_attention = mla_attention
        if kda_implementation == "flashinfer":
            from .mla_fi import FlashInferMLA

            self.mla_attention = FlashInferMLA()
        self.kvcache = get_global_ctx().kv_cache

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

    def forward(self, q, k, v, layer_id, batch):
        raise NotImplementedError("Hybrid model uses KDA and MLA layers")

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        if bs_list:
            raise ValueError("Hybrid attention is eager-only; disable CUDA graphs")

    def prepare_for_capture(self, batch) -> None:
        raise ValueError("Hybrid attention does not support CUDA graph capture")

    def prepare_for_replay(self, batch) -> None:
        raise ValueError("Hybrid attention does not support CUDA graph replay")


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
    if recurrent:
        projected = layer.project(x)
        if len(batch.reqs) > 1 and all(req.extend_len == 1 for req in batch.reqs):
            states = [cache.get_kda_state(layer_id, req.table_idx) for req in batch.reqs]
            initial = KDAState(*(torch.stack(parts) for parts in zip(*states)))
            output, final = layer.update(
                *(part.unsqueeze(1) for part in projected),
                initial,
                implementation=kda_implementation,
            )
            for index, req in enumerate(batch.reqs):
                cache.set_kda_state(
                    layer_id, req.table_idx, KDAState(final.conv[index], final.recurrent[index])
                )
            return layer.finish(x, output[:, 0])
    else:
        queries, latent, key = layer.project(x, positions)
        cache.store_mla(layer_id, latent, key, batch.out_loc[: len(x)])
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
