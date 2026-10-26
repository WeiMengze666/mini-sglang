"""Composition of paged MLA storage and request-local recurrent state."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from minisgl.distributed.parallel import local_size, tp_info

from .base import BaseKVCachePool
from .mla_pool import MLACache
from .recurrent_pool import RecurrentStatePool


class HybridCache(BaseKVCachePool):
    """Layer IDs are zero-based; checkpoint schedule entries are one-based.

    Recurrent state is owned by (table_idx, uid), not by prefix-cache pages.
    begin/finish bracket a complete model forward, never individual layers.
    """

    def __init__(
        self,
        config,
        num_pages: int,
        page_size: int,
        max_running_req: int,
        dtype: torch.dtype,
        device: torch.device,
        max_cached_states: int = 0,
        unified_memory: bool = False,
    ):
        if min(num_pages, page_size, max_running_req) <= 0:
            raise ValueError("Hybrid cache capacities must be positive")

        self._full_layers = {layer: i for i, layer in enumerate(config.mla_layers)}
        self._kda_layers = {layer: i for i, layer in enumerate(config.recurrent_layers)}
        self._num_layers = config.num_layers
        self.max_running_req = max_running_req
        self.num_tokens = num_pages * page_size
        self.latent_dim = config.latent_dim
        self.rope_dim = config.key_dim
        heads = local_size(config.num_heads, tp_info()[1], "KDA heads") if self._kda_layers else 0
        dim = config.head_dim
        self.arena = None
        if unified_memory:
            from .unified import UnifiedMemory

            self.arena = UnifiedMemory(
                num_pages,
                page_size * config.cache_bytes_per_token(dtype),
                config.state_bytes_per_request(dtype),
                device,
            )
        self.mla = MLACache(
            config.mla_layers,
            self.num_tokens,
            self.latent_dim,
            self.rope_dim,
            dtype,
            device,
            arena=self.arena,
            page_size=page_size,
        )
        self.recurrent = RecurrentStatePool(
            config.recurrent_layers,
            max_running_req,
            heads,
            dim,
            config.conv_kernel_size,
            dtype,
            device,
            max_cached_states=max_cached_states,
            arena=self.arena,
        )

    def begin_request(self, *args):
        self.recurrent.begin_request(*args)

    @contextmanager
    def forward_batch(self, batch):
        """Commit lengths only after every layer succeeds; discard partial updates."""
        started = []
        try:
            for req in batch.reqs:
                self.begin_request(req.table_idx, req.uid, req.cached_len)
                started.append(req)
            yield
            for req in started:
                self.finish_request(req.table_idx, req.uid, req.device_len)
        except Exception:
            for req in started:
                self.release_request(req.table_idx, req.uid)
            raise

    def finish_request(self, *args):
        self.recurrent.finish_request(*args)

    def release_request(self, *args):
        self.recurrent.release_request(*args)

    def reset(self):
        self.recurrent.reset()

    def get_kda_state(self, *args):
        return self.recurrent.get_kda_state(*args)

    def set_kda_state(self, *args):
        self.recurrent.set_kda_state(*args)

    def store_mla(self, layer_id, latent, key_suffix, out_loc):
        self.mla.store_mla(layer_id, latent, key_suffix, out_loc)

    def gather_mla(self, layer_id, locations):
        return self.mla.gather_mla(layer_id, locations)

    @property
    def _mla(self):
        return self.mla._mla

    @property
    def allocated_bytes(self) -> int:
        if self.arena is not None:
            return self.arena.buffer.numel()
        return sum(
            t.numel() * t.element_size()
            for t in (self.mla._mla, self.recurrent._conv, self.recurrent._recurrent)
        )

    @property
    def device(self) -> torch.device:
        return self.mla.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mla.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def k_cache(self, index: int) -> torch.Tensor:
        raise NotImplementedError("Hybrid uses compressed MLA storage; use gather_mla")

    def v_cache(self, index: int) -> torch.Tensor:
        raise NotImplementedError("Hybrid uses compressed MLA storage; use gather_mla")

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        raise NotImplementedError("Hybrid uses store_mla and request-local KDA state")
