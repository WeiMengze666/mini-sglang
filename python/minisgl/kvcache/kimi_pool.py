"""Bounded request-local KDA state plus token-paged MLA latent storage."""

from __future__ import annotations

import torch
from minisgl.models.kimi_parallel import local_size, tp_info

from .base import BaseKVCachePool


def kimi_cache_bytes_per_token(config, dtype: torch.dtype) -> int:
    return (
        len(config.linear_attn_config["full_attn_layers"])
        * (config.kv_lora_rank + config.qk_rope_head_dim)
        * dtype.itemsize
    )


def kimi_state_bytes_per_request(config, dtype: torch.dtype, tp_size: int | None = None) -> int:
    linear = config.linear_attn_config
    if not linear["kda_layers"]:
        return 0
    heads = local_size(
        linear["num_heads"], tp_info()[1] if tp_size is None else tp_size, "KDA heads"
    )
    dim = linear["head_dim"]
    conv_width = linear["short_conv_kernel_size"] - 1
    return len(linear["kda_layers"]) * (
        heads * dim * dim * torch.float32.itemsize + 3 * heads * dim * conv_width * dtype.itemsize
    )


class KimiHybridCache(BaseKVCachePool):
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
    ):
        if min(num_pages, page_size, max_running_req) <= 0:
            raise ValueError("Kimi cache capacities must be positive")
        linear = config.linear_attn_config
        self._full_layers = {layer - 1: i for i, layer in enumerate(linear["full_attn_layers"])}
        self._kda_layers = {layer - 1: i for i, layer in enumerate(linear["kda_layers"])}
        self._num_layers = config.num_hidden_layers
        self.max_running_req = max_running_req
        self.num_tokens = num_pages * page_size
        self.latent_dim = config.kv_lora_rank
        self.rope_dim = config.qk_rope_head_dim
        heads = (
            local_size(linear["num_heads"], tp_info()[1], "KDA heads") if self._kda_layers else 0
        )
        dim = linear["head_dim"]
        self._mla = torch.empty(
            (len(self._full_layers), self.num_tokens, self.latent_dim + self.rope_dim),
            dtype=dtype,
            device=device,
        )
        self._recurrent = torch.zeros(
            (len(self._kda_layers), max_running_req, heads, dim, dim),
            dtype=torch.float32,
            device=device,
        )
        self._conv = torch.zeros(
            (
                len(self._kda_layers),
                max_running_req,
                3,
                heads * dim,
                linear["short_conv_kernel_size"] - 1,
            ),
            dtype=dtype,
            device=device,
        )
        self._owners: list[int | None] = [None] * max_running_req
        self._processed_lens = [0] * max_running_req
        self._active = [False] * max_running_req

    def _slot(self, table_idx: int) -> None:
        if not 0 <= table_idx < self.max_running_req:
            raise IndexError("Kimi request-state slot is out of range")

    def begin_request(self, table_idx: int, uid: int, cached_len: int) -> None:
        self._slot(table_idx)
        if cached_len < 0:
            raise ValueError("cached_len cannot be negative")
        if self._active[table_idx]:
            raise RuntimeError("Kimi request already has an unfinished forward")
        if self._owners[table_idx] != uid:
            if cached_len != 0:
                raise RuntimeError("Kimi cannot restore recurrent state from a token prefix")
            self._recurrent[:, table_idx].zero_()
            self._conv[:, table_idx].zero_()
            self._owners[table_idx] = uid
            self._processed_lens[table_idx] = 0
        if self._processed_lens[table_idx] != cached_len:
            raise RuntimeError("Kimi recurrent state does not match request cached_len")
        self._active[table_idx] = True

    def finish_request(self, table_idx: int, uid: int, processed_len: int) -> None:
        self._slot(table_idx)
        if self._owners[table_idx] != uid or not self._active[table_idx]:
            raise RuntimeError("Kimi request-state ownership mismatch")
        if processed_len <= self._processed_lens[table_idx]:
            raise ValueError("Kimi forward must process at least one token")
        self._processed_lens[table_idx] = processed_len
        self._active[table_idx] = False

    def release_request(self, table_idx: int, uid: int) -> None:
        self._slot(table_idx)
        # A late completion must never release the next request using this slot.
        if self._owners[table_idx] != uid:
            return
        self._recurrent[:, table_idx].zero_()
        self._conv[:, table_idx].zero_()
        self._owners[table_idx] = None
        self._processed_lens[table_idx] = 0
        self._active[table_idx] = False

    def reset(self) -> None:
        self._recurrent.zero_()
        self._conv.zero_()
        self._owners[:] = [None] * self.max_running_req
        self._processed_lens[:] = [0] * self.max_running_req
        self._active[:] = [False] * self.max_running_req

    def get_kda_state(self, layer_id: int, table_idx: int):
        from minisgl.models.kimi_attention import KDAState

        self._slot(table_idx)
        if self._owners[table_idx] is None or not self._active[table_idx]:
            raise RuntimeError("begin_request must precede KDA state access")
        layer = self._kda_layers[layer_id]
        return KDAState(
            conv=self._conv[layer, table_idx], recurrent=self._recurrent[layer, table_idx]
        )

    def set_kda_state(self, layer_id: int, table_idx: int, state) -> None:
        target = self.get_kda_state(layer_id, table_idx)
        if state.conv.shape != target.conv.shape or state.recurrent.shape != target.recurrent.shape:
            raise ValueError("KDA state shape mismatch")
        target.conv.copy_(state.conv)
        target.recurrent.copy_(state.recurrent)

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
        self._mla[self._full_layers[layer_id]].index_copy_(
            0, locations, torch.cat((latent, key_suffix), dim=-1).to(self._mla)
        )

    def gather_mla(self, layer_id: int, locations: torch.Tensor):
        data = self._mla[self._full_layers[layer_id]].index_select(0, self._locations(locations))
        return data[:, : self.latent_dim], data[:, self.latent_dim :]

    @property
    def allocated_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self._mla, self._conv, self._recurrent))

    @property
    def device(self) -> torch.device:
        return self._mla.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mla.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def k_cache(self, index: int) -> torch.Tensor:
        raise NotImplementedError("Kimi uses compressed MLA storage; use gather_mla")

    def v_cache(self, index: int) -> torch.Tensor:
        raise NotImplementedError("Kimi uses compressed MLA storage; use gather_mla")

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        raise NotImplementedError("Kimi uses store_mla and request-local KDA state")
