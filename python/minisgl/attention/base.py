from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


@dataclass
class SequenceAttnMetadata(BaseAttnMetadata):
    """Packed query offsets shared by stateful attention algorithms."""

    cu_seqlens_q: torch.Tensor

    @classmethod
    def from_batch(cls, batch, device):
        lengths = [req.extend_len for req in batch.reqs]
        if any(length <= 0 for length in lengths):
            raise ValueError("Attention requests must contain real new tokens")
        return cls(torch.tensor([0] + lengths, dtype=torch.int32, device=device).cumsum(0))

    def get_last_indices(self, bs):
        if not 0 <= bs < self.cu_seqlens_q.numel():
            raise ValueError("Requested batch size exceeds real requests")
        return self.cu_seqlens_q[1 : bs + 1] - 1


@dataclass
class HybridAttnMetadata(BaseAttnMetadata):
    kda: BaseAttnMetadata | None = None
    mla: BaseAttnMetadata | None = None

    def get_last_indices(self, bs):
        return self.kda.get_last_indices(bs)


class BaseAttnBackend(ABC):
    supports_graph_padding = True

    def prepare_model(self, attention_configs) -> None:
        """Bind optional immutable global-layer configurations before graph setup."""

    def checkpoint_quantum(self, batch=None) -> int | None:
        """Optional all-layer prefill tracking capability; no reservation or writes."""
        return None

    def finish_capture(self, batch) -> None:
        """Release any temporary cache ownership used to warm/capture a graph."""

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        """Execute a stateful projection layer, when supported by the backend."""
        raise NotImplementedError("This backend only supports Q/K/V attention")

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class PrefillDecodeAttnBackend(BaseAttnBackend):
    def prepare_model(self, attention_configs):
        self.prefill_backend.prepare_model(attention_configs)
        if self.decode_backend is not self.prefill_backend:
            self.decode_backend.prepare_model(attention_configs)

    def checkpoint_quantum(self, batch=None):
        return self.prefill_backend.checkpoint_quantum(batch)

    @property
    def supports_graph_padding(self):
        return self.decode_backend.supports_graph_padding

    def finish_capture(self, batch):
        self.decode_backend.finish_capture(batch)

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward_layer(layer, layer_id, x, batch, cache, page_table)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)


class HybridAttnBackend(BaseAttnBackend):
    """Compose KDA and MLA backends using a model-independent layer predicate."""

    def __init__(self, kda_backend, mla_backend, is_kda_layer):
        self.kda_backend = kda_backend
        self.mla_backend = mla_backend
        self.is_kda_layer = is_kda_layer
        self._graph_sizes = set()

    @property
    def supports_graph_padding(self):
        return self.kda_backend.supports_graph_padding and self.mla_backend.supports_graph_padding

    def prepare_model(self, attention_configs):
        self.kda_backend.prepare_model(attention_configs)
        if self.mla_backend is not self.kda_backend:
            self.mla_backend.prepare_model(attention_configs)

    def checkpoint_quantum(self, batch=None):
        return self.kda_backend.checkpoint_quantum(batch)

    def _call(self, kind, batch, method, *args):
        metadata = batch.attn_metadata
        assert isinstance(metadata, HybridAttnMetadata)
        batch.attn_metadata = getattr(metadata, kind)
        try:
            return getattr(getattr(self, kind + "_backend"), method)(*args)
        finally:
            setattr(metadata, kind, batch.attn_metadata)
            batch.attn_metadata = metadata

    def _prepare(self, method, batch):
        for kind in ("kda", "mla"):
            self._call(kind, batch, method, batch)

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        kind = "kda" if self.is_kda_layer(layer_id) else "mla"
        return self._call(
            kind, batch, "forward_layer", layer, layer_id, x, batch, cache, page_table
        )

    def forward(self, q, k, v, layer_id, batch):
        kind = "kda" if self.is_kda_layer(layer_id) else "mla"
        return self._call(kind, batch, "forward", q, k, v, layer_id, batch)

    def prepare_metadata(self, batch):
        batch.attn_metadata = HybridAttnMetadata()
        self._prepare("prepare_metadata", batch)

    def init_capture_graph(self, max_seq_len, bs_list):
        self.kda_backend.init_capture_graph(max_seq_len, bs_list)
        self.mla_backend.init_capture_graph(max_seq_len, bs_list)
        self._graph_sizes = set(bs_list)

    def prepare_for_capture(self, batch):
        from minisgl.core import get_global_ctx

        if not self._graph_sizes:
            raise ValueError("No hybrid CUDA graph sizes were initialized")
        # Use real, owned rows for stateful capture instead of a padded dummy row.
        batch.reqs = [replace(batch.reqs[0], table_idx=i, uid=-i - 1) for i in range(batch.size)]
        batch.padded_reqs = batch.reqs
        device = get_global_ctx().kv_cache.device
        batch.out_loc = torch.arange(batch.size, device=device, dtype=torch.int32)
        batch.attn_metadata = HybridAttnMetadata()
        try:
            self._prepare("prepare_for_capture", batch)
        except Exception:
            self.finish_capture(batch)
            raise

    def finish_capture(self, batch):
        try:
            self.kda_backend.finish_capture(batch)
        finally:
            self.mla_backend.finish_capture(batch)

    def prepare_for_replay(self, batch):
        if not self._graph_sizes:
            raise ValueError("No hybrid CUDA graph sizes were initialized")
        self._prepare("prepare_for_replay", batch)
