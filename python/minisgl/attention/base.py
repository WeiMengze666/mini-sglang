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
class HybridAttnMetadata(SequenceAttnMetadata):
    kda: SequenceAttnMetadata
    mla: SequenceAttnMetadata


class BaseAttnBackend(ABC):
    supports_graph_padding = True

    def prepare_model(self, attention_configs) -> None:
        """Bind optional immutable global-layer configurations before graph setup."""

    def prepare_communication(self, *, dtype: torch.dtype) -> None:
        """Prepare selected decode communication once, before graph warmup."""

    def checkpoint_quantum(self, batch=None) -> int | None:
        """Optional all-layer prefill tracking capability; no reservation or writes."""
        return None

    def finish_capture(self, batch) -> None:
        """Release any temporary cache ownership used to warm/capture a graph."""

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class QKVAttnBackend(BaseAttnBackend):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...


class LayerAttnBackend(BaseAttnBackend):
    """Attention with layer-owned projections and algorithm-specific batch state."""

    @abstractmethod
    def forward_layer(
        self, layer, layer_id, x, batch, cache, page_table, *, residual=None, metadata=None
    ): ...

    @abstractmethod
    def build_metadata(self, batch): ...

    @abstractmethod
    def capture_metadata(self, batch): ...

    @abstractmethod
    def replay_metadata(self, batch, metadata): ...

    def prepare_metadata(self, batch):
        batch.attn_metadata = self.build_metadata(batch)

    def prepare_for_capture(self, batch):
        batch.attn_metadata = self.capture_metadata(batch)

    def prepare_for_replay(self, batch):
        self.replay_metadata(batch, batch.attn_metadata)


class PrefillDecodeAttnBackend(BaseAttnBackend):
    def prepare_communication(self, *, dtype: torch.dtype):
        self.decode_backend.prepare_communication(dtype=dtype)

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

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table, *, residual=None):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward_layer(
            layer, layer_id, x, batch, cache, page_table, residual=residual
        )

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


class HybridAttnBackend(LayerAttnBackend):
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

    def prepare_communication(self, *, dtype: torch.dtype):
        self.mla_backend.prepare_communication(dtype=dtype)

    def checkpoint_quantum(self, batch=None):
        return self.kda_backend.checkpoint_quantum(batch)

    def forward_layer(
        self, layer, layer_id, x, batch, cache, page_table, *, residual=None, metadata=None
    ):
        metadata = batch.attn_metadata if metadata is None else metadata
        if self.is_kda_layer(layer_id):
            backend, child = self.kda_backend, metadata.kda
        else:
            backend, child = self.mla_backend, metadata.mla
        return backend.forward_layer(
            layer, layer_id, x, batch, cache, page_table, metadata=child, residual=residual
        )

    def build_metadata(self, batch):
        kda = self.kda_backend.build_metadata(batch)
        mla = self.mla_backend.build_metadata(batch, cu_seqlens_q=kda.cu_seqlens_q)
        return HybridAttnMetadata(kda.cu_seqlens_q, kda, mla)

    def init_capture_graph(self, max_seq_len, bs_list):
        self.kda_backend.init_capture_graph(max_seq_len, bs_list)
        self.mla_backend.init_capture_graph(max_seq_len, bs_list)
        self._graph_sizes = set(bs_list)

    def capture_metadata(self, batch):
        from minisgl.core import get_global_ctx

        if not self._graph_sizes:
            raise ValueError("No hybrid CUDA graph sizes were initialized")
        # Use real, owned rows for stateful capture instead of a padded dummy row.
        batch.reqs = [replace(batch.reqs[0], table_idx=i, uid=-i - 1) for i in range(batch.size)]
        batch.padded_reqs = batch.reqs
        device = get_global_ctx().kv_cache.device
        batch.out_loc = torch.arange(batch.size, device=device, dtype=torch.int32)
        try:
            kda = self.kda_backend.capture_metadata(batch)
            mla = self.mla_backend.capture_metadata(batch, cu_seqlens_q=kda.cu_seqlens_q)
            return HybridAttnMetadata(kda.cu_seqlens_q, kda, mla)
        except Exception:
            self.finish_capture(batch)
            raise

    def finish_capture(self, batch):
        try:
            self.kda_backend.finish_capture(batch)
        finally:
            self.mla_backend.finish_capture(batch)

    def replay_metadata(self, batch, metadata):
        if not self._graph_sizes:
            raise ValueError("No hybrid CUDA graph sizes were initialized")
        self.kda_backend.replay_metadata(batch, metadata.kda)
        self.mla_backend.replay_metadata(batch, metadata.mla)
