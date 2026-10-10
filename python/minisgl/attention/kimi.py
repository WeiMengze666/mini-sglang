"""Eager-only metadata for the explicit Kimi KDA/MLA model path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class KimiMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if not 0 <= bs < self.cu_seqlens_q.numel():
            raise ValueError("Requested batch size exceeds real Kimi requests")
        return self.cu_seqlens_q[1 : bs + 1] - 1


class KimiBackend(BaseAttnBackend):
    def __init__(self, config):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache

    def prepare_metadata(self, batch) -> None:
        lengths = [req.extend_len for req in batch.reqs]
        if any(length <= 0 for length in lengths):
            raise ValueError("Kimi requests must contain real new tokens")
        batch.attn_metadata = KimiMetadata(
            torch.tensor([0] + lengths, dtype=torch.int32, device=self.kvcache.device).cumsum(0)
        )

    def forward(self, q, k, v, layer_id, batch):
        raise NotImplementedError("Kimi model uses explicit KDA and MLA attention operations")

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        if bs_list:
            raise ValueError("Kimi attention is eager-only; disable CUDA graphs")

    def prepare_for_capture(self, batch) -> None:
        raise ValueError("Kimi attention does not support CUDA graph capture")

    def prepare_for_replay(self, batch) -> None:
        raise ValueError("Kimi attention does not support CUDA graph replay")
