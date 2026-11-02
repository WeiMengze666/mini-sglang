"""Request-local KDA execution, independent of MLA and model layer schedules."""

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx

from ..base import BaseAttnBackend, SequenceAttnMetadata
from .registry import KDAKernelSpec, get_kda_kernel


@dataclass
class KDAMetadata(SequenceAttnMetadata):
    state_indices: torch.Tensor | None = None


class KDAAttentionBackend(BaseAttnBackend):
    supports_graph_padding = False

    def __init__(self, config, kernel: KDAKernelSpec | None = None, *, cache=None):
        self.config = config
        self.kernel = get_kda_kernel("torch") if kernel is None else kernel
        self.kvcache = get_global_ctx().kv_cache if cache is None else cache
        self._capture_metadata = {}
        self._capture_reqs = []
        self._graph_sizes = set()

    def prepare_metadata(self, batch):
        batch.attn_metadata = KDAMetadata.from_batch(batch, self.kvcache.device)

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        metadata = batch.attn_metadata
        single_step = all(req.extend_len == 1 for req in batch.reqs)
        if single_step:
            indices = metadata.state_indices
            if indices is None:
                indices = cache.recurrent.batch_indices([req.table_idx for req in batch.reqs])
                metadata.state_indices = indices
            initial = cache.recurrent.get_kda_batch(layer_id, indices)
            if self.kernel.decode is not None and batch.is_decode:
                fused = layer.decode(x, initial, kernel=self.kernel)
                if fused is not None:
                    result, final = fused
                    cache.recurrent.set_kda_batch(layer_id, indices, final)
                    return result
        projected = layer.project(x)
        if single_step:
            output, final = layer.update(
                *(part.unsqueeze(1) for part in projected), initial, kernel=self.kernel
            )
            cache.recurrent.set_kda_batch(layer_id, indices, final)
            return layer.finish(x, output[:, 0])
        outputs, offset = [], 0
        for req in batch.reqs:
            rows = slice(offset, offset + req.extend_len)
            state = cache.get_kda_state(layer_id, req.table_idx)
            output, state = layer.update(
                *(part[rows] for part in projected), state, kernel=self.kernel
            )
            cache.set_kda_state(layer_id, req.table_idx, state)
            outputs.append(output)
            offset += req.extend_len
        return layer.finish(x, torch.cat(outputs))

    def forward(self, q, k, v, layer_id, batch):
        raise NotImplementedError("KDA uses projected gates and recurrent state")

    def init_capture_graph(self, max_seq_len, bs_list):
        if bs_list and (
            self.kvcache.recurrent.conv_dtype != torch.bfloat16
            or self.config.cache_config.head_dim != 128
        ):
            raise ValueError("KDA graph decode supports BF16 and head_dim=128")
        self._graph_sizes = set(bs_list)

    def prepare_for_capture(self, batch):
        if batch.size not in self._graph_sizes:
            raise ValueError("No KDA CUDA graph was initialized for this batch size")
        if not self.kvcache.recurrent.is_idle:
            raise RuntimeError("Capture requires an idle recurrent pool before admission")
        self._capture_reqs = batch.reqs
        self.prepare_metadata(batch)
        self._capture_metadata[batch.size] = batch.attn_metadata

    def finish_capture(self, batch):
        for req in self._capture_reqs:
            self.kvcache.release_request(req.table_idx, req.uid)
        self._capture_reqs = []

    def prepare_for_replay(self, batch):
        if batch.size not in self._graph_sizes:
            raise ValueError("No KDA CUDA graph was initialized for this batch size")
        captured = self._capture_metadata[batch.size]
        captured.state_indices.copy_(
            self.kvcache.recurrent.batch_indices([req.table_idx for req in batch.reqs])
        )
