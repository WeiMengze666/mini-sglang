"""Request-local KDA execution, independent of MLA and model layer schedules."""

from dataclasses import dataclass, field

import torch
from minisgl.core import get_global_ctx

from ..base import BaseAttnBackend, SequenceAttnMetadata
from .registry import KDAKernelSpec, get_kda_kernel


@dataclass
class KDAMetadata(SequenceAttnMetadata):
    state_indices: torch.Tensor | None = None
    lengths: tuple[int, ...] = ()
    has_initial_states: tuple[bool, ...] = ()
    kernel_data: dict = field(default_factory=dict)


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
        if self.kernel.indexed_prefill is not None and not getattr(batch, "is_decode", False):
            batch.attn_metadata.lengths = tuple(req.extend_len for req in batch.reqs)
            batch.attn_metadata.has_initial_states = tuple(req.cached_len > 0 for req in batch.reqs)

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        metadata = batch.attn_metadata
        is_decode = getattr(batch, "is_decode", False)
        single_step = all(req.extend_len == 1 for req in batch.reqs)
        # Packed prefill dispatch precedes the one-token-per-row shortcut.
        # All-T1 batches with more than one total token are still prefill.
        if not is_decode and self.kernel.indexed_prefill is not None:
            if metadata.state_indices is None:
                metadata.state_indices = cache.recurrent.batch_indices(
                    [req.table_idx for req in batch.reqs]
                )
            output = self.kernel.indexed_prefill(
                layer, x, cache.recurrent, layer_id, metadata.state_indices, metadata
            )
            if output is not None:
                return layer.finish(x, output)
        if single_step:
            indices = metadata.state_indices
            if indices is None:
                indices = cache.recurrent.batch_indices([req.table_idx for req in batch.reqs])
                metadata.state_indices = indices
            if is_decode and self.kernel.indexed_decode is not None:
                result = self.kernel.indexed_decode(layer, x, cache.recurrent, layer_id, indices)
                if result is not None:
                    return layer.finish(x, result.output, normed=result.normed)
            initial = cache.recurrent.get_kda_batch(layer_id, indices)
            if self.kernel.decode is not None and is_decode:
                fused = layer.decode(x, initial, kernel=self.kernel)
                if fused is not None:
                    result, final = fused
                    cache.recurrent.set_kda_batch(layer_id, indices, final)
                    return result
        projected = (
            layer.project(x) if self.kernel.project is None else self.kernel.project(layer, x)
        )
        if single_step:
            output, final = layer.update(
                *(part.unsqueeze(1) for part in projected),
                initial,
                kernel=self.kernel,
                is_decode=is_decode,
            )
            cache.recurrent.set_kda_batch(layer_id, indices, final)
            return layer.finish(x, output[:, 0])
        outputs, offset = [], 0
        for req in batch.reqs:
            rows = slice(offset, offset + req.extend_len)
            state = cache.get_kda_state(layer_id, req.table_idx)
            output, state = layer.update(
                *(part[rows] for part in projected),
                state,
                kernel=self.kernel,
                is_decode=is_decode,
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
