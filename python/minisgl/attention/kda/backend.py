"""Request-local KDA execution, independent of MLA and model layer schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch
from minisgl.attention.config import KDAConfig, MLAConfig
from minisgl.core import get_global_ctx
from minisgl.kvcache.recurrent_pool import CheckpointWrite

from ..base import BaseAttnBackend, SequenceAttnMetadata
from .registry import KDAKernelSpec, get_kda_kernel


@dataclass(frozen=True)
class KDACheckpoints:
    """Validated compact native tracking metadata; never owns pool views."""

    plans: tuple[CheckpointWrite, ...]
    layer_configs: Mapping[int, KDAConfig]
    rows: tuple[int, ...]
    conv_indices: torch.Tensor
    cu_starts: torch.Tensor
    destinations: torch.Tensor
    track_rows: torch.Tensor
    quantum: int


@dataclass
class KDAMetadata(SequenceAttnMetadata):
    state_indices: torch.Tensor | None = None
    lengths: tuple[int, ...] = ()
    has_initial_states: tuple[bool, ...] = ()
    kernel_data: dict = field(default_factory=dict)
    checkpoints: KDACheckpoints | None = None


class KDAAttentionBackend(BaseAttnBackend):
    supports_graph_padding = False

    def __init__(self, config, kernel: KDAKernelSpec | None = None, *, cache=None):
        self.config = config
        self.kernel = get_kda_kernel("torch") if kernel is None else kernel
        self.kvcache = get_global_ctx().kv_cache if cache is None else cache
        self._capture_metadata = {}
        self._capture_reqs = []
        self._graph_sizes = set()
        self.checkpoint_layer_configs: Mapping[int, KDAConfig] | None = None

    def prepare_model(self, attention_configs):
        # None retains Slice A's explicit component-only entry for undeclared models.
        self.checkpoint_layer_configs = None if not attention_configs else MappingProxyType({})
        if not attention_configs:
            return

        try:
            if not isinstance(attention_configs, tuple) or any(
                not isinstance(record, tuple) or len(record) != 2 for record in attention_configs
            ):
                return
            declared = dict(attention_configs)
            cache = self.config.cache_config
            if (
                len(declared) != len(attention_configs)
                or any(type(i) is not int for i in declared)
                or set(declared) != set(range(cache.num_layers))
                or any(not isinstance(c, (KDAConfig, MLAConfig)) for c in declared.values())
            ):
                return
            layers = {i: c for i, c in declared.items() if isinstance(c, KDAConfig)}
            if set(layers) != set(cache.recurrent_layers) or {
                i for i, c in declared.items() if isinstance(c, MLAConfig)
            } != set(cache.mla_layers):
                return
            configs = tuple((i, layers[i]) for i in cache.recurrent_layers)
        except (TypeError, ValueError):
            return
        self.checkpoint_layer_configs = MappingProxyType(dict(configs))

    def checkpoint_quantum(self, batch=None):
        states = self.kvcache.recurrent
        query = self.kernel.checkpoint_quantum
        if (
            not self.checkpoint_layer_configs
            or self.kernel.prepare_checkpoints is None
            or query is None
            or states.max_cached_states <= 0
            or getattr(self.kvcache, "arena", None) is not None
            or (
                self.kvcache.supports_overlap
                and not getattr(self.kvcache, "supports_checkpoint_overlap", False)
            )
        ):
            return None
        return query(
            self.checkpoint_layer_configs,
            self.config.cache_config,
            self.kvcache,
            get_global_ctx().bf16_precision,
            batch,
        )

    def prepare_metadata(self, batch):
        batch.attn_metadata = KDAMetadata.from_batch(batch, self.kvcache.device)
        if self.kernel.indexed_prefill is not None and not getattr(batch, "is_decode", False):
            batch.attn_metadata.lengths = tuple(req.extend_len for req in batch.reqs)
            batch.attn_metadata.has_initial_states = tuple(req.cached_len > 0 for req in batch.reqs)
        if getattr(batch, "checkpoint_plan", ()) and self.checkpoint_layer_configs is not None:
            self.prepare_checkpoints(batch)

    def _abort_checkpoints(self, batch, error):
        states = self.kvcache.recurrent
        if states.device_abort is not None:
            states.fail_device(error)
        for plan in getattr(batch, "checkpoint_plan", ()):
            if isinstance(plan, CheckpointWrite):
                try:
                    self.kvcache.recurrent.abort_checkpoint(plan)
                except BaseException as exc:
                    error.add_note(f"Checkpoint abort failed: {exc}")

    def prepare_checkpoints(self, batch, layers_by_id: Mapping[int, KDAConfig] | None = None):
        """Validate the model and immutable write leases before native metadata setup."""
        if not getattr(batch, "checkpoint_plan", ()):
            return
        try:
            metadata = batch.attn_metadata
            if self.checkpoint_layer_configs is not None:
                if not self.checkpoint_layer_configs:
                    raise ValueError("Declared model cannot track every recurrent layer")
                if layers_by_id is not None and layers_by_id != self.checkpoint_layer_configs:
                    raise ValueError("Checkpoint configs differ from the bound model")
                layers_by_id = self.checkpoint_layer_configs
            if self.kvcache.supports_overlap and not getattr(
                self.kvcache, "supports_checkpoint_overlap", False
            ):
                raise ValueError("Cache cannot preserve checkpoints across overlapping forwards")
            states = self.kvcache.recurrent
            if not isinstance(metadata, KDAMetadata) or metadata.checkpoints is not None:
                raise ValueError("Checkpoint preparation requires fresh KDA metadata")
            prepare = self.kernel.prepare_checkpoints
            if prepare is None:
                raise ValueError("Selected KDA prefill cannot track checkpoint states")
            for plan in getattr(batch, "checkpoint_plan", ()):
                if not isinstance(plan, CheckpointWrite):
                    raise TypeError("Checkpoint plans must be immutable write leases")
                states.validate_checkpoint_write(plan)
            # Validate every layer and plan before attaching any metadata.
            prepared = prepare(
                batch,
                metadata,
                layers_by_id,
                self.config.cache_config,
                self.kvcache.device,
                get_global_ctx().page_size,
                cache=self.kvcache,
            )
        except BaseException as error:
            self._abort_checkpoints(batch, error)
            raise
        metadata.checkpoints = prepared

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        metadata = batch.attn_metadata
        if getattr(batch, "checkpoint_plan", ()):
            try:
                tracked = getattr(metadata, "checkpoints", None)
                if tracked is None or tracked.plans != tuple(batch.checkpoint_plan):
                    raise ValueError(
                        "Nonempty checkpoint plan requires explicit validated metadata"
                    )
                if tracked.layer_configs.get(layer_id) != layer._config:
                    raise ValueError("Checkpoint layer configuration changed after validation")
            except BaseException as error:
                self._abort_checkpoints(batch, error)
                raise
        is_decode = getattr(batch, "is_decode", False)
        single_step = all(req.extend_len == 1 for req in batch.reqs)
        # Packed prefill dispatch precedes the one-token-per-row shortcut.
        # All-T1 batches with more than one total token are still prefill.
        if not is_decode and self.kernel.indexed_prefill is not None:
            if metadata.state_indices is None:
                metadata.state_indices = cache.recurrent.batch_indices(
                    [req.table_idx for req in batch.reqs]
                )
            try:
                output = self.kernel.indexed_prefill(
                    layer, x, cache.recurrent, layer_id, metadata.state_indices, metadata
                )
                if output is None and getattr(batch, "checkpoint_plan", ()):
                    raise ValueError("Validated checkpoint prefill cannot silently fall back")
                if output is not None:
                    return layer.finish(x, output)
            except BaseException as error:
                self._abort_checkpoints(batch, error)
                raise
        if single_step:
            indices = metadata.state_indices
            if indices is None:
                indices = cache.recurrent.batch_indices([req.table_idx for req in batch.reqs])
                metadata.state_indices = indices
            if is_decode and self.kernel.indexed_decode is not None:
                result = self.kernel.indexed_decode(
                    layer, x, cache.recurrent, layer_id, indices, metadata
                )
                if result is not None:
                    return layer.finish(
                        x, result.output, normed=result.normed, gate=result.gate
                    )
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
