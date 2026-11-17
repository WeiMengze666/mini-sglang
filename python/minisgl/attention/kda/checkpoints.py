"""Shared metadata and eligibility for native indexed KDA checkpoints."""

from math import isfinite, lcm
from types import MappingProxyType

import torch


def _checkpoint_device_supported(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


def validate_checkpoint_layers(layers_by_id, config):
    """Pure all-layer native portfolio validation; no metadata or state access."""
    from minisgl.attention.config import KDAConfig

    expected = tuple(config.recurrent_layers)
    if not expected or set(layers_by_id) != set(expected):
        raise ValueError("Checkpoint configuration must cover every recurrent layer")
    configs = tuple((layer_id, layers_by_id[layer_id]) for layer_id in expected)
    for _, layer in configs:
        if not isinstance(layer, KDAConfig):
            raise TypeError("Checkpoint layers must supply actual KDAConfig objects")
        if (
            layer.num_heads != config.num_heads
            or layer.head_dim != config.head_dim
            or layer.conv_kernel_size != config.conv_kernel_size
            or layer.head_dim != 128
            or layer.conv_kernel_size != 4
            or layer.gate_lower_bound is None
            or not isfinite(layer.gate_lower_bound)
            or layer.gate_lower_bound >= 0
        ):
            raise ValueError("Every checkpoint layer must use the native KDA prefill portfolio")
    return configs


def checkpoint_quantum(layers_by_id, config, cache, precision, batch=None):
    """Native KDA eligibility only; querying never reserves or writes state."""
    try:
        validate_checkpoint_layers(layers_by_id, config)
    except (TypeError, ValueError):
        return None
    states = cache.recurrent
    if (
        states.conv_dtype != torch.bfloat16
        or states.state_dtype != torch.bfloat16
        or precision != "default"
        or not _checkpoint_device_supported(cache.device)
    ):
        return None
    if batch is not None and (
        not batch.is_prefill
        or torch.cuda.is_current_stream_capturing()
        or all(req.extend_len == 1 for req in batch.reqs)
        or any(req.extend_len <= 0 for req in batch.reqs)
    ):
        return None
    return 64


def prepare_checkpoints(batch, metadata, layers_by_id, config, device, page_size, *, cache):
    """Validate all actual layers/leases before building compact native metadata."""
    from .backend import KDACheckpoints

    if cache.recurrent.conv_dtype != torch.bfloat16 or cache.recurrent.state_dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("Checkpoint tracking requires native BF16 convolution storage")
    if not batch.is_prefill:
        raise ValueError("Checkpoint tracking is eager prefill only")
    configs = validate_checkpoint_layers(layers_by_id, config)
    lengths = tuple(req.extend_len for req in batch.reqs)
    if (
        not lengths
        or any(length <= 0 for length in lengths)
        or all(length == 1 for length in lengths)
        or metadata.lengths != lengths
        or metadata.has_initial_states != tuple(req.cached_len > 0 for req in batch.reqs)
    ):
        raise ValueError("Checkpoint metadata requires a non-T1 packed prefill batch")
    plans = tuple(batch.checkpoint_plan)
    request_rows = {(req.table_idx, req.uid): row for row, req in enumerate(batch.reqs)}
    if len(request_rows) != len(batch.reqs):
        raise ValueError("Checkpoint batch cannot repeat a request owner")
    quantum, history = 64, config.conv_kernel_size - 1
    counts = [length // quantum for length in lengths]
    starts, query_starts = [0], [0]
    for count, length in zip(counts, lengths):
        starts.append(starts[-1] + count)
        query_starts.append(query_starts[-1] + length)
    destinations, rows, conv_indices = [-1] * starts[-1], [], []
    for destination, plan in enumerate(plans):
        row = request_rows.get((plan.table_idx, plan.uid))
        if row is None or row in rows:
            raise ValueError("Checkpoint leases must identify distinct real batch rows")
        req = batch.reqs[row]
        frontier = plan.checkpoint.length
        relative = frontier - plan.start_len
        if (
            plan.start_len != req.cached_len
            or plan.end_len != req.device_len
            or not plan.start_len < frontier < plan.end_len
            or frontier % lcm(page_size, quantum)
            or relative % quantum
            or relative < history
        ):
            raise ValueError("Checkpoint frontier does not match its forward/grid/history")
        destinations[starts[row] + relative // quantum - 1] = destination
        rows.append(row)
        conv_indices.append(
            list(range(query_starts[row] + relative - history, query_starts[row] + relative))
        )
    if not _checkpoint_device_supported(device):
        raise ValueError("Native KDA checkpoint tracking requires SM100 or SM103")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("Checkpoint tracking requires eager prefill")
    # No tensor allocation/native execution above: unsupported later layers cannot
    # leave partial batch metadata or a partially written destination.
    return KDACheckpoints(
        plans=plans,
        layer_configs=MappingProxyType(dict(configs)),
        rows=tuple(rows),
        conv_indices=torch.tensor(conv_indices, dtype=torch.int64, device=device),
        cu_starts=torch.tensor(starts, dtype=torch.int64, device=device),
        destinations=torch.tensor(destinations, dtype=torch.int32, device=device),
        track_rows=torch.arange(len(plans), dtype=torch.int64, device=device),
        quantum=quantum,
    )
