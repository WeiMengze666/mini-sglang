"""Public FlashInfer KDA adapter; external state uses the framework's K,V order."""

from contextlib import ExitStack
from functools import cache
from math import isfinite, lcm
from types import MappingProxyType

import torch


def recurrent_kda(
    q, k, v, forget, beta_logits, a_log, dt_bias, state, lower_bound, *, is_decode=False
):
    if is_decode and q.shape[0] != 1:
        raise ValueError("KDA decode requires one token per request")
    if q.shape[0] == 1 or lower_bound is None:
        # The native SDK portfolio preserves either BF16 or FP32 state for
        # cases the public FlashInfer portfolio does not currently cover.
        from .registry import get_kda_kernel

        return get_kda_kernel("sgl").recurrence(
            q, k, v, forget, beta_logits, a_log, dt_bias, state, lower_bound, is_decode=is_decode
        )
    from flashinfer import recurrent_kda as run

    # A contiguous request slice can still start at an unaligned byte offset.
    # The FP32-state prefill backend requires a 16-byte-aligned beta base.
    beta_logits = beta_logits.contiguous()
    if beta_logits.data_ptr() % 16:
        beta_logits = beta_logits.clone()
    # FlashInfer stores V,K while the portable recurrence stores K,V. Never
    # reinterpret the same square matrix without exchanging its semantic axes.
    initial = (
        state.float().transpose(-1, -2).contiguous().unsqueeze(0)
        if state is not None
        else torch.zeros(
            (1, q.shape[1], v.shape[-1], q.shape[-1]), dtype=torch.float32, device=q.device
        )
    )
    output, final = run(
        q.contiguous().unsqueeze(0),
        k.contiguous().unsqueeze(0),
        v.contiguous().unsqueeze(0),
        forget.contiguous().unsqueeze(0),
        beta_logits.unsqueeze(0),
        backend="cute-dsl",
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        lower_bound=lower_bound,
        ssm_state_indices=torch.zeros(1, dtype=torch.int32, device=q.device),
    )
    return output[0], final[0].transpose(-1, -2).contiguous()


# The official class retains only its callable, not scheduler or request state.
@cache
def _prefill_kernel():
    from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
        FlashInferKDAPrefillKernel,
    )

    return FlashInferKDAPrefillKernel()


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
            raise ValueError("Every checkpoint layer must use the native FI prefill portfolio")
    return configs


def checkpoint_quantum(layers_by_id, config, cache, precision, batch=None):
    """Native FI eligibility only; querying never reserves or writes state."""
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
        raise ValueError("Native FI checkpoint tracking requires SM100 or SM103")
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


def indexed_prefill(layer, x, pool, layer_id, indices, metadata):
    """Opt-in native FI prefill preserving the original indexed state storage."""
    import math

    from . import sgl
    from .prefill import prepare, supports

    tracked = getattr(metadata, "checkpoints", None)
    if not supports(layer, x, pool):
        if tracked is not None:
            raise ValueError("Checkpoint prefill input no longer meets its validated portfolio")
        return None
    lower_bound = layer._config.gate_lower_bound
    # This pinned SDK facade reshapes gate bias to [H,D], but FI's grouped
    # all-T1 portfolio requires a flat bias. Use mature chunk prefill before
    # any convolution/state write; this is an ABI capability limit, not arithmetic.
    # Capability rejection precedes every convolution/state write.
    if (
        len(x) <= 1
        or (metadata.lengths and all(length == 1 for length in metadata.lengths))
        or lower_bound is None
        or not math.isfinite(lower_bound)
        or lower_bound >= 0
        or torch.cuda.get_device_capability(x.device) not in ((10, 0), (10, 3))
    ):
        if tracked is not None:
            raise ValueError("Checkpoint prefill cannot switch to an untracked fallback")
        return sgl.indexed_prefill(layer, x, pool, layer_id, indices, metadata)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Native FI KDA prefill requires eager execution; full prefill graph is unsupported"
        )
    kernel = _prefill_kernel()
    _, _, stream = sgl._kernels()
    with ExitStack() as borrows:
        destinations = (
            [
                borrows.enter_context(pool.borrow_checkpoint(plan, layer_id))
                for plan in tracked.plans
            ]
            if tracked is not None
            else []
        )
        conv_tracking = (
            {"checkpoint_conv": tuple(conv for conv, _ in destinations)}
            if tracked is not None
            else {}
        )
        q, k, v, g, beta, recurrent, plan, indices = prepare(
            layer, x, pool, layer_id, indices, metadata, **conv_tracking
        )
        tracking = {}
        if tracked is not None:
            snapshots = recurrent.new_empty(
                (len(tracked.plans), *recurrent.shape[1:]), dtype=torch.float32
            )
            tracking = {
                "state_checkpoint_cu_starts": tracked.cu_starts,
                "num_state_checkpoints": len(tracked.plans),
                "state_checkpoint_every_n_tokens": tracked.quantum,
                "state_checkpoint_indices": tracked.destinations,
                "track_ssm_h_batch_src": tracked.track_rows,
                "track_state": snapshots,
            }
        # No exception recovery after prepare: a native error aborts this forward.
        with stream():
            output = kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=recurrent,
                cache_indices=indices,
                query_start_loc=plan["cu_seqlens"],
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                lower_bound=lower_bound,
                return_intermediate_states=tracked is not None,
                **tracking,
            )
        if tracked is not None:
            output, _ = output
            for (_, recurrent_state), snapshot in zip(destinations, snapshots):
                # Public destination is K,V; the mature kernel snapshot is V,K.
                recurrent_state.transpose(-1, -2).copy_(snapshot)
        return output[0]
