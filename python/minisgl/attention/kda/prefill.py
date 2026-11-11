"""Shared indexed varlen convolution and storage adaptation for native prefill."""

from functools import cache

import torch


def supports(layer, x, pool):
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and pool.state_dtype in (torch.bfloat16, torch.float32)
        and pool.conv_dtype == torch.bfloat16
        and layer._dim == 128
        and layer._config.conv_kernel_size == 4
    )


@cache
def _conv_kernel():
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    return causal_conv1d_fn


def prepare(layer, x, pool, layer_id, indices, metadata, *, checkpoint_conv=()):
    """Validate before mutation, then return packed projections and original V,K state.

    Callers select/load their recurrence before entering this function. Once the
    convolution starts, errors propagate: retrying would advance state twice.
    """
    from .sgl import _kernels, project

    if (
        not metadata.lengths
        or any(length <= 0 for length in metadata.lengths)
        or sum(metadata.lengths) != len(x)
        or len(metadata.has_initial_states) != len(metadata.lengths)
        or indices.numel() != len(metadata.lengths)
    ):
        raise ValueError("Packed KDA metadata must describe every real query row")
    conv, recurrent = pool.indexed_kda_state(layer_id)
    heads, dim = layer._heads, layer._dim
    if (
        recurrent.shape[1:] != (heads, dim, dim)
        or recurrent.stride()[1:] != (dim * dim, dim, 1)
        or recurrent.data_ptr() % 16
        or recurrent.stride(0) * recurrent.element_size() % 16
        or recurrent.dtype != pool.state_dtype
        or recurrent.device != x.device
        or conv.device != x.device
        or conv.dtype != pool.conv_dtype
        or indices.device != x.device
    ):
        raise ValueError("Indexed KDA requires the original aligned native V,K state pool")
    plan = metadata.kernel_data.get("indexed_prefill")
    if plan is None:
        plan = {
            "cu_seqlens": metadata.cu_seqlens_q.to(dtype=torch.int32).contiguous(),
            "lengths": list(metadata.lengths),
            "initial": torch.tensor(metadata.has_initial_states, device=x.device, dtype=torch.bool),
            "indices": indices.to(dtype=torch.int32).contiguous(),
        }
        metadata.kernel_data["indexed_prefill"] = plan
    if (
        plan["cu_seqlens"].device != x.device
        or plan["cu_seqlens"].numel() != len(metadata.lengths) + 1
    ):
        raise ValueError("Packed KDA offsets must describe the active batch on its device")
    weights = layer.prepare_kernel(
        "sgl_indexed_conv",
        lambda: torch.cat(
            [op.weight.squeeze(1) for op in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
        ),
    )
    q, k, v, forget, beta = project(layer, x)
    if any(
        value.dtype != torch.bfloat16 or value.device != x.device
        for value in (q, k, v, forget, beta)
    ):
        raise ValueError("Native KDA prefill projections must be BF16 on the input device")
    _, _, stream = _kernels()
    conv_fn = _conv_kernel()
    raw = torch.cat((q, k, v), dim=-1)
    if checkpoint_conv:
        tracking = metadata.checkpoints
        if tracking is None or len(checkpoint_conv) != len(tracking.plans):
            raise ValueError("Convolution checkpoint destinations do not match metadata")
        windows = raw.index_select(0, tracking.conv_indices.flatten()).view(
            len(checkpoint_conv), layer._config.conv_kernel_size - 1, 3, heads * dim
        )
        for destination, window in zip(checkpoint_conv, windows):
            destination.copy_(window.permute(1, 2, 0))
    # Metadata, projections and both kernel imports are ready before state writes.
    with stream():
        mixed = conv_fn(
            raw.t(),
            weights,
            None,
            conv,
            query_start_loc=plan["cu_seqlens"],
            seq_lens_cpu=plan["lengths"],
            cache_indices=plan["indices"],
            has_initial_state=plan["initial"],
            activation="silu",
        ).t()
    q, k, v = mixed.split(heads * dim, dim=-1)
    shape = (1, len(x), heads, dim)
    beta = beta.contiguous().unsqueeze(0)
    if beta.data_ptr() % 16:
        beta = beta.clone()
    return (
        q.contiguous().view(shape),
        k.contiguous().view(shape),
        v.contiguous().view(shape),
        forget.contiguous().view(shape),
        beta,
        recurrent,
        plan,
        plan["indices"],
    )
