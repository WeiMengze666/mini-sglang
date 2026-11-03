"""Adapt native causal-convolution kernels to request-local functional state."""

from functools import cache

import torch


@cache
def _kernels():
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    return causal_conv1d_fn, causal_conv1d_update


def causal_conv1d_update(x, weight, state, *, is_decode=False):
    batched = x.ndim == 3
    batch, tokens, channels = x.shape if batched else (1, *x.shape)
    history = weight.shape[-1] - 1
    # Native pools keep the channel axis contiguous. Always own the writable
    # pool, even when the caller already uses this transposed physical layout.
    pool = (
        state.reshape(batch, channels, history)
        .transpose(1, 2)
        .clone(memory_format=torch.contiguous_format)
        .transpose(1, 2)
    )
    prefill, decode = _kernels()
    if is_decode:
        if tokens != 1:
            raise ValueError("Convolution decode requires one token per request")
        output = decode(x.reshape(batch, channels), pool, weight, activation="silu")
        output = output.reshape(batch, 1, channels)
    else:
        packed = x.reshape(batch * tokens, channels).transpose(0, 1)
        output = (
            prefill(
                packed,
                weight,
                None,
                conv_states=pool,
                query_start_loc=torch.arange(batch + 1, device=x.device, dtype=torch.int32)
                * tokens,
                seq_lens_cpu=[tokens] * batch,
                cache_indices=torch.arange(batch, device=x.device, dtype=torch.int32),
                has_initial_state=torch.ones(batch, device=x.device, dtype=torch.bool),
                activation="silu",
            )
            .transpose(0, 1)
            .reshape(batch, tokens, channels)
        )
    return (output, pool) if batched else (output[0], pool[0])
