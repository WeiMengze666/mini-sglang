"""Public FlashInfer KDA adapter; external state uses the framework's K,V order."""

from functools import cache

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


def indexed_prefill(layer, x, pool, layer_id, indices, metadata):
    """Opt-in native FI prefill preserving the original indexed state storage."""
    import math

    from . import sgl
    from .prefill import prepare, supports

    if not supports(layer, x, pool):
        return None
    lower_bound = layer._config.gate_lower_bound
    # The official portfolio falls back only for TOTAL packed tokens <= 1.
    # Capability rejection precedes every convolution/state write.
    if (
        len(x) <= 1
        or lower_bound is None
        or not math.isfinite(lower_bound)
        or lower_bound >= 0
        or torch.cuda.get_device_capability(x.device) not in ((10, 0), (10, 3))
    ):
        return sgl.indexed_prefill(layer, x, pool, layer_id, indices, metadata)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Native FI KDA prefill requires eager execution; full prefill graph is unsupported"
        )
    kernel = _prefill_kernel()
    _, _, stream = sgl._kernels()
    q, k, v, g, beta, recurrent, plan, indices = prepare(
        layer, x, pool, layer_id, indices, metadata
    )
    # No exception recovery after prepare: a native error aborts this forward.
    # FI all-T1 may use a TVM-FFI environment-stream decode portfolio.
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
            return_intermediate_states=False,
        )
    return output[0]
