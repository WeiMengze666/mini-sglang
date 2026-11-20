"""Public FlashInfer KDA adapter; external state uses the framework's K,V order."""

from contextlib import ExitStack

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

        return get_kda_kernel("triton").recurrence(
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


def indexed_prefill(layer, x, pool, layer_id, indices, metadata):
    """Opt-in native FI prefill preserving the original indexed state storage."""
    import math

    from flashinfer.kda import recurrent_kda as run
    from tvm_ffi import use_torch_stream

    from . import triton
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
        return triton.indexed_prefill(layer, x, pool, layer_id, indices, metadata)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Native FI KDA prefill requires eager execution; full prefill graph is unsupported"
        )
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
        prepared, gate = prepare(layer, x, pool, layer_id, indices, metadata, **conv_tracking)
        q, k, v, g, beta, recurrent, plan, indices = prepared
        snapshots = (
            recurrent.new_empty((len(destinations), *recurrent.shape[1:]))
            if tracked is not None
            else None
        )
        with use_torch_stream():
            output = run(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                g=g[:, : q.shape[1]].contiguous(),
                beta=beta[:, : q.shape[1]].contiguous(),
                A_log=layer.A_log.reshape(-1).float().contiguous(),
                dt_bias=layer.dt_bias.reshape(q.shape[2], 128).float().contiguous(),
                initial_state=recurrent,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                lower_bound=lower_bound,
                cu_seqlens=plan["cu_seqlens"],
                ssm_state_indices=indices,
                beta_is_logit=True,
                state_checkpoints=snapshots,
                checkpoint_cu_starts=tracked.cu_starts if tracked else None,
                checkpoint_state_indices=tracked.destinations if tracked else None,
                checkpoint_every_n_tokens=tracked.quantum if tracked else 0,
                backend="cute-dsl",
            )[0]
        if tracked is not None:
            for (_, recurrent_state), snapshot in zip(destinations, snapshots):
                # Public destination is K,V; the mature kernel snapshot is V,K.
                recurrent_state.transpose(-1, -2).copy_(snapshot)
        from .registry import KDAResult

        return KDAResult(output=output[0], gate=gate)
