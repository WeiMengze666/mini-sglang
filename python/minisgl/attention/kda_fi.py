"""Public FlashInfer KDA adapter; external state uses the framework's K,V order."""

import torch


def recurrent_kda(q, k, v, forget, beta_logits, a_log, dt_bias, state, lower_bound):
    if q.shape[0] == 1:
        # The small-BH public FI decode route stores BF16 state. Use our native
        # K,V recurrence so decode preserves the FP32 checkpoint contract.
        from .kda_kernel import decode_kda

        return decode_kda(q, k, v, forget, beta_logits, a_log, dt_bias, state, lower_bound)
    if lower_bound is None:
        # FI's unbounded prefill portfolio currently requires BF16 state.
        # Keep the portable FP32 contract for this unsupported combination.
        from .kda import kda_log_decay, kda_recurrence

        decay = kda_log_decay(forget, a_log, dt_bias)
        return kda_recurrence(q, k, v, decay, beta_logits.float().sigmoid(), state)
    from flashinfer import recurrent_kda as run

    # A contiguous request slice can still start at an unaligned byte offset.
    # The FP32-state prefill backend requires a 16-byte-aligned beta base.
    beta_logits = beta_logits.contiguous()
    if beta_logits.data_ptr() % 16:
        beta_logits = beta_logits.clone()
    # FlashInfer stores V,K while the portable recurrence stores K,V. Never
    # reinterpret the same square matrix without exchanging its semantic axes.
    initial = (
        state.transpose(-1, -2).contiguous().unsqueeze(0)
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
