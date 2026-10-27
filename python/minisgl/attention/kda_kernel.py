"""Single-token delta recurrence with FP32 K,V state and raw gates."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _decode(
    Q,
    K,
    V,
    G,
    B,
    A,
    DT,
    S,
    NEXT,
    O,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    VALUE: tl.constexpr,
    LOWER: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(0)
    values = tl.program_id(1) * BV + tl.arange(0, BV)
    keys = tl.arange(0, DIM)
    q = tl.load(Q + head * DIM + keys).to(tl.float32)
    k = tl.load(K + head * DIM + keys).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (DIM**-0.5)
    k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    forget = tl.load(G + head * DIM + keys).to(tl.float32) + tl.load(DT + head * DIM + keys)
    a = tl.exp(tl.load(A + head))
    if LOWER is not None:
        gate = LOWER * tl.sigmoid(a * forget)
    else:
        gate = -a * (tl.maximum(forget, 0) + tl.log(1 + tl.exp(-tl.abs(forget))))
    offsets = head * DIM * VALUE + keys[:, None] * VALUE + values[None, :]
    state = tl.load(S + offsets, values[None, :] < VALUE, 0) * tl.exp(gate[:, None])
    prediction = tl.sum(k[:, None] * state, 0)
    beta = tl.sigmoid(tl.load(B + head).to(tl.float32))
    delta = (
        tl.load(V + head * VALUE + values, values < VALUE, 0).to(tl.float32) - prediction
    ) * beta
    state += k[:, None] * delta[None, :]
    tl.store(NEXT + offsets, state, values[None, :] < VALUE)
    output = tl.sum(q[:, None] * state, 0)
    tl.store(O + head * VALUE + values, output, values < VALUE)


def decode_kda(q, k, v, forget, beta, a, dt, state, lower):
    _, heads, dim = q.shape
    value = v.shape[-1]
    initial = (
        state
        if state is not None
        else torch.zeros((heads, dim, value), device=q.device, dtype=torch.float32)
    )
    initial = initial.contiguous()
    next_state = torch.empty_like(initial)
    output = torch.empty_like(v)
    _decode[(heads, tr.cdiv(value, 16))](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        forget.contiguous(),
        beta.contiguous(),
        a,
        dt,
        initial,
        next_state,
        output,
        heads,
        dim,
        value,
        lower,
        16,
        num_warps=4,
    )
    return output, next_state
