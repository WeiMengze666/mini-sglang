"""Single-token depthwise convolution and SiLU with explicit FP32 arithmetic."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _update(
    X,
    W,
    S,
    Y,
    NEXT,
    C: tl.constexpr,
    N: tl.constexpr,
    XB: tl.constexpr,
    SB: tl.constexpr,
    SC: tl.constexpr,
    ST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < N
    batch, channel = row // C, row % C
    history = batch * SB + channel * SC
    h0 = tl.load(S + history, valid, 0)
    h1 = tl.load(S + history + ST, valid, 0)
    h2 = tl.load(S + history + 2 * ST, valid, 0)
    current = tl.load(X + batch * XB + channel, valid, 0)
    p0 = h0.to(tl.float32) * tl.load(W + channel * 4, valid, 0)
    p1 = h1.to(tl.float32) * tl.load(W + channel * 4 + 1, valid, 0)
    p2 = h2.to(tl.float32) * tl.load(W + channel * 4 + 2, valid, 0)
    p3 = current.to(tl.float32) * tl.load(W + channel * 4 + 3, valid, 0)
    # Match the existing four-tap Torch sum and SiLU, without contracting
    # products into FMAs or rounding through the input dtype before activation.
    value = (p0 + p2) + (p1 + p3)
    output = tl.div_rn(value, 1.0 + libdevice.exp(-value))
    tl.store(Y + row, output, valid)
    tl.store(NEXT + row * 3, h1, valid)
    tl.store(NEXT + row * 3 + 1, h2, valid)
    tl.store(NEXT + row * 3 + 2, current, valid)


def causal_conv1d_update(x, weight, state):
    """Return output and new history; borrowed request state is never mutated.

    The caller checks the four-tap, single-token CUDA layout. History strides
    remain explicit so slices of a grouped recurrent pool need no packing copy.
    """
    channels = x.shape[-1]
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    tail = state.new_empty(state.shape)
    _update[(tr.cdiv(x.numel(), 128),)](
        x,
        weight,
        state,
        output,
        tail,
        channels,
        x.numel(),
        x.stride(0) if x.ndim == 3 else 0,
        state.stride(0) if state.ndim == 3 else 0,
        state.stride(-2),
        state.stride(-1),
        128,
        enable_fp_fusion=False,
    )
    return output, tail
