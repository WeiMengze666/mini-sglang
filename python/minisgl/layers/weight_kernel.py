"""Shared on-device decoding of packed E2M1 weights and E8M0 scales."""

import triton as tr
import triton.language as tl


@tr.jit
def decode_mxfp4(W, S, rows, cols, N: tl.constexpr, K: tl.constexpr):
    packed = tl.load(
        W + rows[:, None] * (K // 2) + cols[None, :] // 2,
        (rows[:, None] < N) & (cols[None, :] < K),
        0,
    )
    code = (packed >> (4 * (cols[None, :] % 2))) & 15
    exponent = (code >> 1) & 3
    half = (code & 1).to(tl.float32) * 0.5
    magnitude = tl.where(exponent == 0, half, (1 + half) * tl.exp2(exponent.to(tl.float32) - 1))
    scale = tl.load(
        S + rows[:, None] * (K // 32) + cols[None, :] // 32,
        (rows[:, None] < N) & (cols[None, :] < K),
        127,
    )
    factor = tl.exp2(scale.to(tl.float32) - 127)
    factor = tl.where(scale == 255, float("nan"), factor)
    return tl.where((code & 8) != 0, -magnitude, magnitude) * factor
