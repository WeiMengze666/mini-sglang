"""FP32-accumulating BF16 projections with a fixed per-row reduction layout."""

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["M"])
def _linear(
    X,
    W,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    YS: tl.constexpr,
    BIAS: tl.constexpr,
):
    head = tl.program_id(2)
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 64)
    acc = tl.zeros((16, 32), tl.float32)
    compensation = tl.zeros((16, 32), tl.float32)
    for start in range(tr.cdiv(K, 64)):
        k = start * 64 + inner
        x = tl.load(
            X + rows[:, None] * XS[0] + head * XS[1] + k[None, :] * XS[2],
            (rows[:, None] < M) & (k[None, :] < K),
            0,
        )
        w = tl.load(
            W + head * WS[0] + cols[:, None] * WS[1] + k[None, :] * WS[2],
            (cols[:, None] < N) & (k[None, :] < K),
            0,
        )
        block = tl.dot(x, tl.trans(w))
        corrected = block - compensation
        total = acc + corrected
        compensation = (total - acc) - corrected
        acc = total
    if BIAS:
        acc += tl.load(B + cols, cols < N, 0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * YS[0] + head * YS[1] + cols[None, :] * YS[2],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _project(x, weight, bias, dtype):
    tokens, heads, k = x.shape
    n = weight.shape[1]
    out = torch.empty((tokens, heads, n), dtype=dtype, device=x.device)
    if tokens:
        _linear[(tr.cdiv(tokens, 16), tr.cdiv(n, 32), heads)](
            x,
            weight,
            bias,
            out,
            tokens,
            n,
            k,
            x.stride(),
            weight.stride(),
            out.stride(),
            bias is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def linear(x, weight, bias=None, *, dtype=None):
    """Use one tile shape across singleton, batched and chunked projections."""
    flat = x.reshape(-1, x.shape[-1])
    out = _project(flat.unsqueeze(1), weight.unsqueeze(0), bias, dtype or x.dtype)
    return out.view(*x.shape[:-1], weight.shape[0])


def headwise_linear(x, weight):
    """Project [tokens, heads, input] with strided [heads, output, input] weights."""
    return _project(x, weight, None, x.dtype)
