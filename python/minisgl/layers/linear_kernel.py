"""Projection kernels for dense BF16 and packed E2M1/E8M0 weights."""

import torch
import triton as tr
import triton.language as tl

from .weight_kernel import decode_mxfp4 as _decode


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


@tr.jit
def _gemv(
    X,
    W,
    S,
    B,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    token = tl.program_id(1)
    offsets = tl.arange(0, BK)
    acc = tl.zeros((BN,), tl.float32)
    for block in range(tr.cdiv(K, BK)):
        cols = block * BK + offsets
        x = tl.load(X + token * K + cols, cols < K, 0).to(tl.float32)
        w = _decode(W, S, rows, cols, N, K).to(Y.dtype.element_ty).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    if BIAS:
        acc += tl.load(B + rows, rows < N, 0).to(tl.float32)
    tl.store(Y + token * N + rows, acc, rows < N)


@tr.jit(do_not_specialize=["M"])
def _gemm(
    X,
    W,
    S,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tokens = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tr.cdiv(K, BK)):
        cols = block * BK + offsets
        x = tl.load(
            X + tokens[:, None] * K + cols[None, :], (tokens[:, None] < M) & (cols[None, :] < K), 0
        )
        w = _decode(W, S, rows, cols, N, K).to(x.dtype)
        acc = tl.dot(x, tl.trans(w), acc, input_precision="tf32x3")
    if BIAS:
        acc += tl.load(B + rows, rows < N, 0)[None, :].to(tl.float32)
    tl.store(
        Y + tokens[:, None] * N + rows[None, :], acc, (tokens[:, None] < M) & (rows[None, :] < N)
    )


def mxfp4_linear(x, packed, scales, bias=None):
    if x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Triton MXFP4 requires CUDA BF16, FP16 or FP32 inputs")
    if not packed.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Triton MXFP4 requires contiguous packed weights and scales")
    n, k = packed.shape[0], x.shape[-1]
    flat = x.reshape(-1, k).contiguous()
    output = x.new_empty((len(flat), n))
    if len(flat) <= 4:
        if len(flat):
            _gemv[(tr.cdiv(n, 16), len(flat))](
                flat, packed, scales, bias, output, n, k, bias is not None, 16, 256, num_warps=4
            )
    else:
        _gemm[(tr.cdiv(len(flat), 16), tr.cdiv(n, 32))](
            flat,
            packed,
            scales,
            bias,
            output,
            len(flat),
            n,
            k,
            bias is not None,
            16,
            32,
            64,
            num_warps=4,
        )
    return output.view(*x.shape[:-1], n)
