"""Legacy packed E2M1/E8M0 projection fallback."""

import torch
import triton as tr
import triton.language as tl

from .weight_kernel import decode_mxfp4 as _decode


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
