from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _tl_tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _situ_and_mul_kernel(
    x_ptr,  # [M, 2N] gate;up halves (non-interleaved)
    out_ptr,  # [M, N]
    N,
    situ_beta,
    linear_beta,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    base = x_ptr + pid_m * stride_xm
    gate = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + N + offs, mask=mask, other=0.0).to(tl.float32)
    gate = situ_beta * _tl_tanh(gate / situ_beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * _tl_tanh(up / linear_beta)
    out = gate * up
    tl.store(
        out_ptr + pid_m * stride_om + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def situ_and_mul(
    output: torch.Tensor,
    x: torch.Tensor,
    situ_beta: float,
    linear_beta: Optional[float],
) -> None:
    """SiTU gated activation (Kimi K3), fused into one elementwise kernel:
    out = situ_beta*tanh(gate/situ_beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
    where x = [gate; up] halves along the last dim.
    """
    M, N2 = x.shape
    N = N2 // 2
    assert output.shape == (M, N)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _situ_and_mul_kernel[grid](
        x,
        output,
        N,
        float(situ_beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        x.stride(0),
        output.stride(0),
        BLOCK_N=BLOCK_N,
        HAS_LINEAR_BETA=linear_beta is not None,
    )
