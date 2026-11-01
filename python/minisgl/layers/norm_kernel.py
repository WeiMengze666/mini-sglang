"""Per-row RMS normalization with explicit intermediate rounding semantics."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _rms(
    X,
    W,
    G,
    Y,
    D: tl.constexpr,
    XS: tl.constexpr,
    GS: tl.constexpr,
    EPS: tl.constexpr,
    ROUND: tl.constexpr,
    GATED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * XS[0] + cols * XS[1], cols < D, 0).to(tl.float32)
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    normalized = x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    if ROUND:
        normalized = normalized.to(X.dtype.element_ty).to(tl.float32)
    output = normalized * w
    if GATED:
        gate = tl.load(G + row * GS[0] + cols * GS[1], cols < D, 0).to(tl.float32)
        output *= tl.sigmoid(gate)
    tl.store(Y + row * D + cols, output, cols < D)


def rms_norm(x, weight, eps, *, gate=None, round_before_weight=False):
    """Normalize every row with the same reduction, independent of token grouping."""
    flat = x.reshape(-1, x.shape[-1])
    gates = None if gate is None else gate.reshape_as(flat)
    out = torch.empty(flat.shape, dtype=x.dtype, device=x.device)
    if len(flat):
        _rms[(len(flat),)](
            flat,
            weight,
            gates,
            out,
            flat.shape[1],
            flat.stride(),
            (0, 0) if gates is None else gates.stride(),
            eps,
            round_before_weight,
            gate is not None,
            tr.next_power_of_2(flat.shape[1]),
            num_warps=4 if flat.shape[1] < 1024 else 8,
            enable_fp_fusion=False,
        )
    return out.view(x.shape)
