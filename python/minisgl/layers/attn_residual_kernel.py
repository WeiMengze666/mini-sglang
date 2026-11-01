"""Fused per-token RMS scores and FP32 attention over a residual bank."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _scores(
    X, NORM, PROJ, S, BANK: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    token, block = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + (token * BANK + block) * H + cols, cols < H, 0).to(tl.float32)
    norm = tl.load(NORM + cols, cols < H, 0).to(tl.float32)
    proj = tl.load(PROJ + cols, cols < H, 0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    score = tl.sum(((x * inverse) * norm) * proj, 0)
    tl.store(S + token * BANK + block, score)


@tr.jit
def _mix(X, S, Y, BANK: tl.constexpr, H: tl.constexpr, BB: tl.constexpr):
    token = tl.program_id(0)
    blocks = tl.arange(0, BB)
    cols = tl.program_id(1) * 256 + tl.arange(0, 256)
    scores = tl.load(S + token * BANK + blocks, blocks < BANK, -float("inf"))
    weights = tl.exp(scores - tl.max(scores, 0))
    weights /= tl.sum(weights, 0)
    rows = tl.load(
        X + (token * BANK + blocks[:, None]) * H + cols[None, :],
        (blocks[:, None] < BANK) & (cols[None, :] < H),
        0,
    ).to(tl.float32)
    mixed = tl.sum(weights[:, None] * rows, 0)
    tl.store(Y + token * H + cols, mixed, cols < H)


def attention_residual(rows, norm_weight, projection_weight, eps):
    tokens, bank, hidden = rows.shape
    scores = torch.empty((tokens, bank), dtype=torch.float32, device=rows.device)
    output = torch.empty((tokens, hidden), dtype=rows.dtype, device=rows.device)
    if tokens:
        _scores[(tokens, bank)](
            rows,
            norm_weight,
            projection_weight,
            scores,
            bank,
            hidden,
            eps,
            tr.next_power_of_2(hidden),
            enable_fp_fusion=False,
        )
        _mix[(tokens, tr.cdiv(hidden, 256))](
            rows, scores, output, bank, hidden, tr.next_power_of_2(bank), enable_fp_fusion=False
        )
    return output
