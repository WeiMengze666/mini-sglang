from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# Kimi-K3 Attention Residual: snapshot bank + aggregation.
#
# The public API is the AttnResidual class (constructed once per forward pass).
# It owns the frozen snapshot bank [T, NB, H] and the valid-row counter, and
# dispatches each aggregation point (score rows → softmax → weighted sum →
# RMSNorm) by hardware capability:
#   fast  — warp-specialized TMA kernel: cp.async.bulk producer +
#           online-softmax consumers over a double-buffered chunk ring, out
#           norm fused, per-nvb tuned launch config, one persistent CTA per
#           SM. Taken on SM100+ except SM12x with H=7168.
#   hip   — single Triton kernel, everything in one launch; taken on ROCm
#           within its register budget.
#   fused — Triton 2-kernel pipeline with full H-parallelism; the fallback
#           everywhere the fast kernel does not apply.
# aggregate_stream_torch is the eager reference (tests and the
# H % _BLOCK_H != 0 shape fallback of aggregate_stream).
import torch
import triton
import triton.language as tl

_BLOCK_H: int = 1024  # H = 7168 = 7 x 1024

_MAX_ROWS: int = 16  # next_pow2(8 + 1), K3 has <= 8 snapshots

_FAST_SUPPORTED = None

_HIP_SHAPE_GATE = None


def _supports_attn_res_tma(capability: tuple[int, int]) -> bool:
    """Return whether the device is eligible for the TMA fast path."""
    major, _ = capability
    return major >= 10 and major != 12


def _use_fast(hidden_size: int) -> bool:
    """The TMA kernel needs SM100+ except SM12x (tcgen05, cp.async.bulk)
    and its H=7168 template; everything else takes the triton pipeline."""
    global _FAST_SUPPORTED
    if not torch.cuda.is_available() or torch.version.hip:
        return False
    if _FAST_SUPPORTED is None:
        _FAST_SUPPORTED = _supports_attn_res_tma(torch.cuda.get_device_capability())
    return _FAST_SUPPORTED and hidden_size == 7168


def get_cw(
    proj: object,
    norm: object,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cached product norm_weight ⊙ proj_weight (both [H]) in `dtype`.

    Cached per dtype: the fast kernel consumes bf16 while the triton path
    consumes fp32, and a shared slot would hand one path the other's dtype."""
    cache = getattr(proj, "_attn_res_cw_cache", None)
    if cache is None:
        cache = {}
        proj._attn_res_cw_cache = cache
    cw = cache.get(dtype)
    if cw is None:
        cw = (norm.weight.float() * proj.weight.squeeze().float()).contiguous()
        cw = cache[dtype] = cw.to(dtype)
    return cw


def _aggregate_fast(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: object,
    score_norm: object,
    out_norm: object,
    write_bank_row: bool = False,
) -> torch.Tensor:
    """Warp-specialized TMA kernel: online softmax over row chunks with the
    output RMSNorm fused, one persistent CTA per SM, per-nvb tuned launch
    config (GB300 benchmark winner across nvb). With write_bank_row the kernel
    also snapshots the prefix row into bank[:, nvb, :] (bit-exact, zero extra
    reads — the row streams through the score pass anyway)."""
    from .residual_tma import attn_res_fused_tma

    # The kernel applies one eps to both the score norm and the output norm.
    assert score_norm.variance_epsilon == out_norm.variance_epsilon

    cw = get_cw(score_proj, score_norm, dtype=torch.bfloat16)
    out = torch.empty_like(prefix_sum)
    attn_res_fused_tma(
        prefix_sum,
        bank,
        cw,
        out_norm.weight,
        out,
        nvb,
        score_norm.variance_epsilon,
        write_prefix=write_bank_row,
    )
    return out


@triton.jit
def _score_kernel(
    prefix_ptr,  # [T, H]
    bank_ptr,  # [T, NB_total, H]
    cw_ptr,  # [H] fp32
    scores_ptr,  # [T, MAX_ROWS] fp32
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One CTA per (token, row): scan H, output one scalar score."""
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    if j > NVB:
        return
    sumsq = 0.0
    dotv = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(tl.float32)
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        cw = tl.load(cw_ptr + offs_h)
        sumsq += tl.sum(v * v)
        dotv += tl.sum(v * cw)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + j, dotv * rrms)


@triton.jit
def _combine_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,  # [T, MAX_ROWS] fp32
    out_ptr,  # [T, H]
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """One CTA per (token, H-chunk): softmax(scores) → weighted sum → write chunk.

    Softmax is redundantly computed by each H-chunk CTA (≤16 elements, trivial).
    This gives full H-parallelism: 7 CTAs for H=7168/1024.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h0 = pid_h * BLOCK_H

    # Softmax (redundant per chunk, 16 fp32 ops)
    offs_b = tl.arange(0, MAX_ROWS)
    mask_b = offs_b <= NVB
    raw = tl.load(scores_ptr + pid_t * stride_sm + offs_b, mask=mask_b, other=float("-inf"))
    m = tl.max(raw, axis=0)
    e = tl.where(mask_b, tl.exp(raw - m), 0.0)
    p = e / tl.sum(e, axis=0)

    # Weighted sum for this H chunk
    offs_h = h0 + tl.arange(0, BLOCK_H)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for j in range(0, NVB + 1):
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(tl.float32)
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
        acc += p_j * v
    tl.store(
        out_ptr + pid_t * stride_om + offs_h,
        acc.to(out_ptr.dtype.element_ty),
    )


def _mix_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: object,
    score_norm: object,
) -> torch.Tensor:
    """Triton score + combine pair: returns the pre-norm mixture."""
    T, H = prefix_sum.shape
    if T == 0:
        return prefix_sum
    cw = get_cw(score_proj, score_norm)
    n_h_blocks = H // _BLOCK_H

    # Step 1: score each row (2D grid, full row-parallelism)
    scores = torch.empty((T, _MAX_ROWS), dtype=torch.float32, device=prefix_sum.device)
    _score_kernel[(T, nvb + 1)](
        prefix_sum,
        bank,
        cw,
        scores,
        nvb,
        score_norm.variance_epsilon,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=H,
        BLOCK_H=_BLOCK_H,
        num_warps=8,
    )

    # Step 2: softmax + weighted sum (2D grid, full H-parallelism)
    out = torch.empty_like(prefix_sum)
    _combine_kernel[(T, n_h_blocks)](
        prefix_sum,
        bank,
        scores,
        out,
        nvb,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )
    return out


def _aggregate_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: object,
    score_norm: object,
    out_norm: object,
) -> torch.Tensor:
    # Step 3: standard RMSNorm (sglang's optimized kernel)
    return out_norm(_mix_fused(prefix_sum, bank, nvb, score_proj, score_norm))


def aggregate_stream_torch(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: object,
    score_norm: object,
) -> torch.Tensor:
    """Eager reference for aggregate_stream (materializes [T, R, H])."""
    if nvb == 0:
        return prefix_sum
    T, H = prefix_sum.shape
    # rows = [bank[0..nvb-1], prefix_sum]  shape [T, nvb+1, H]
    rows = torch.cat([bank[:, :nvb, :], prefix_sum.unsqueeze(1)], dim=1)
    R = nvb + 1
    normed = score_norm(rows.reshape(T * R, H))
    scores = score_proj(normed)[0].reshape(T, R)
    probs = torch.softmax(scores.float(), dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return mixed.to(prefix_sum.dtype)


def aggregate_stream(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: object,
    score_norm: object,
) -> torch.Tensor:
    """Pre-norm aggregated stream value (softmax mixture, no output norm):
    the K3 analogue of the residual stream, for dspark aux capture -- the
    raw wire only carries the current block's running prefix."""
    if nvb == 0:
        return prefix_sum
    if prefix_sum.shape[1] % _BLOCK_H != 0:
        return aggregate_stream_torch(prefix_sum, bank, nvb, score_proj, score_norm)
    return _mix_fused(prefix_sum, bank, nvb, score_proj, score_norm)


def _aggregate(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: object,
    score_norm: object,
    out_norm: object,
    write_bank_row: bool = False,
) -> torch.Tensor:
    """Single aggregation point: score → softmax → mix → norm.

    Caller handles nvb == 0 (layer 0 attn side: just out_norm(prefix_sum)).
    write_bank_row is fast-path only (in-kernel snapshot of the prefix row
    into bank[:, nvb, :]); the triton path keeps the standalone .write() copy —
    the caller (AttnResidual.forward) owns that fallback.
    """
    if prefix_sum.shape[0] == 0:
        return prefix_sum
    if _use_fast(prefix_sum.shape[1]):
        return _aggregate_fast(
            prefix_sum,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )
    assert not write_bank_row, "fused bank write is fast-path only"
    return _aggregate_fused(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)
