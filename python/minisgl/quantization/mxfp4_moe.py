"""Grouped weight-only experts over existing MXFP4 allocations.

A small pointer table avoids stacking or duplicating resident checkpoint weights.
Routing stays global; this executor returns the local FP32 mixture before any
collective or optional latent normalization. CPU/reference execution is separate.
"""

import torch
import triton as tr
import triton.language as tl
from minisgl.moe.fused import moe_align_block_size
from minisgl.layers.activation import resolve_gated_activation

from .mxfp4_kernel import _decode


@tr.jit
def _expert_ptr(table, expert, column: tl.constexpr):
    return tl.load(table + expert * 6 + column).to(tl.pointer_type(tl.uint8))


@tr.jit(do_not_specialize=["R"])
def _gate_up(
    X,
    P,
    Sorted,
    Experts,
    Length,
    A,
    R,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    if block * BM >= tl.load(Length):
        return
    expert = tl.load(Experts + block)
    if (expert < 0) | (expert >= E):
        return
    routes = tl.load(Sorted + block * BM + tl.arange(0, BM)).to(tl.int64)
    valid = (routes >= 0) & (routes < R)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    wg, sg = _expert_ptr(P, expert, 0), _expert_ptr(P, expert, 1)
    wu, su = _expert_ptr(P, expert, 2), _expert_ptr(P, expert, 3)
    for block_k in range(tr.cdiv(H, BK)):
        cols = block_k * BK + offsets
        x = tl.load(
            X + routes[:, None] // TOPK * H + cols[None, :],
            valid[:, None] & (cols[None, :] < H),
            0,
        )
        w1 = _decode(wg, sg, rows, cols, I, H).to(x.dtype)
        w3 = _decode(wu, su, rows, cols, I, H).to(x.dtype)
        gate = tl.dot(x, tl.trans(w1), gate, input_precision="tf32x3")
        up = tl.dot(x, tl.trans(w3), up, input_precision="tf32x3")
    # Preserve each Linear's output rounding before the nonlinear activation.
    gate = gate.to(X.dtype.element_ty).to(tl.float32)
    up = up.to(X.dtype.element_ty).to(tl.float32)
    if ACTIVATION == "situ":
        gate = BETA * tl.extra.cuda.libdevice.tanh(gate / BETA) * tl.sigmoid(gate)
        if LINEAR_BETA > 0:
            up = LINEAR_BETA * tl.extra.cuda.libdevice.tanh(up / LINEAR_BETA)
    else:
        gate = (gate * tl.sigmoid(gate)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(
        A + routes[:, None] * I + rows[None, :], gate * up, valid[:, None] & (rows[None, :] < I)
    )


@tr.jit(do_not_specialize=["R"])
def _down(
    A,
    P,
    Sorted,
    Experts,
    Length,
    Weights,
    Y,
    R,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    if block * BM >= tl.load(Length):
        return
    expert = tl.load(Experts + block)
    if (expert < 0) | (expert >= E):
        return
    routes = tl.load(Sorted + block * BM + tl.arange(0, BM)).to(tl.int64)
    valid = (routes >= 0) & (routes < R)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    w, s = _expert_ptr(P, expert, 4), _expert_ptr(P, expert, 5)
    for block_k in range(tr.cdiv(I, BK)):
        cols = block_k * BK + offsets
        a = tl.load(
            A + routes[:, None] * I + cols[None, :], valid[:, None] & (cols[None, :] < I), 0
        )
        weight = _decode(w, s, rows, cols, H, I).to(a.dtype)
        acc = tl.dot(a, tl.trans(weight), acc, input_precision="tf32x3")
    acc = acc.to(A.dtype.element_ty).to(tl.float32)
    mixing = tl.load(Weights + routes, valid, 0)
    tl.store(
        Y + routes[:, None] * H + rows[None, :],
        acc * mixing[:, None],
        valid[:, None] & (rows[None, :] < H),
    )


class MXFP4Experts:
    """Homogeneous, bias-free w1/w3/w2 experts with contiguous global IDs."""

    def __init__(self, experts, *, activation, beta=1.0, linear_beta=None):
        items = list(experts)
        first = items[0][1]
        self._start = items[0][0]
        self._experts = tuple(expert for _, expert in items)  # retain weight allocations
        self._hidden = first.w1._input_size
        self._intermediate = first.w1._output_size
        activation_spec = resolve_gated_activation(
            activation, beta=beta, linear_beta=linear_beta
        )
        if activation_spec.name not in ("silu", "situ"):
            raise ValueError("Unsupported packed expert activation")
        self._activation, self._beta = activation_spec.name, activation_spec.beta
        self._linear_beta = activation_spec.linear_beta or 0.0
        pointers = []
        for index, expert in items:
            if index != self._start + len(pointers):
                raise ValueError("Packed expert IDs must be contiguous")
            row = []
            for op, shape in zip(
                (expert.w1, expert.w3, expert.w2),
                ((self._intermediate, self._hidden),) * 2 + ((self._hidden, self._intermediate),),
            ):
                if (op._output_size, op._input_size) != shape or op.bias is not None:
                    raise ValueError("Packed expert projections must be homogeneous and bias-free")
                row.extend((op.weight_packed.data_ptr(), op.weight_scale.data_ptr()))
            pointers.append(row)
        self._pointers = torch.tensor(
            pointers, dtype=torch.int64, device=first.w1.weight_packed.device
        )

    def forward(self, x, weights, ids):
        hidden, intermediate = self._hidden, self._intermediate
        count, topk = ids.shape
        experts = len(self._experts)
        local = ids - self._start
        local = local.masked_fill((local < 0) | (local >= experts), -1).int().contiguous()
        bm, bn, bk = 16, 32, 64
        sorted_ids, expert_ids, length = moe_align_block_size(local, bm, experts)
        # At most one padded block tail per nonempty bucket; no host readback.
        routes = count * topk
        blocks = tr.cdiv(routes + min(experts + 1, routes) * (bm - 1), bm)
        activated = x.new_empty((routes, intermediate))
        # The alignment primitive assigns remote routes to expert -1, which both kernels skip.
        partial = torch.zeros((count, topk, hidden), device=x.device, dtype=torch.float32)
        _gate_up[(blocks, tr.cdiv(intermediate, bn))](
            x.contiguous(),
            self._pointers,
            sorted_ids,
            expert_ids,
            length,
            activated,
            routes,
            experts,
            hidden,
            intermediate,
            topk,
            self._activation,
            self._beta,
            self._linear_beta,
            bm,
            bn,
            bk,
            num_warps=4,
        )
        _down[(blocks, tr.cdiv(hidden, bn))](
            activated,
            self._pointers,
            sorted_ids,
            expert_ids,
            length,
            weights.float().contiguous(),
            partial,
            routes,
            experts,
            hidden,
            intermediate,
            bm,
            bn,
            bk,
            num_warps=4,
        )
        return partial.sum(1)
