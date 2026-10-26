"""Causal convolution and delta attention with explicit FP32 recurrent state."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class KDAState(NamedTuple):
    conv: torch.Tensor
    recurrent: torch.Tensor


def causal_conv1d(x, weight, state=None):
    """Depthwise causal convolution + SiLU, preserving *unactivated* history."""
    weight = weight.reshape(weight.shape[0], -1)
    channels, width = weight.shape
    if x.ndim != 2 or x.shape[1] != channels or width < 1:
        raise ValueError("Expected x[T,C] and depthwise weight[C,1,K] or [C,K]")
    if state is None:
        state = x.new_zeros(channels, width - 1)
    if state.shape != (channels, width - 1):
        raise ValueError("Convolution history has incorrect shape")
    history = torch.cat((state.to(x), x.T), dim=-1)
    if x.shape[0] == 0:
        return x.clone(), state.clone()
    windows = history.unfold(-1, width, 1)
    y = (windows.float() * weight.float()[:, None, :]).sum(-1).T
    tail = history[:, -(width - 1) :].clone() if width > 1 else history[:, :0].clone()
    return F.silu(y).to(x.dtype), tail


def kda_log_decay(forget_gate, A_log, dt_bias, lower_bound=None):
    """Safe KDA gate is LB * sigmoid(exp(A) * (f + bias)), not a clamp."""
    heads, dim = forget_gate.shape[-2:]
    a = A_log.reshape(-1)[:heads].float().exp().reshape(heads, 1)
    f = forget_gate.float() + dt_bias.float().reshape(heads, dim)
    if lower_bound is not None:
        if lower_bound >= 0:
            raise ValueError("KDA gate_lower_bound must be negative")
        return lower_bound * torch.sigmoid(a * f)
    return -a * F.softplus(f)


def kda_recurrence(q, k, v, log_decay, beta, state=None, *, scale=None):
    """Kimi delta update; q/k are normalized here, beta is already sigmoid'ed.

    S <- diag(exp(g)) S; S <- S + beta k (v - k^T S)^T.
    All recurrent arithmetic/state is float32, output follows v's dtype.
    """
    if q.shape != k.shape or q.shape != log_decay.shape or q.ndim != 3:
        raise ValueError("KDA q, k and log_decay must have matching [T,H,D] shapes")
    tokens, heads, dim = q.shape
    if v.shape[:2] != (tokens, heads) or beta.shape != (tokens, heads):
        raise ValueError("KDA v/beta head or token dimensions do not match")
    # FLA's L2 norm uses rsqrt(sum(x*x) + 1e-6), not RMS normalization.
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    q = q * (dim**-0.5 if scale is None else scale)
    shape = (heads, dim, v.shape[-1])
    if state is None:
        state = torch.zeros(shape, dtype=torch.float32, device=q.device)
    elif state.shape != shape:
        raise ValueError("KDA recurrent state has incorrect shape")
    else:
        state = state.float().clone()
    output = torch.empty_like(v)
    for t in range(tokens):
        state = state * log_decay[t].float().exp().unsqueeze(-1)
        prediction = (k[t].unsqueeze(-1) * state).sum(-2)
        delta = (v[t].float() - prediction) * beta[t].float().unsqueeze(-1)
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
        output[t] = (q[t].unsqueeze(-1) * state).sum(-2).to(v.dtype)
    return output, state


def gated_rms_norm(x, gate, weight, eps=1e-6):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (y * weight.float() * torch.sigmoid(gate.float())).to(x.dtype)
