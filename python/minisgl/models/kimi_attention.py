"""Small eager reference operations for Kimi attention, with explicit request state.

Only PyTorch is imported: the numerical implementation also runs on CPU. KDA
states are [head, key, value]; convolution history is [channel, kernel - 1].
The MLA cache stores normalized latent vectors and the shared key suffix, never
expanded per-head keys/values. Expansion below is temporary, for correctness.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class KDAState(NamedTuple):
    conv: torch.Tensor
    recurrent: torch.Tensor


class MLAState(NamedTuple):
    latent: torch.Tensor
    key: torch.Tensor
    positions: torch.Tensor


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


def apply_rotary(x, positions, theta=10000.0, *, interleaved=True):
    """Rotate a [T,...,R] suffix only; Kimi-K3 NoPE deliberately skips this."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("Rotary dimension must be even")
    freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angle = positions.float()[:, None] * freq[None, :]
    angle = angle.reshape(x.shape[0], *([1] * (x.ndim - 2)), dim // 2)
    cos, sin = angle.cos(), angle.sin()
    if interleaved:
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2).to(x.dtype)
    a, b = x.float().chunk(2, -1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype)


def mla_attention(
    q, latent, key, kv_b_weight, q_positions, kv_positions, *, nope_dim, value_dim, scale=None
):
    """Causal expanded MLA over low-rank cache; key is shared across heads.

    The key suffix can be rotated (regular MLA) or unrotated (K3 NoPE).
    Position IDs are used for masking independently of positional encoding.
    """
    tokens, heads, qdim = q.shape
    if latent.shape[0] != key.shape[0] or key.shape[1] != qdim - nope_dim:
        raise ValueError("MLA latent/key cache dimensions do not match query")
    if kv_b_weight.shape != (heads * (nope_dim + value_dim), latent.shape[-1]):
        raise ValueError("MLA kv_b projection dimensions do not match")
    expanded = F.linear(latent, kv_b_weight).reshape(-1, heads, nope_dim + value_dim)
    keys, values = expanded.split((nope_dim, value_dim), dim=-1)
    scores = torch.einsum("thd,shd->hts", q[..., :nope_dim].float(), keys.float())
    scores += torch.einsum("thr,sr->hts", q[..., nope_dim:].float(), key.float())
    scores *= qdim**-0.5 if scale is None else scale
    mask = kv_positions[None, :] <= q_positions[:, None]
    scores.masked_fill_(~mask[None, :, :], float("-inf"))
    probabilities = scores.softmax(-1)
    # Empty attention rows are never valid for the model but remain well defined.
    probabilities = torch.nan_to_num(probabilities)
    return torch.einsum("hts,shv->thv", probabilities, values.float()).to(q.dtype)
