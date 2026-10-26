"""Multi-head latent attention over compressed KV and an optional RoPE suffix."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class MLAState(NamedTuple):
    latent: torch.Tensor
    key: torch.Tensor
    positions: torch.Tensor


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
