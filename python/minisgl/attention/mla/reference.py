"""Multi-head latent attention over compressed KV and an optional RoPE suffix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F


class MLAState(NamedTuple):
    latent: torch.Tensor
    key: torch.Tensor
    positions: torch.Tensor


class MLAOutput(NamedTuple):
    """TP-local readout when LSE is absent; otherwise a partial over cyclic KV.

    A complete prefill readout needs no context exchange even when DCP is enabled.
    """

    values: torch.Tensor
    lse: torch.Tensor | None = None


class MLALatentOutput(MLAOutput):
    """Compressed readout; merge KV shards before the layer's value projection."""


class CausalLayout(NamedTuple):
    """Host-known geometry of contiguous queries and regularly strided keys."""

    query_start: int
    key_start: int = 0
    key_stride: int = 1


@dataclass
class MLAPagedMetadata:
    """Physical-page views shared by library plans, prepared once per batch.

    CSR serves FlashAttention plans; dense views serve native DCP decode.
    Lengths retained on CPU are used only by host-planned library fallbacks.
    """

    indptr: torch.Tensor  # CPU page counts
    indices: torch.Tensor  # device physical page IDs
    lengths: torch.Tensor  # CPU local token counts, including the new token
    page_size: int
    nonempty: torch.Tensor | None = None
    capture_size: int | None = None
    block_table: torch.Tensor | None = None
    local_lengths: torch.Tensor | None = None
    global_lengths: torch.Tensor | None = None
    max_seq_len: int = 0
    cp_size: int = 1
    cp_rank: int = 0


def apply_rotary(x, positions, theta=10000.0, *, interleaved=True):
    """Rotate a [T,...,R] suffix; NoPE callers leave it unchanged."""
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
    q,
    latent,
    key,
    kv_b_weight,
    q_positions,
    kv_positions,
    *,
    nope_dim,
    value_dim,
    scale=None,
    return_lse=False,
    causal_layout: CausalLayout | None = None,
):
    """Causal expanded MLA over low-rank cache; key is shared across heads.

    The key suffix can be rotated (regular MLA) or unrotated (K3 NoPE).
    Position IDs are used for masking independently of positional encoding.
    Native backends may use the equivalent host layout to avoid GPU readbacks.
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
    output = torch.einsum("hts,shv->thv", probabilities, values.float())
    if return_lse:
        return output, scores.logsumexp(-1).T.contiguous()
    return output.to(q.dtype)
