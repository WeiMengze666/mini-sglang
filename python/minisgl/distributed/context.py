"""Interleaved KV ownership and one packed attention output/LSE exchange."""

import torch
import torch.distributed as dist

from .layout import get_parallel_layout

_DCP_GROUP = None


def init_context_parallel(*, backend=None):
    """Every TP rank creates subgroups in the same order, once at startup."""
    global _DCP_GROUP
    layout = get_parallel_layout()
    _DCP_GROUP = None
    if layout.dcp_size == 1:
        return
    for start in range(0, layout.size, layout.dcp_size):
        ranks = list(range(start, start + layout.dcp_size))
        group = dist.new_group(ranks, backend=backend)
        if layout.rank in ranks:
            _DCP_GROUP = group


def merge_attention(output, lse):
    """Combine [shards,...,value] partial outputs using natural-log LSE.

    A shard with no visible keys contributes zero, including rows where every
    shard is empty. Arithmetic stays FP32 until the caller's output projection.
    """
    maximum = lse.amax(0)
    # Avoid -inf - -inf only for all-empty rows. Invalid active LSE must
    # propagate instead of silently removing that shard's contribution.
    maximum = maximum.masked_fill(torch.isneginf(maximum), 0)
    weights = (lse - maximum).exp()
    weights = weights / weights.sum(0).clamp_min(torch.finfo(weights.dtype).tiny)
    # Empty softmax rows may have undefined outputs. Zero weight alone cannot
    # discard them: IEEE arithmetic keeps NaN * 0 as NaN.
    partial = output.float().masked_fill(torch.isneginf(lse)[..., None], 0)
    return (partial * weights[..., None]).sum(0)


class ContextParallel:
    def __init__(self):
        layout = get_parallel_layout()
        self.rank, self.size = layout.dcp_rank, layout.dcp_size

    def owns(self, positions):
        return positions.remainder(self.size) == self.rank

    def exchange(self, output, lse):
        """Send each destination its normal TP head slice, then merge KV shards."""
        if self.size == 1:
            return output
        if _DCP_GROUP is None:
            raise RuntimeError("DCP communication must be initialized before attention")
        tokens, heads, values = output.shape
        if heads % self.size or lse.shape != (tokens, heads):
            raise ValueError("DCP partial attention shapes do not match the head layout")
        # Destination is the leading axis, so one all-to-all carries O and LSE.
        packed = torch.cat((output.float(), lse[..., None]), -1)
        send = packed.view(tokens, self.size, heads // self.size, values + 1)
        send = send.transpose(0, 1).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=_DCP_GROUP)
        return merge_attention(received[..., :values], received[..., values])
