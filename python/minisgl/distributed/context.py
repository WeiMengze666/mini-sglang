"""Interleaved KV ownership and one packed attention output/LSE exchange."""

import torch
import torch.distributed as dist

from .impl import DistributedCommunicator, TorchDistributedImpl
from .layout import get_parallel_layout


def init_context_parallel(*, backend=None, device: torch.device | None = None):
    """Every TP rank creates subgroups in the same order, once at startup."""
    layout = get_parallel_layout()
    previous = DistributedCommunicator.groups.pop("context", None)
    if previous is not None:
        previous.close()
    if layout.dcp_size == 1:
        return
    for start in range(0, layout.tensor.size, layout.dcp_size):
        ranks = list(range(start, start + layout.dcp_size))
        group = dist.new_group(ranks, backend=backend)
        if layout.tensor.rank in ranks:
            implementation = TorchDistributedImpl(group)
            if device is not None:
                implementation.warmup(device)
            DistributedCommunicator.enable(implementation, group="context")


def merge_attention(output, lse):
    """Combine [shards,...,value] partial outputs using natural-log LSE.

    A shard with no visible keys contributes zero, including rows where every
    shard is empty. Arithmetic stays FP32 until the caller's output projection.
    """
    if output.is_cuda:
        from minisgl.kernel.dcp import combine_partial_attention

        combined = combine_partial_attention(output, lse)
        if combined is not None:
            return combined
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
        self.group = layout.context
        self._comm = DistributedCommunicator("context")

    @property
    def rank(self):
        return self.group.rank

    @property
    def size(self):
        return self.group.size

    def owns(self, positions):
        return positions.remainder(self.size) == self.rank

    def gather_interleaved(self, local: torch.Tensor, total: int) -> torch.Tensor:
        """Restore a cyclic token partition using its host-known global length."""
        if total < 0 or len(local) != max(0, (total + self.size - 1 - self.rank) // self.size):
            raise ValueError("Local token count does not match the cyclic partition")
        if self.size == 1 or total == 0:
            return local
        rows = (total + self.size - 1) // self.size
        padded = local.new_zeros((rows, *local.shape[1:]))
        padded[: len(local)].copy_(local)
        joined = self._comm.all_gather(padded)
        return (
            joined.view(self.size, rows, *local.shape[1:])
            .transpose(0, 1)
            .reshape(-1, *local.shape[1:])[:total]
        )

    def exchange(self, output, lse):
        """Send each destination its normal TP head slice, then merge KV shards."""
        if self.size == 1:
            return output
        tokens, heads, values = output.shape
        if heads % self.size or lse.shape != (tokens, heads):
            raise ValueError("DCP partial attention shapes do not match the head layout")
        # Destination is the leading axis, so one all-to-all carries O and LSE.
        packed = torch.cat((output.float(), lse[..., None]), -1)
        send = packed.view(tokens, self.size, heads // self.size, values + 1)
        send = send.transpose(0, 1).contiguous()
        received = self._comm.all_to_all(send)
        return merge_attention(received[..., :values], received[..., values])


def create_expert_cpu_group(layout, tp_cpu_group):
    """Create a startup-only Gloo resource for the existing expert-TP layout.

    Every TP member calls this once in the same order. The caller closes the
    returned TorchDistributedImpl before destroying the TP bootstrap group.
    Single-rank expert execution has no timing collective and returns None.
    """
    if layout.expert_tensor.size == 1:
        return None
    if not dist.is_initialized() or str(dist.get_backend(tp_cpu_group)).lower() != "gloo":
        raise ValueError("Expert preparation requires an initialized TP Gloo group")
    ranks = dist.get_process_group_ranks(tp_cpu_group)
    if len(ranks) != layout.tensor.size or ranks[layout.tensor.rank] != dist.get_rank():
        raise ValueError("TP CPU group ordering must match ParallelLayout.tensor")
    if layout.ep_size == 1:
        return TorchDistributedImpl(tp_cpu_group)  # Borrow; never destroy the bootstrap.
    resource = None
    try:
        for expert_rank in range(layout.ep_size):
            members = ranks[expert_rank :: layout.ep_size]
            # Non-TP ranks need not participate; groups in this loop are disjoint.
            group = dist.new_group(members, backend="gloo", use_local_synchronization=True)
            if layout.experts.rank == expert_rank:
                resource = TorchDistributedImpl(group, owns_group=True)
        return resource
    except BaseException:
        if resource is not None:
            resource.close()
        raise
