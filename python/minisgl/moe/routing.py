"""Stable integer routing plans for grouped expert kernels."""

import torch


def align_routes(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group local routes in their original order, padding each expert bucket.

    ``-1`` marks a remote route. Padding uses the route count as its sentinel;
    remote and unused blocks have expert ``-1``. All metadata shapes are fixed
    by input shape, so changing the routing values is CUDA-graph safe.
    """
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Routing requires a two-dimensional integer ID tensor")
    if block_size <= 0 or num_experts <= 0:
        raise ValueError("Block size and expert count must be positive")
    if not topk_ids.is_cuda and ((topk_ids < -1) | (topk_ids >= num_experts)).any():
        raise ValueError("Expert IDs must be local or -1 for remote routes")
    routes = topk_ids.numel()
    capacity = routes + min(routes, num_experts + 1) * (block_size - 1)
    blocks = (capacity + block_size - 1) // block_size
    capacity = blocks * block_size
    if capacity >= 2**31:
        raise ValueError("Routing metadata exceeds int32 indexing")
    flat = topk_ids.reshape(-1).long()
    buckets = torch.where(flat < 0, num_experts, flat)
    positions = torch.arange(routes, device=topk_ids.device)
    # Unique keys avoid depending on an implementation's equal-key ordering.
    order = torch.argsort(buckets * (routes + 1) + positions)
    ordered_buckets = buckets[order]
    boundaries = torch.searchsorted(
        ordered_buckets, torch.arange(num_experts + 2, device=topk_ids.device)
    )
    counts = boundaries[1:] - boundaries[:-1]
    padded = ((counts + block_size - 1) // block_size) * block_size
    ends = padded.cumsum(0)
    starts = ends - padded
    destinations = starts[ordered_buckets] + positions - boundaries[ordered_buckets]
    sorted_ids = torch.full((capacity,), routes, dtype=torch.int32, device=topk_ids.device)
    sorted_ids.scatter_(0, destinations, order.int())
    block_starts = torch.arange(blocks, device=topk_ids.device) * block_size
    owners = torch.searchsorted(ends, block_starts, right=True)
    owners = torch.where(owners < num_experts, owners, -1).int()
    return sorted_ids, owners, ends[-1:].int()
