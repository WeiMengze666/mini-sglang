"""Expert/inner-tensor groups and cyclic source-row reconstruction."""

import torch
import torch.distributed as dist

from .impl import DistributedCommunicator, TorchDistributedImpl


class ExpertParallel:
    """Prepared row communication for rank = tensor_rank * EP + expert_rank.

    The caller completes dispatch/combine before reduce_source, so corresponding
    rows have the same source-token order across the expert-tensor group. Drop
    captured graphs and native transport views/owners before close; borrowed TP
    and DCP resources are never destroyed here. Native failures require worker
    abort, not collective cleanup or another communication route.
    """

    @classmethod
    def prepare(cls, layout, tp_cpu_group, device):
        if layout.ep_size == 1:
            return None
        device = torch.device(device)
        if str(dist.get_backend(tp_cpu_group)).lower() != "gloo":
            raise ValueError("Expert preparation requires TP Gloo bootstrap")
        members = tuple(dist.get_process_group_ranks(tp_cpu_group))
        error = None
        if (
            len(members) != layout.tensor.size
            or members[layout.tensor.rank] != dist.get_rank()
            or device.type not in ("cpu", "cuda")
            or (device.type == "cuda" and device.index is None)
        ):
            error = "Expert bootstrap ordering/device differs from its TP layout"
        rows = [None] * len(members)
        signature = (layout.tensor.size, layout.ep_size, device.type)
        dist.all_gather_object(rows, (signature, error), group=tp_cpu_group)
        if any(row[0] != signature or row[1] for row in rows):
            raise ValueError("Expert parallel preparation disagreed across TP ranks")
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare expert groups before CUDA graph capture")
        return cls(layout, tp_cpu_group, device, members)

    def __init__(self, layout, tp_cpu_group, device, members):
        self.info, self.tensor_info = layout.experts, layout.expert_tensor
        self.device, self.closed, self.failed = device, False, False
        ep = self.info.size
        self.ep_members = members[self.tensor_info.rank * ep : (self.tensor_info.rank + 1) * ep]
        self.etp_members = members[self.info.rank :: ep]
        self._owned = []
        self._reduce = None
        if ep == layout.tensor.size:
            self.cpu_group = tp_cpu_group
            self._gather = DistributedCommunicator()._backend()  # Borrow the prepared TP provider.
        else:
            # All TP members create groups in this fixed order. Other TP replicas
            # need not enter these local-group rendezvous.
            for start in range(0, len(members), ep):
                ranks = members[start : start + ep]
                group = dist.new_group(ranks, backend="gloo", use_local_synchronization=True)
                if dist.get_rank() in ranks:
                    resource = TorchDistributedImpl(group, owns_group=True)
                    self._owned.append(resource)
                    self.cpu_group = group
                    if device.type == "cpu":
                        self._gather = resource
            if device.type == "cuda":
                for start in range(0, len(members), ep):
                    ranks = members[start : start + ep]
                    group = dist.new_group(ranks, backend="nccl", use_local_synchronization=True)
                    if dist.get_rank() in ranks:
                        self._gather = TorchDistributedImpl(group, owns_group=True)
                        self._owned.append(self._gather)
        if self.tensor_info.size > 1:
            for expert_rank in range(ep):
                ranks = members[expert_rank::ep]
                group = dist.new_group(
                    ranks,
                    backend="nccl" if device.type == "cuda" else "gloo",
                    use_local_synchronization=True,
                )
                if dist.get_rank() in ranks:
                    self._reduce = TorchDistributedImpl(group, owns_group=True)
                    self._owned.append(self._reduce)
        # NCCL allocation must precede cache budgeting. Borrowed TP was warmed
        # by its owner; CPU Gloo has no CUDA initialization or warmup.
        if ep < layout.tensor.size:
            self._gather.warmup(device)
        if self._reduce is not None:
            self._reduce.warmup(device)

    def source_counts(self, total):
        """Host-known cyclic rows, including physical graph-padding tokens."""
        if type(total) is not int or total < 0:
            raise ValueError("Expert source token count must be a nonnegative integer")
        return tuple(
            (total + self.info.size - 1 - rank) // self.info.size for rank in range(self.info.size)
        )

    def _check(self, value):
        if self.closed or self.failed:
            raise RuntimeError("Expert communication is closed or failed; it cannot be retried")
        if (
            value.ndim != 2
            or value.device != self.device
            or not value.is_contiguous()
            or value.requires_grad
        ):
            self.failed = True
            raise ValueError(
                "Expert source rows must be contiguous inference matrices on their prepared device"
            )

    def reduce_source(self, value):
        """Sum corresponding source-owner rows across ETP without a dtype cast."""
        self._check(value)
        if self._reduce is None or value.numel() == 0:
            return value
        try:
            return self._reduce.all_reduce(value)
        except BaseException:
            self.failed = True
            raise

    def gather_source(self, value, total):
        """Padded EP gather and inverse stripes, preserving input values/dtype."""
        self._check(value)
        counts = self.source_counts(total)
        if len(value) != counts[self.info.rank]:
            self.failed = True
            raise ValueError("Expert local rows differ from the shared cyclic token plan")
        if total == 0:
            return value
        rows = max(counts)
        padded = value.new_zeros((rows, value.shape[1]))
        padded[: len(value)].copy_(value)
        try:
            joined = self._gather.all_gather(padded)
            return (
                joined.view(self.info.size, rows, value.shape[1])
                .transpose(0, 1)
                .reshape(-1, value.shape[1])[:total]
            )
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        if self.failed:
            raise RuntimeError(
                "Failed expert communication requires worker abort, not collective cleanup"
            )
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Release expert graphs before closing groups")
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            for resource in reversed(self._owned):
                resource.close()
            self.closed = True
        except BaseException:
            self.failed = True
            raise
