"""Interleaved KV ownership and one packed attention output/LSE exchange."""

import torch
import torch.distributed as dist

from .fi_dcp import AttentionExchangeSpec, FlashInferDCPExchange, native_candidate
from .impl import DistributedCommunicator, TorchDistributedImpl
from .layout import get_parallel_layout


class _ContextGroup(TorchDistributedImpl):
    def __init__(self, group, *, members, tp_members, cpu_group, device):
        super().__init__(group, owns_group=True)
        self.members, self.tp_members = tuple(members), tuple(tp_members)
        self.cpu_group, self.device = cpu_group, device  # Borrow TP Gloo.
        self.spec = self.transport = None
        self.prepared = self.closed = False

    def prepare_exchange(self, spec, *, enabled=True):
        if self.closed or self.prepared:
            raise RuntimeError("Context exchange preparation is once per worker group")
        if (
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("Prepare context exchange before graph capture")
        candidate, error = None, None
        try:
            signature = spec.signature() if spec is not None else None
            if enabled and self.cpu_group is not None:
                candidate = native_candidate(
                    spec, self.device, tp_size=len(self.tp_members), cp_size=len(self.members)
                )
        except Exception as exc:
            signature, error = None, f"{type(exc).__name__}: {exc}"
        local = (self.members, signature, candidate is not None, error)
        rows = [local]
        if self.cpu_group is not None:
            rows = [None] * len(self.tp_members)
            dist.all_gather_object(rows, local, group=self.cpu_group)
            rows = [row for rank, row in zip(self.tp_members, rows) if rank in self.members]
        if any(row[0] != self.members or row[1] != signature or row[3] for row in rows):
            raise RuntimeError("Context exchange startup agreement failed")
        self.prepared = True
        if not all(row[2] for row in rows):
            return False
        transport = FlashInferDCPExchange(
            candidate,
            self.group,
            rank=dist.get_rank(self.group),
            size=len(self.members),
            device=self.device,
        )
        self.spec, self.transport = spec, transport
        return True

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.transport is not None:
            self.transport.close()  # Logical close; FI memory lasts for the worker.
        super().close()  # Own CP device PG only, never borrowed TP Gloo.


def init_context_parallel(*, backend=None, device: torch.device | None = None, cpu_group=None):
    """Every TP rank creates subgroups in the same order, once at startup."""
    layout = get_parallel_layout()
    previous = DistributedCommunicator.groups.get("context")
    if isinstance(previous, _ContextGroup) and previous.transport is not None:
        raise RuntimeError("Native context resources require a fresh worker, not reinitialization")
    DistributedCommunicator.groups.pop("context", None)
    if previous is not None:
        previous.close()
    if layout.dcp_size == 1:
        return
    tp_members = list(range(layout.tensor.size))
    if cpu_group is not None:
        if str(dist.get_backend(cpu_group)).lower() != "gloo":
            raise ValueError("Context preparation requires borrowed TP Gloo")
        tp_members = dist.get_process_group_ranks(cpu_group)
        if (
            len(tp_members) != layout.tensor.size
            or tp_members[layout.tensor.rank] != dist.get_rank()
        ):
            raise ValueError("TP Gloo ordering must match ParallelLayout.tensor")
    for start in range(0, layout.tensor.size, layout.dcp_size):
        ranks = tp_members[start : start + layout.dcp_size]
        group = dist.new_group(ranks, backend=backend)
        if tp_members[layout.tensor.rank] in ranks:
            implementation = _ContextGroup(
                group, members=ranks, tp_members=tp_members, cpu_group=cpu_group, device=device
            )
            if device is not None:
                implementation.warmup(device)
            DistributedCommunicator.enable(implementation, group="context")


def merge_attention(output, lse, *, lse_base="e"):
    """Combine neutral [shards,...,value] partials, returning the input dtype.

    Producers supply O=0/LSE=-inf for empty shards. Like the current SDK, NaN
    and +inf LSE have zero weight; O itself is still multiplied, not sanitized.
    FP32 O accumulation is rounded once to the producer dtype. An all-empty
    neutral row returns zero (the current SDK clamps its denominator to one).
    """
    if lse_base not in ("e", "2"):
        raise ValueError("LSE base must be e or 2")
    if output.is_cuda:
        from minisgl.kernel.dcp import combine_partial_attention

        combined = combine_partial_attention(output, lse, is_lse_base_on_e=lse_base == "e")
        if combined is not None:
            return combined
    statistics = lse.masked_fill(torch.isnan(lse) | torch.isposinf(lse), -torch.inf)
    maximum = statistics.amax(0)
    maximum = maximum.masked_fill(torch.isneginf(maximum), 0)
    centered = statistics - maximum
    weights = centered.exp() if lse_base == "e" else centered.exp2()
    combined = (output.float() * weights[..., None]).sum(0)
    combined = combined / weights.sum(0).clamp_min(1)[..., None]
    return combined.to(output.dtype)


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

    def gather_heads(self, local: torch.Tensor, *, dim=0) -> torch.Tensor:
        """Concatenate contiguous TP head shards within this context group."""
        if self.size == 1:
            return local
        gathered = self._comm.all_gather(local.contiguous()).view(self.size, *local.shape)
        if dim == 0:
            return gathered.flatten(0, 1)
        return torch.cat(tuple(gathered.unbind(0)), dim=dim)

    def gather_interleaved(self, local: torch.Tensor, total: int) -> torch.Tensor:
        """Restore a cyclic token partition using its host-known global length."""
        if total < 0 or len(local) != max(0, (total + self.size - 1 - self.rank) // self.size):
            raise ValueError("Local token count does not match the cyclic partition")
        if self.size == 1 or total == 0:
            return local
        rows = (total + self.size - 1) // self.size
        if len(local) == rows:
            padded = local.contiguous()
        else:
            padded = local.new_zeros((rows, *local.shape[1:]))
            padded[: len(local)].copy_(local)
        joined = self._comm.all_gather(padded)
        return (
            joined.view(self.size, rows, *local.shape[1:])
            .transpose(0, 1)
            .reshape(-1, *local.shape[1:])[:total]
        )

    def prepare_exchange(self, spec: AttentionExchangeSpec | None, *, enabled=True):
        if self.size == 1:
            return False
        return self._comm._backend().prepare_exchange(spec, enabled=enabled)

    def exchange(self, output, lse, *, lse_base="e"):
        """Send each destination its normal TP head slice, then merge KV shards."""
        if lse_base not in ("e", "2"):
            raise ValueError("LSE base must be e or 2")
        if self.size == 1:
            return output
        tokens, heads, values = output.shape
        if heads % self.size or lse.shape != (tokens, heads):
            raise ValueError("DCP partial attention shapes do not match the head layout")
        owner = self._comm._backend()
        if isinstance(owner, _ContextGroup) and owner.transport is not None:
            if (
                output.dtype != owner.spec.dtype
                or values != owner.spec.value_dim
                or lse.dtype != torch.float32
                or lse_base != owner.spec.lse_base
            ):
                raise ValueError("Prepared DCP producer does not match its declared spec")
            received, received_lse = owner.transport.exchange(output, lse)
            return merge_attention(received, received_lse, lse_base=lse_base)
        # Destination is the leading axis, so one all-to-all carries O and LSE.
        packed = torch.cat((output.float(), lse[..., None]), -1)
        send = packed.view(tokens, self.size, heads // self.size, values + 1)
        send = send.transpose(0, 1).contiguous()
        received = self._comm.all_to_all(send)
        return merge_attention(received[..., :values], received[..., values], lse_base=lse_base).to(
            output.dtype
        )


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
