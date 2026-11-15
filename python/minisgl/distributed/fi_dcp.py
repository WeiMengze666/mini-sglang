"""Optional FI DCP byte transport; arithmetic stays in the existing merge."""

import importlib
from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace
from typing import Literal

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class AttentionExchangeSpec:
    dtype: torch.dtype
    value_dim: int
    lse_base: Literal["e", "2"] = "e"

    def signature(self):
        return str(self.dtype), self.value_dim, self.lse_base

    def native_layout(self):
        # FI requires 16B fields; the existing mature merge covers power-of-two D.
        return (
            self.dtype in (torch.float16, torch.bfloat16)
            and type(self.value_dim) is int
            and 0 < self.value_dim <= 1024
            and not self.value_dim & (self.value_dim - 1)
            and self.value_dim * 2 % 16 == 0
            and self.lse_base in ("e", "2")
        )


@cache
def _native_api():
    try:
        fi = importlib.import_module("flashinfer.comm.dcp_alltoall")
        mapping = importlib.import_module("flashinfer.comm.mapping")
        mnnvl = importlib.import_module("flashinfer.comm.mnnvl")
        kernels = importlib.import_module("sglang.kernels.ops.attention.dcp_kernels")
        common = importlib.import_module("sglang.srt.utils.common")
    except (ImportError, OSError):
        return None
    fields = {
        "allocate": getattr(fi, "decode_cp_a2a_allocate_mnnvl_workspace", None),
        "initialize": getattr(fi, "decode_cp_a2a_init_workspace", None),
        "exchange": getattr(fi, "decode_cp_a2a_alltoall", None),
        "Mapping": getattr(mapping, "Mapping", None),
        "MnnvlConfig": getattr(mnnvl, "MnnvlConfig", None),
        "CommBackend": getattr(mnnvl, "TorchDistBackend", None),
        "pack": getattr(kernels, "dcp_pack_a2a_send", None),
        "combine": getattr(kernels, "dcp_lse_combine_triton", None),
        "topology_supported": getattr(common, "is_fi_a2a_supported", None),
    }
    return SimpleNamespace(**fields) if all(callable(fn) for fn in fields.values()) else None


def native_candidate(spec, device, *, tp_size, cp_size):
    if spec is None or not spec.native_layout() or device is None or device.type != "cuda":
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare DCP transport before graph capture")
    api = _native_api()
    if api is None or torch.cuda.current_device() != device.index:
        return None
    if not api.topology_supported(dcp_size=cp_size, tp_size=tp_size, pp_size=1, nnodes=1):
        return None
    return api, torch.cuda.device_count()


def cp_comm_backend(group, base):
    class CPCommBackend(base):
        def Split(self, color, key):
            # The raw group already is CP-only. FI base Split creates global
            # groups from local ranks; retain this exact group and ordering.
            if color != 0 or key != self.Get_rank():
                raise ValueError("CP-only mapping changed its communicator partition")
            return self

    return CPCommBackend(group)


class FlashInferDCPExchange:
    """One workspace per CP group, retained by FI for the worker lifetime.

    Public FI has no workspace-release API. close only prohibits further use;
    it does not claim to release fabric memory or mutate FI private keepalive.
    The owning group is closed after the caller drains streams and graphs.
    """

    def __init__(self, candidate, group, *, rank, size, device):
        self.api, gpus_per_node = candidate
        self.rank, self.size, self.device = rank, size, device
        self.closed = False
        mapping = self.api.Mapping(
            world_size=size,
            rank=rank,
            gpus_per_node=gpus_per_node,
            cp_size=size,
            tp_size=1,
            pp_size=1,
        )
        backend = cp_comm_backend(group, self.api.CommBackend)
        # No error handler after entering native allocation/init: owned worker
        # supervision must terminate peers rather than retrying another operator.
        self.workspace = self.api.allocate(
            mapping, mnnvl_config=self.api.MnnvlConfig(comm_backend=backend)
        )
        self.api.initialize(self.workspace, rank, size)  # FI synchronizes its init stream.
        dist.barrier(group=group)  # Mandatory before any rank exchanges data.

    def exchange(self, output, lse):
        if self.closed:
            raise RuntimeError("DCP transport is closed")
        if (
            not output.is_cuda
            or output.device != self.device
            or lse.device != output.device
            or output.stride(-1) != 1
            or output.storage_offset() % 2
            or any(stride % 2 for stride in output.stride()[:-1])
        ):
            raise ValueError("Prepared DCP producer violated its local storage contract")
        batch, heads, dim = output.shape
        local_heads = heads // self.size
        partial = output.new_empty(batch, local_heads, self.size, dim)
        stats = torch.empty(
            (batch, local_heads, self.size, 2), device=output.device, dtype=torch.float32
        )
        self.api.pack(output, lse, partial.permute(2, 0, 1, 3), stats[..., 0].permute(2, 0, 1))
        received, received_stats = self.api.exchange(
            partial, stats, self.workspace, self.rank, self.size
        )
        return received.permute(2, 0, 1, 3), received_stats[..., 0].permute(2, 0, 1)

    def close(self):
        self.closed = True
