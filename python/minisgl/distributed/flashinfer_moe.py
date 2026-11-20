"""Instance-owned resources for FlashInfer's functional MoE byte transport."""

import importlib
from functools import cache
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .info import DistributedInfo
from .topology import supports_flashinfer_alltoall


@cache
def _native_api():
    modules = []
    for name in (
        "flashinfer.comm.trtllm_moe_alltoall",
        "flashinfer.comm.mnnvl",
        "flashinfer.comm.dlpack_utils",
        "flashinfer.comm.comm_backend",
        "tvm_ffi",
    ):
        try:
            modules.append(importlib.import_module(name))
        except ModuleNotFoundError as error:
            if error.name and (name == error.name or name.startswith(error.name + ".")):
                return None
            raise
    ops, memory, dlpack, backend, stream = modules
    fields = {
        "size": getattr(ops, "moe_a2a_get_workspace_size_per_rank", None),
        "initialize": getattr(ops, "moe_a2a_initialize", None),
        "dispatch": getattr(ops, "moe_a2a_dispatch", None),
        "combine": getattr(ops, "moe_a2a_combine", None),
        "sanitize": getattr(ops, "moe_a2a_sanitize_expert_ids", None),
        "memory": getattr(memory, "SymmDeviceMemory", None),
        "pack": getattr(dlpack, "pack_strided_memory", None),
        "backend": getattr(backend, "TorchDistBackend", None),
        "supported": supports_flashinfer_alltoall,
        "stream": getattr(stream, "use_torch_stream", None),
    }
    return SimpleNamespace(**fields) if all(callable(fn) for fn in fields.values()) else None


class FlashInferMoEWorkspace:
    """One prepared group and execution stream; no FI global caches or comm changes.

    Allocate before cache budgeting. close drains work and prohibits reuse, but
    keeps the mature allocation owner alive for any retained views/graphs. Drop
    those references and this owner before destroying its borrowed CPU group.
    Native failures poison the resource; worker supervision must abort peers.
    """

    @classmethod
    def prepare(
        cls,
        info: DistributedInfo,
        group,
        device,
        *,
        capacity,
        dispatch_bytes,
        combine_bytes,
        layout,
        error=None,
    ):
        if info.size == 1:
            return None
        if str(dist.get_backend(group)).lower() != "gloo":
            raise ValueError("MoE transport preparation requires its CPU Gloo group")
        device = torch.device(device)
        members = tuple(dist.get_process_group_ranks(group))
        if (dist.get_rank(group), len(members)) != (info.rank, info.size):
            error = error or "MoE transport group differs from its logical coordinates"
        if any(
            type(value) is not int or value <= 0
            for value in (capacity, dispatch_bytes, combine_bytes)
        ):
            error = error or "MoE transport capacities must be positive integers"
        api = None
        supported = False
        if error is None and device.type == "cuda" and device.index is not None:
            if torch.cuda.is_current_stream_capturing():
                error = "Prepare MoE transport before graph capture"
            else:
                try:
                    api = _native_api()
                    supported = bool(
                        api is not None
                        and torch.cuda.current_device() == device.index
                        and api.supported(
                            dcp_size=info.size, tp_size=info.size, pp_size=1, nnodes=1
                        )
                    )
                except Exception as failure:
                    error = f"MoE capability preparation failed: {failure}"
        signature = (members, capacity, dispatch_bytes, combine_bytes, layout)
        rows = [None] * info.size
        dist.all_gather_object(rows, (signature, error, supported), group=group)
        errors = [row[1] for row in rows if row[1] is not None]
        if errors or any(row[0] != signature for row in rows):
            raise ValueError(f"MoE transport preparation disagreed: {errors or 'layout mismatch'}")
        if not all(row[2] for row in rows):
            return None
        # After collective agreement, allocation/init errors are fatal, never retried.
        return cls(api, info, group, device, capacity, dispatch_bytes, combine_bytes)

    def __init__(self, api, info, group, device, capacity, dispatch_bytes, combine_bytes):
        self.api, self.info, self.group, self.device = api, info, group, device
        self.capacity = capacity
        self.closed = self.failed = self.busy = False
        self.stream = torch.cuda.current_stream(device)
        self.views = {}
        size = api.size(info.size, capacity, dispatch_bytes, combine_bytes, backend="trtllm")
        self.memory = api.memory(
            buf_size=size,
            group_size=info.size,
            group_rank=info.rank,
            device_idx=device.index,
            comm_backend_for_handle_transfer=api.backend(group),
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        stride = self.memory.get_allocation_size()
        pointers = self.memory.get_buffer_ptrs_host()
        if (
            not self.memory.mapped
            or stride < size
            or pointers != [pointers[0] + i * stride for i in range(info.size)]
        ):
            raise RuntimeError("Native MoE memory does not provide the required peer row stride")
        self.workspace = api.pack(pointers[0], size, stride, info.size, torch.uint8, device.index)
        if (
            self.workspace.shape != (info.size, size)
            or self.workspace.dtype != torch.uint8
            or self.workspace.device != device
            or self.workspace.stride() != (stride, 1)
        ):
            raise RuntimeError("Native MoE workspace violated its strided byte ABI")
        # The DLPack view's deleter does not own the native allocation.
        self.workspace._moe_memory_owner = self.memory
        with api.stream():
            self.metainfo = api.initialize(
                self.workspace, info.rank, info.size, capacity, backend="trtllm"
            )
        if (
            self.metainfo.device.type != "cpu"
            or self.metainfo.dtype != torch.int64
            or self.metainfo.ndim != 1
        ):
            raise RuntimeError("Native MoE initialization must return CPU int64 metainfo")
        dist.barrier(group=group)

    def check(self):
        if self.closed or self.failed:
            raise RuntimeError("MoE transport is closed or failed; it cannot be retried")
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("A MoE workspace belongs to one execution stream/channel")

    def close(self):
        if self.closed:
            return
        if self.failed:
            raise RuntimeError("Failed MoE transport requires worker abort, not collective cleanup")
        if self.busy or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Finish MoE handles and graphs before closing transport")
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.views.clear()
        self.closed = True
