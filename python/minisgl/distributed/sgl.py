"""Standalone bootstrap adapter for SGLang's native custom all-reduce v2."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from .impl import TorchDistributedImpl
from .info import DistributedInfo
from .mnnvl import MNNVLAllReduceProvider


def prepare(*, tp_info: DistributedInfo, device: torch.device) -> None:
    """Warm the SDK's real peer cache while no framework coordinator exists.

    Local workers serialize the official probe through a hardware-scoped cache.
    The native communicator later performs its normal topology checks using that
    cache. It owns every collective kernel and symmetric-memory allocation.
    """
    if tp_info.size == 1:
        return
    if device.type != "cuda":
        raise ValueError("Native MNNVL requires a CUDA device")
    if dist.is_initialized():
        raise RuntimeError("Native MNNVL preparation must precede process-group creation")
    if os.getenv("SGLANG_SKIP_P2P_CHECK", "0") == "1":
        raise ValueError(
            "Unset SGLANG_SKIP_P2P_CHECK: standalone MNNVL needs the official peer probe"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Native MNNVL requires CUDA")
    count = torch.cuda.device_count()
    if count < 1 or (device.index is not None and not 0 <= device.index < count):
        raise ValueError("Native MNNVL device is outside CUDA_VISIBLE_DEVICES")
    local_size = int(os.getenv("LOCAL_WORLD_SIZE", str(min(tp_info.size, count))))
    if count != local_size:
        raise ValueError("Limit CUDA_VISIBLE_DEVICES to this communicator's local workers")
    uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in range(count)]
    driver = Path("/proc/driver/nvidia/version")
    identity = "|".join(
        [*uuids, str(torch.version.cuda), driver.read_text() if driver.exists() else ""]
    )
    key = hashlib.sha256(identity.encode()).hexdigest()
    cache = Path(tempfile.gettempdir()) / f"minisgl-mnnvl-{os.getuid()}" / key
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    sdk = importlib.import_module(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_utils"
    )
    with (cache / "bootstrap.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = os.environ.get("SGLANG_CACHE_DIR")
        try:
            os.environ["SGLANG_CACHE_DIR"] = str(cache)
            # One call initializes the SDK's complete local peer matrix. Its
            # return value is not a group capability decision; the ctor owns it.
            sdk.gpu_p2p_access_check(0, min(1, count - 1))
        finally:
            if previous is None:
                os.environ.pop("SGLANG_CACHE_DIR", None)
            else:
                os.environ["SGLANG_CACHE_DIR"] = previous


class SGLangAllReduceProvider(MNNVLAllReduceProvider):
    """Optional native epilogues bound to this communicator, never SDK globals."""

    def __init__(self, comm, *, device, norm_module=None, **kwargs):
        super().__init__(comm, **kwargs)
        self._device = device
        self._norm_module = (
            norm_module
            if not getattr(comm, "disabled", False) and getattr(comm, "has_multicast", False)
            else None
        )
        if self._norm_module is not None:
            # The SDK allocates the actual PushPlane counter/slots from these
            # finalized fields in CustomAllReduceV2._init_workspace.
            self._norm_push_blocks = comm.config.num_push_blocks
            self._norm_push_bytes = min(512 * 1024, comm.max_push_size)

    def try_all_reduce_norm(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor | None:
        # Fixed width and cluster geometry belong to this compiled kernel,
        # not to model or MoE abstractions. Pointer/stride differences normalize
        # locally; they must never choose a different collective on peer ranks.
        if (
            self._norm_module is None
            or getattr(self.comm, "disabled", False)
            or x.device != self._device
            or weight.device != x.device
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or x.ndim != 2
            or x.shape[1] != 3584
            or weight.shape != (3584,)
            or not 0 < x.shape[0]
            or x.numel() * x.element_size() > self._norm_push_bytes
            or min(x.shape[0], 96) >= self._norm_push_blocks
            or not math.isfinite(eps)
            or eps < 0
        ):
            return None
        if not x.is_contiguous() or x.data_ptr() % self.input_alignment:
            x = x.clone(memory_format=torch.contiguous_format)
        if not weight.is_contiguous() or weight.data_ptr() % self.input_alignment:
            weight = weight.clone(memory_format=torch.contiguous_format)
        with self.execution_context():
            self._norm_module.push_norm(self.comm.obj, x.view(-1), weight, eps, x.shape[0])
        return x

    def close(self):
        super().close()
        self._norm_module = None


def create_provider(*, tp_info, cpu_group, device_group, device, max_bytes):
    """Adapt native collectives to the regular provider, with Torch fallback."""
    for group in (cpu_group, device_group):
        if (dist.get_rank(group), dist.get_world_size(group)) != (tp_info.rank, tp_info.size):
            raise ValueError("MNNVL process group differs from tensor-parallel coordinates")
    if dist.get_process_group_ranks(cpu_group) != dist.get_process_group_ranks(device_group):
        raise ValueError("MNNVL bootstrap and device groups must have the same ordered ranks")
    native = importlib.import_module(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_v2"
    )
    capability = torch.cuda.get_device_capability(device)
    norm_module = None
    if (
        tp_info.size > 1
        and capability in ((10, 0), (10, 3))
        and tp_info.size in native.get_supported_world_sizes()
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native communication kernels must be prepared before capture")
        module = importlib.import_module("sglang.kernels.ops.communication.all_reduce_residual")
        # Compile before creating the native communicator: a failed JIT must not
        # leave one rank holding resources whose close needs a group barrier.
        norm_module = module._jit_module(tp_info.size)
    TorchDistributedImpl(device_group).warmup(device)
    from tvm_ffi import use_torch_stream

    with use_torch_stream():
        comm = native.CustomAllReduceV2(cpu_group, device, max_size=max_bytes)
    # The SDK's PDL staging copy uses wider vectors than its 16-byte entry
    # check on Blackwell. Views must meet the copy's alignment on every rank.
    alignment = 32 if capability[0] >= 10 else 16
    return SGLangAllReduceProvider(
        comm,
        device=device,
        norm_module=norm_module,
        group=device_group,
        execution_context=use_torch_stream,
        input_alignment=alignment,
    )
