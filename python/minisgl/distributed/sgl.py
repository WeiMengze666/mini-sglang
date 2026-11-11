"""Standalone bootstrap adapter for SGLang's native custom all-reduce v2."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
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
    """Instance-bound native capabilities; no SDK World or global comm map."""

    def __init__(self, comm, *, device, gemm_module=None, gemm_shape=None, **kwargs):
        super().__init__(comm, **kwargs)
        self._device = device
        self._gemm_module = (
            gemm_module
            if not getattr(comm, "disabled", False)
            and getattr(comm, "has_multicast", False)
            and comm.world_size == 8
            else None
        )
        self._gemm_shape = gemm_shape
        if self._gemm_module is not None:
            # These finalized fields allocate the actual PushPlane slots/counters.
            self._gemm_slot_bytes = comm.max_push_size
            self._gemm_push_blocks = comm.config.num_push_blocks

    def try_linear_all_gather_add(self, x, weight, addend) -> torch.Tensor | None:
        if self._gemm_module is None or getattr(self.comm, "disabled", False):
            return None
        inputs, outputs, max_rows = self._gemm_shape
        if (
            x.device != self._device
            or weight.device != x.device
            or addend.device != x.device
            or any(t.dtype != torch.bfloat16 for t in (x, weight, addend))
            or x.ndim != 2
            or not 1 <= x.shape[0] <= max_rows
            or x.shape[1] != inputs
            or weight.shape != (outputs, inputs)
            or addend.shape != (x.shape[0], outputs)
            or any(t.requires_grad for t in (x, weight, addend))
            or x.shape[0] * (outputs // 8) * 2 > self._gemm_slot_bytes
            or (x.shape[0] * outputs + 1023) // 1024 + 1 > self._gemm_push_blocks
        ):
            return None
        # Large weights are normalized by Linear after load, never here. Layout
        # violations are caller errors, not rank-local collective fallbacks.
        if not weight.is_contiguous() or weight.data_ptr() % 32:
            raise ValueError("Prepare replicated dense weight storage before GEMM-all-gather")
        if not x.is_contiguous() or x.data_ptr() % 32:
            x = x.clone(memory_format=torch.contiguous_format)
        if not addend.is_contiguous() or addend.data_ptr() % 16:
            addend = addend.clone(memory_format=torch.contiguous_format)
        output = x.new_empty((x.shape[0], outputs))
        with self.execution_context():
            self._gemm_module.run(self.comm.obj, x, weight, addend, None, output)
        return output

    def close(self):
        super().close()
        self._gemm_module = None


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
    gemm_module, gemm_shape = None, None
    if tp_info.size == 8 and capability in ((10, 0), (10, 3)):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native communication kernels must be prepared before capture")
        module = importlib.import_module("sglang.kernels.ops.communication.gemm_ag")
        # A failed JIT must not leave a native communicator needing barrier cleanup.
        gemm_module = module._jit_module()
        gemm_shape = (module.K, module.N, module.MAX_TOKENS)
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
        gemm_module=gemm_module,
        gemm_shape=gemm_shape,
        group=device_group,
        execution_context=use_torch_stream,
        input_alignment=alignment,
    )
