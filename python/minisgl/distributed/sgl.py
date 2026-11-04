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


def create_provider(*, tp_info, cpu_group, device_group, device, max_bytes):
    """Adapt native collectives to the regular provider, with Torch fallback."""
    for group in (cpu_group, device_group):
        if (dist.get_rank(group), dist.get_world_size(group)) != (tp_info.rank, tp_info.size):
            raise ValueError("MNNVL process group differs from tensor-parallel coordinates")
    if dist.get_process_group_ranks(cpu_group) != dist.get_process_group_ranks(device_group):
        raise ValueError("MNNVL bootstrap and device groups must have the same ordered ranks")
    TorchDistributedImpl(device_group).warmup(device)
    native = importlib.import_module(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_v2"
    )
    from tvm_ffi import use_torch_stream

    with use_torch_stream():
        comm = native.CustomAllReduceV2(cpu_group, device, max_size=max_bytes)
    return MNNVLAllReduceProvider(comm, group=device_group, execution_context=use_torch_stream)
