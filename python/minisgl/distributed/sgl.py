"""Standalone bootstrap adapter for SGLang's native custom all-reduce v2."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import os
import tempfile
from functools import cached_property
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
    """Adapt native collectives; the SDK owns workspace sizing and fallback.

    max_bytes remains a generic factory hint, not the native workspace cap.
    """
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
        comm = native.CustomAllReduceV2(cpu_group, device)
    # The SDK's PDL staging copy uses wider vectors than its 16-byte entry
    # check on Blackwell. Views must meet the copy's alignment on every rank.
    alignment = 32 if torch.cuda.get_device_capability(device)[0] >= 10 else 16
    return SGLangMNNVLProvider(
        comm,
        group=device_group,
        execution_context=use_torch_stream,
        input_alignment=alignment,
        output_group=cpu_group,
        device=device,
    )


class SGLangMNNVLProvider(MNNVLAllReduceProvider):
    """Instance-bound SDK in-place AR + a group-replicated residual.

    Native pull is an explicit prepared route: output_key names a reservation
    and inputs must be corresponding slices on every rank. Plain large tensors
    keep generic fallback. Capability checks never inspect local addresses to
    select an algorithm, and no collective error is retried.
    """

    # The adapted native AR/res consumer writes BF16 only; decline other
    # reservations through the existing group availability agreement.
    _reserved_output_dtypes = (torch.bfloat16,)

    @cached_property
    def _residual_ops(self):
        return importlib.import_module("sglang.kernels.ops.communication.all_reduce_residual")

    @cached_property
    def _push_max_bytes(self):
        # Pure SDK selector input; do not initialize its process-global state.
        module = importlib.import_module("sglang.srt.layers.communication.k3_ar_fusion")
        return module._PUSH_MAX_BYTES

    @cached_property
    def _residual_kernel(self):
        return self._residual_ops._jit_module(self.comm.world_size)

    @cached_property
    def _linear_all_gather_ops(self):
        return importlib.import_module("sglang.kernels.ops.communication.gemm_ag")

    @cached_property
    def _linear_all_gather_kernel(self):
        return self._linear_all_gather_ops._jit_module()

    def supports_linear_all_gather(self, x, weight, addend) -> bool:
        if self._closed:
            raise RuntimeError("Cannot project on a closed communicator")
        if (
            getattr(self.comm, "disabled", False)
            or not getattr(self.comm, "has_multicast", False)
            or self.comm.world_size != 8
            or not all(
                value.is_cuda and value.dtype == torch.bfloat16 for value in (x, weight, addend)
            )
            or x.ndim != 2
            or weight.ndim != 2
            or addend.ndim != 2
        ):
            return False
        ops = self._linear_all_gather_ops
        rows = x.shape[0]
        blocks = getattr(getattr(self.comm, "config", None), "num_push_blocks", 0)
        window = self.comm.max_push_size
        # gemm_ag.cuh:25-28,249-281: 8 BF16/thread, 128 threads/block,
        # one cleanup block. Only the consumer grid is counter-limited.
        consumers = (rows * ops.N + 1023) // 1024
        return (
            0 < rows <= ops.MAX_TOKENS
            and x.shape[1] == ops.K
            and tuple(weight.shape) == (ops.N, ops.K)
            and tuple(addend.shape) == (rows, ops.N)
            and rows * (ops.N // 8) * 2 <= window
            and 0 < window * self.comm.world_size < 1 << 32
            and consumers + 1 <= blocks
        )

    def linear_all_gather(self, x, weight, addend) -> torch.Tensor:
        """Instance-bound SDK column GEMV + AG + add with replicated operands.

        The SDK slices the FULL weight inside its producer. The returned tensor
        owns fresh storage; x, weight and addend remain read-only. Local storage
        violations raise after uniform capability selection, never switching one
        rank back to a replicated GEMM. Warm the SDK module before graph capture.
        """
        if not self.supports_linear_all_gather(x, weight, addend):
            raise ValueError("Native linear all-gather does not cover these operands")
        for value, alignment in ((x, 32), (weight, 32), (addend, 16)):
            if (
                value.device != self.device
                or not value.is_contiguous()
                or value.data_ptr() % alignment
                or value.requires_grad
            ):
                raise ValueError(
                    "Native linear all-gather requires aligned contiguous local inference operands"
                )
        if (
            "_linear_all_gather_kernel" not in self.__dict__
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("Warm native linear all-gather before graph capture")
        kernel = self._linear_all_gather_kernel
        output = torch.empty_like(addend)
        with self.execution_context():
            # b is already-reduced shared; c belongs to the outer residual owner.
            kernel.run(self.comm.obj, x, weight, addend, None, output)
        return output

    def supports_all_reduce_residual(
        self, x, residual=None, *, norm=None, output_key: str | None = None, overlap: bool = False
    ):
        # The enabled SDK communicator already validates world size and
        # multicast capability; SM100/103 is only its wrapper's auto policy.
        # Only group-uniform semantics and sizes select an algorithm. Local
        # strides, alignment and multicast aliases never choose a fallback.
        if self._closed:
            raise RuntimeError("Cannot reduce on a closed communicator")
        if output_key is not None and (not isinstance(output_key, str) or not output_key):
            raise ValueError("output_key must be a nonempty reservation name or None")
        if type(overlap) is not bool:
            raise ValueError("Reduction overlap hint must be a bool")
        if overlap and norm is not None:
            return False  # Low-SM overlap currently describes the plain shared reduction.
        if norm is not None:
            from minisgl.layers.norm import RMSNormEpilogue

            if (
                not isinstance(norm, RMSNormEpilogue)
                or residual is not None
                or norm.width != 3584
                or norm.weight.dtype != torch.bfloat16
                or type(norm.num_norm_rows) is not int
                or not 0 <= norm.num_norm_rows <= x.numel() // 3584
                or x.numel() % 3584
            ):
                return False
        if (
            getattr(self.comm, "disabled", False)
            or not getattr(self.comm, "has_multicast", False)
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or x.numel() == 0
            or x.numel() % 8
        ):
            return False
        if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype):
            return False
        if not overlap and x.nbytes <= min(self._push_max_bytes, self.comm.max_push_size):
            # SDK ar_fusion.cuh:768-771 requires row_clusters < push.num_blocks.
            # At its <=512KiB push bound, rows<=73<96, so row_clusters==rows.
            return norm is None or x.numel() // 3584 < self.comm.config.num_push_blocks
        return bool(
            output_key is not None
            and getattr(self.comm, "pull_enabled", False)
            and x.numel() // 8 < 1 << 31
        )

    def all_reduce_residual(
        self, x, residual=None, *, norm=None, output_key: str | None = None, overlap: bool = False
    ):
        """Native in-place AR + residual or a prefix RMSNorm epilogue.

        overlap=True selects the SDK's low-SM pull for the plain shared
        reduction; it requires an explicit multicast key even for small inputs.
        The SDK norm kernels reduce the entire packed buffer and normalize
        only the requested leading rows. Push computes RMS statistics from
        FP32 sums before rounding its numerator; this intentionally follows
        the official fused algorithm rather than a bitwise split-op oracle.

        Kernels mutate their input in place. Push may first copy a strided or
        unaligned input and returns that copy. A reserved result is borrowed
        until the same slot is written again; callers copy retained aliases.
        The residual is group-replicated and logically read-only. Overlapping
        residual storage is snapshotted before dispatch; aliases of mutated x
        still observe its writes, so callers copy aliases they retain. Native
        pull additionally requires the same explicit output_key and slice on
        every rank; local storage violations raise before launch. First
        invocation must run during warmup, before graph capture.
        """
        if not self.supports_all_reduce_residual(
            x, residual, norm=norm, output_key=output_key, overlap=overlap
        ):
            raise ValueError("Native AR + residual does not cover this input")
        if x.device != self.device or (residual is not None and residual.device != x.device):
            raise ValueError("Native AR + residual input is on the wrong local device")
        if norm is not None:
            weight = norm.weight
            if (
                weight.device != x.device
                or weight.ndim != 1
                or not weight.is_contiguous()
                or weight.data_ptr() % 16
                or torch._C._overlaps(weight, x)
            ):
                raise ValueError(
                    "Native RMSNorm weights require separate contiguous aligned local storage"
                )
        push = not overlap and x.nbytes <= min(self._push_max_bytes, self.comm.max_push_size)
        if push:
            if not x.is_contiguous() or x.data_ptr() % 16:
                x = x.clone(memory_format=torch.contiguous_format)
            mc_ptr = None
        else:
            mc_ptr = self._output_mc_ptr(x, key=output_key)
            if not x.is_contiguous() or x.data_ptr() % 16 or mc_ptr is None:
                raise ValueError(
                    f"Native pull requires a contiguous reserved multicast alias of {output_key!r}"
                )
        if residual is not None and (
            not residual.is_contiguous()
            or residual.data_ptr() % 16
            or torch._C._overlaps(x, residual)
        ):
            residual = residual.clone(memory_format=torch.contiguous_format)
        if "_residual_kernel" not in self.__dict__ and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm native AR + residual before graph capture")
        # Exact SDK Communicator ABI, bound to this instance rather than its
        # public wrappers' world-size-global _COMM_MAP.
        kernel = self._residual_kernel
        flat = x.view(-1)
        residual = residual.view(-1) if residual is not None else None
        with self.execution_context():
            if push:
                if norm is None:
                    kernel.push_res(self.comm.obj, flat, residual)
                else:
                    kernel.push_norm(self.comm.obj, flat, norm.weight, norm.eps, norm.num_norm_rows)
            else:
                tuning = self._residual_ops._resolve_tuning(
                    self._residual_ops.RES_TUNING
                    if norm is None
                    else self._residual_ops.NORM_TUNING,
                    nbytes=x.nbytes,
                    # Official K3 shared side-stream policy (model:1413):
                    # low-SM always uses pull, even below the push threshold.
                    num_blocks=4 if overlap else None,
                    unroll=8 if overlap else None,
                )
                if norm is None:
                    kernel.pull_res(
                        self.comm.obj, flat, residual, mc_ptr, tuning.num_blocks, tuning.unroll
                    )
                else:
                    kernel.pull_norm(
                        self.comm.obj,
                        flat,
                        norm.weight,
                        norm.eps,
                        norm.num_norm_rows,
                        mc_ptr,
                        tuning.num_blocks,
                        tuning.unroll,
                    )
        return x

    def supports_weighted_all_reduce(self, out, *, norm, topk: int) -> bool:
        from minisgl.layers.norm import RMSNormEpilogue

        if self._closed:
            raise RuntimeError("Cannot reduce on a closed communicator")
        return bool(
            not getattr(self.comm, "disabled", False)
            and getattr(self.comm, "has_multicast", False)
            and out.is_cuda
            and out.dtype == torch.bfloat16
            and out.ndim == 2
            and out.shape[1] == 3584
            and type(topk) is int
            and topk == 16
            and isinstance(norm, RMSNormEpilogue)
            and norm.width == 3584
            and norm.weight.dtype == torch.bfloat16
            and type(norm.num_norm_rows) is int
            and norm.num_norm_rows == out.shape[0]
            and 0 < out.nbytes <= min(self._push_max_bytes, self.comm.max_push_size)
            # SDK ar_fusion.cuh:819-821; <=512KiB implies rows<=73<96.
            and out.shape[0] < self.comm.config.num_push_blocks
        )

    def weighted_all_reduce(self, values, indices, weights, *, out, norm):
        """Official output-only top16 finalize + push AR + RMSNorm, once.

        Selection uses group-uniform algorithm metadata. Local storage and
        native triple errors raise; an already-launched FI operation is never
        retried through another finalizer.
        """
        if (
            weights.ndim != 2
            or weights.shape != (out.shape[0], 16)
            or indices.ndim != 1
            or indices.numel() != out.shape[0] * 16
            or indices.dtype != torch.int32
            or values.ndim != 2
            or values.shape[1] != 3584
            or values.shape[0] == 0
            or values.dtype != torch.bfloat16
            or weights.dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("Native weighted reduction needs the exact BF16/top16 FI triple")
        if not self.supports_weighted_all_reduce(out, norm=norm, topk=16):
            raise ValueError("Native weighted reduction does not cover this output/epilogue")
        tensors = (out, values, indices, weights, norm.weight)
        if any(
            value.device != self.device
            or not value.is_contiguous()
            or value.data_ptr() % 16
            or value.requires_grad
            for value in tensors
        ):
            raise ValueError("Native weighted reduction requires aligned contiguous local storage")
        if norm.weight.ndim != 1 or any(torch._C._overlaps(out, value) for value in tensors[1:]):
            raise ValueError("Native weighted output must not alias its triple or RMS weights")
        if "_residual_kernel" not in self.__dict__ and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm native weighted reduction before graph capture")
        with self.execution_context():
            self._residual_kernel.finalize_push_norm(
                self.comm.obj, out.view(-1), values, indices, weights, norm.weight, norm.eps
            )
        return out
