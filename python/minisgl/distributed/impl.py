from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


def _all_reduce_options(norm, output_key, overlap: bool):
    """Validate the resource hint and preserve the default provider call ABI."""
    if type(overlap) is not bool:
        raise ValueError("Reduction overlap hint must be a bool")
    options = {"norm": norm, "output_key": output_key}
    if overlap:
        options["overlap"] = True
    return options


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...

    def all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This communication backend does not support all-to-all")

    def supports_all_to_all(self, x: torch.Tensor) -> bool:
        return False

    def supports(self, x: torch.Tensor) -> bool:
        """Backward-compatible capability alias for all-reduce."""
        return self.supports_all_reduce(x)

    def supports_all_reduce(self, x: torch.Tensor) -> bool:
        """Whether this backend can reduce this tensor without fallback."""
        return True

    def supports_all_gather(self, x: torch.Tensor) -> bool:
        """Whether this backend can gather this tensor without fallback."""
        return True

    def supports_linear_all_gather(self, x, weight, addend) -> bool:
        """Whether this backend covers a replicated linear projection and add."""
        return False

    def linear_all_gather(self, x, weight, addend) -> torch.Tensor:
        """Return owned x @ weight.T + addend via column GEMM and all-gather.

        Every group member supplies the same replicated x, FULL replicated
        weight, and already-reduced addend. Inputs are read-only and must remain
        live until current-stream consumption completes. This optional inference
        operation is not row-parallel linear or a reduction of local partials.
        Callers select capability before launch and keep their ordinary Linear
        path when unsupported; native errors must never trigger a retry.
        """
        raise NotImplementedError("This communication backend does not support linear all-gather")

    def reserve_output(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, *, enabled: bool = True
    ):
        """Collectively reserve stable provider-owned output storage, or decline.

        Call in the same order on every group member during preparation, after
        KV allocation and before graph capture. Storage lives until backend
        close; its contents last only until the next write to the same slot.
        Every rank must use the corresponding reservation and slice. Callers
        copy retained aliases before reuse, without growing the reservation.
        The optional all_reduce_residual output_key names this reservation;
        all group members must supply the same key and corresponding slice.
        enabled is caller readiness: every rank still participates, and any
        disabled member declines preparation for the whole group.
        """
        return None

    def supports_all_reduce_residual(
        self,
        x: torch.Tensor,
        residual=None,
        *,
        norm=None,
        output_key: str | None = None,
        overlap: bool = False,
    ) -> bool:
        """Whether this backend can fuse reduction with a residual epilogue."""
        return False

    def all_reduce_residual(
        self,
        x: torch.Tensor,
        residual=None,
        *,
        norm=None,
        output_key: str | None = None,
        overlap: bool = False,
    ) -> torch.Tensor:
        """Reduce ``x`` and apply an optional residual and unary epilogue.

        Hardware backends can override this to use a fused communication
        kernel. The fallback keeps the operation correct for every backend.
        overlap is an execution hint for work overlapping other GPU kernels;
        the generic composition may ignore it without changing the result.
        output_key is an explicit prepared-storage route hint, ignored by the
        generic composition. Backends may mutate x; copy retained input aliases
        before dispatch. Native pull requires the same reservation key and slice
        on every rank; omitting the key preserves generic large-tensor routing.
        The generic composition reduces before adding residual: callers copy an
        overlapping residual first when the reduction may mutate x. A native
        backend may document stronger snapshot semantics for its fused path.
        """
        _all_reduce_options(norm, output_key, overlap)
        result = self.all_reduce(x)
        if residual is not None:
            result = result + residual
        return norm(result) if norm is not None else result

    def supports_weighted_all_reduce(self, out, *, norm, topk: int) -> bool:
        """Advertise an output-only weighted reduction before producers launch."""
        return False

    def weighted_all_reduce(self, values, indices, weights, *, out, norm):
        """Write SUM_rank SUM_k weights[n,k] * values[indices[n,k]], then norm.

        Negative indices contribute zero. Indices are flattened [N*topk], weights
        are [N,topk], and values contain local expert rows. A supported native
        implementation owns accumulation/rounding. The destination is output-only
        and may be borrowed storage; retain copies before its next producer write.
        There is deliberately no generic gather/finalize kernel or CUDA retry.
        Call supports_weighted_all_reduce before launching an unfinalized producer.
        """
        raise NotImplementedError("This backend has no weighted reduction consumer")

    def capture(self):
        """Register communication resources around a complete CUDA graph capture.

        Entry precedes warm-up and capture; exit runs after CUDA capture has
        ended, when peer-input registration may use host collectives.
        """
        return nullcontext()

    def close(self) -> None:
        """Release owned communication resources before the bootstrap group."""


@dataclass
class FallbackDistributedImpl(DistributedImpl):
    """Route unsupported tensors to a stable, already-tested backend."""

    primary: DistributedImpl
    fallback: DistributedImpl

    @staticmethod
    def _supports(backend: DistributedImpl, operation: str, x: torch.Tensor) -> bool:
        method = getattr(backend, f"supports_{operation}", None)
        return bool(method(x) if method is not None else backend.supports(x))

    def supports_all_reduce(self, x: torch.Tensor) -> bool:
        return self._supports(self.primary, "all_reduce", x) or self._supports(
            self.fallback, "all_reduce", x
        )

    def supports_all_gather(self, x: torch.Tensor) -> bool:
        return self._supports(self.primary, "all_gather", x) or self._supports(
            self.fallback, "all_gather", x
        )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.primary if self._supports(self.primary, "all_reduce", x) else self.fallback
        return backend.all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.primary if self._supports(self.primary, "all_gather", x) else self.fallback
        return backend.all_gather(x)

    def supports_all_to_all(self, x: torch.Tensor) -> bool:
        return self._supports(self.primary, "all_to_all", x) or self._supports(
            self.fallback, "all_to_all", x
        )

    def all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.primary if self._supports(self.primary, "all_to_all", x) else self.fallback
        return backend.all_to_all(x)

    def supports_linear_all_gather(self, x, weight, addend) -> bool:
        return self.primary.supports_linear_all_gather(
            x, weight, addend
        ) or self.fallback.supports_linear_all_gather(x, weight, addend)

    def linear_all_gather(self, x, weight, addend) -> torch.Tensor:
        for backend in (self.primary, self.fallback):
            if backend.supports_linear_all_gather(x, weight, addend):
                return backend.linear_all_gather(x, weight, addend)
        return super().linear_all_gather(x, weight, addend)

    def reserve_output(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, *, enabled: bool = True
    ):
        result = self.primary.reserve_output(key, shape, dtype, enabled=enabled)
        return (
            self.fallback.reserve_output(key, shape, dtype, enabled=enabled)
            if result is None
            else result
        )

    def supports_all_reduce_residual(
        self,
        x: torch.Tensor,
        residual=None,
        *,
        norm=None,
        output_key: str | None = None,
        overlap: bool = False,
    ) -> bool:
        options = _all_reduce_options(norm, output_key, overlap)
        return self.primary.supports_all_reduce_residual(
            x, residual, **options
        ) or self.fallback.supports_all_reduce_residual(x, residual, **options)

    def all_reduce_residual(
        self,
        x: torch.Tensor,
        residual=None,
        *,
        norm=None,
        output_key: str | None = None,
        overlap: bool = False,
    ) -> torch.Tensor:
        options = _all_reduce_options(norm, output_key, overlap)
        for backend in (self.primary, self.fallback):
            if backend.supports_all_reduce_residual(x, residual, **options):
                return backend.all_reduce_residual(x, residual, **options)
        # A backend may reduce this tensor without covering its epilogue.
        # Keep capability-based routing for the reduction and compose the rest.
        return super().all_reduce_residual(x, residual, **options)

    def supports_weighted_all_reduce(self, out, *, norm, topk: int) -> bool:
        return any(
            backend.supports_weighted_all_reduce(out, norm=norm, topk=topk)
            for backend in (self.primary, self.fallback)
        )

    def weighted_all_reduce(self, values, indices, weights, *, out, norm):
        if weights.ndim != 2:
            raise ValueError("Weighted reduction weights must have [rows, topk] shape")
        for backend in (self.primary, self.fallback):
            if backend.supports_weighted_all_reduce(out, norm=norm, topk=weights.shape[1]):
                return backend.weighted_all_reduce(values, indices, weights, out=out, norm=norm)
        return super().weighted_all_reduce(values, indices, weights, out=out, norm=norm)

    @contextmanager
    def capture(self):
        # Either backend may serve tensors in the same graph. Both must register
        # their inputs, and partial entry failures must unwind the first backend.
        backends = (
            (self.primary,) if self.primary is self.fallback else (self.primary, self.fallback)
        )
        with ExitStack() as stack:
            for backend in backends:
                stack.enter_context(backend.capture())
            yield

    def close(self) -> None:
        self.primary.close()
        if self.fallback is not self.primary:
            self.fallback.close()


@dataclass
class TorchDistributedImpl(DistributedImpl):
    group: dist.ProcessGroup | None = None
    owns_group: bool = False

    def __post_init__(self):
        if self.owns_group and self.group is None:
            raise ValueError("An owned process group must be explicit; None denotes WORLD")

    def close(self) -> None:
        if self.owns_group:
            self.owns_group = False
            dist.destroy_process_group(self.group)

    def warmup(self, device: torch.device) -> None:
        """Allocate NCCL collective resources before measuring cache capacity."""
        if dist.get_backend(self.group) != dist.Backend.NCCL:
            return
        size = dist.get_world_size(self.group)
        if size == 1:
            return
        probe = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.all_reduce(probe)
        self.all_gather(probe)
        self.all_to_all(probe)
        torch.cuda.synchronize(device)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size(self.group)
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size(self.group)
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x, group=self.group)
        return out

    def supports_all_to_all(self, x: torch.Tensor) -> bool:
        return True

    def all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        if dist.get_world_size(self.group) == 1:
            return x
        result = torch.empty_like(x)
        dist.all_to_all_single(result, x, group=self.group)
        return result


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        from tvm_ffi import use_torch_stream

        with use_torch_stream() if x.is_cuda else nullcontext():
            self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        from tvm_ffi import use_torch_stream

        with use_torch_stream() if x.is_cuda else nullcontext():
            self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """Backend selection for tensor-wide and explicitly scoped process groups."""

    plugins: List[DistributedImpl] = [TorchDistributedImpl()]
    groups: dict[str, DistributedImpl] = {}

    def __init__(self, group: str = "tensor"):
        self.group = group

    @classmethod
    def enable(cls, backend: DistributedImpl, *, group: str = "tensor") -> None:
        if group == "tensor":
            cls.plugins.append(backend)
        else:
            cls.groups[group] = backend

    def _backend(self):
        if self.group == "tensor":
            return self.plugins[-1]
        try:
            return self.groups[self.group]
        except KeyError:
            raise RuntimeError(f"Communication group {self.group!r} is not initialized") from None

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend().all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend().all_gather(x)

    def all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend().all_to_all(x)

    def supports_linear_all_gather(self, x, weight, addend) -> bool:
        return self._backend().supports_linear_all_gather(x, weight, addend)

    def linear_all_gather(self, x, weight, addend) -> torch.Tensor:
        return self._backend().linear_all_gather(x, weight, addend)

    def reserve_output(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, *, enabled: bool = True
    ):
        return self._backend().reserve_output(key, shape, dtype, enabled=enabled)

    def all_reduce_residual(
        self,
        x: torch.Tensor,
        residual=None,
        *,
        norm=None,
        output_key: str | None = None,
        overlap: bool = False,
    ) -> torch.Tensor:
        options = _all_reduce_options(norm, output_key, overlap)
        return self._backend().all_reduce_residual(x, residual, **options)

    def supports_weighted_all_reduce(self, out, *, norm, topk: int) -> bool:
        return self._backend().supports_weighted_all_reduce(out, norm=norm, topk=topk)

    def weighted_all_reduce(self, values, indices, weights, *, out, norm):
        return self._backend().weighted_all_reduce(values, indices, weights, out=out, norm=norm)

    @contextmanager
    def capture(self):
        backends = [self._backend()]
        if self.group == "tensor":
            backends.extend(self.groups.values())
        with ExitStack() as stack:
            seen = set()
            for backend in backends:
                if id(backend) not in seen:
                    stack.enter_context(backend.capture())
                    seen.add(id(backend))
            yield


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.enable(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    backends = [
        *DistributedCommunicator.groups.values(),
        *reversed(DistributedCommunicator.plugins),
    ]
    seen = set()
    for backend in backends:
        if id(backend) not in seen:
            backend.close()
            seen.add(id(backend))
    DistributedCommunicator.groups = {}
    DistributedCommunicator.plugins = []
