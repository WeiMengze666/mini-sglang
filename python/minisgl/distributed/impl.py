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

    def supports_all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> bool:
        """Whether this backend can fuse reduction with a residual epilogue."""
        return False

    def all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> torch.Tensor:
        """Reduce ``x`` and apply an optional residual and unary epilogue.

        Hardware backends can override this to use a fused communication
        kernel.  The fallback keeps the operation correct for every backend.
        """
        result = self.all_reduce(x)
        if residual is not None:
            result = result + residual
        return norm(result) if norm is not None else result

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

    def supports_all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> bool:
        return self.primary.supports_all_reduce_residual(
            x, residual, norm=norm
        ) or self.fallback.supports_all_reduce_residual(x, residual, norm=norm)

    def all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> torch.Tensor:
        for backend in (self.primary, self.fallback):
            if backend.supports_all_reduce_residual(x, residual, norm=norm):
                return backend.all_reduce_residual(x, residual, norm=norm)
        # A backend may reduce this tensor without covering its epilogue.
        # Keep capability-based routing for the reduction and compose the rest.
        return super().all_reduce_residual(x, residual, norm=norm)

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

    def all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> torch.Tensor:
        return self._backend().all_reduce_residual(x, residual, norm=norm)

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
