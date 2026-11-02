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
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    @classmethod
    def enable(cls, backend: DistributedImpl) -> None:
        cls.plugins.append(backend)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)

    def all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> torch.Tensor:
        return self.plugins[-1].all_reduce_residual(x, residual, norm=norm)

    def capture(self):
        return self.plugins[-1].capture()


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
    for plugin in reversed(DistributedCommunicator.plugins):
        plugin.close()
    DistributedCommunicator.plugins = []
