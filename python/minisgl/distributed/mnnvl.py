"""Optional MNNVL communication adapter.

MNNVL is a transport/backend capability, not a model feature.  The adapter
accepts an externally compiled provider so mini-sglang can use symmetric-memory
or multicast collectives when present while keeping the default Torch/PyNCCL
implementations self-contained and portable.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager, nullcontext
from typing import Protocol

import torch
import torch.distributed as dist

from .impl import DistributedCommunicator, DistributedImpl


class MNNVLProvider(Protocol):
    """Minimal provider contract implemented by a hardware communication stack."""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


class MNNVLAllReduceProvider:
    """Use a CustomAllReduceV2-compatible object without importing SGLang.

    The object is intentionally duck-typed: an external runtime supplies
    ``should_custom_ar``, ``custom_all_reduce`` and ``close``.  Unsupported
    collectives use the same process group through ordinary Torch/NCCL calls.
    """

    def __init__(self, comm, *, group=None, execution_context=nullcontext, input_alignment=1):
        self.comm = comm
        self.group = group
        self.execution_context = execution_context
        if input_alignment < 1 or input_alignment & (input_alignment - 1):
            raise ValueError("Collective input alignment must be a positive power of two")
        self.input_alignment = input_alignment

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize locally instead of choosing a different collective on peers
        # whose views have different offsets or strides.
        if not x.is_contiguous() or x.data_ptr() % self.input_alignment:
            x = x.clone(memory_format=torch.contiguous_format)
        if not getattr(self.comm, "disabled", False) and self.comm.should_custom_ar(x):
            with self.execution_context():
                return self.comm.custom_all_reduce(x)
        dist.all_reduce(x, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        shape[0] *= dist.get_world_size(group=self.group)
        output = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(output, x, group=self.group)
        return output

    @contextmanager
    def capture(self):
        capture = getattr(self.comm, "capture", None)
        with self.execution_context(), capture() if capture is not None else nullcontext():
            yield

    def close(self) -> None:
        close = getattr(self.comm, "close", None)
        if close is not None:
            close()


class MNNVLDistributedImpl(DistributedImpl):
    """Adapt an MNNVL provider to mini's normal distributed backend contract."""

    name = "mnnvl"

    def __init__(self, provider: MNNVLProvider):
        self.provider = provider

    def _supports(self, operation: str, x: torch.Tensor) -> bool:
        method = getattr(self.provider, f"supports_{operation}", None)
        return bool(method(x)) if method is not None else True

    def supports_all_reduce(self, x: torch.Tensor) -> bool:
        return self._supports("all_reduce", x)

    def supports_all_gather(self, x: torch.Tensor) -> bool:
        return self._supports("all_gather", x)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.provider.all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.provider.all_gather(x)

    def supports_all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> bool:
        method = getattr(self.provider, "supports_all_reduce_residual", None)
        return bool(method(x, residual, norm=norm)) if method is not None else False

    def all_reduce_residual(self, x: torch.Tensor, residual=None, *, norm=None) -> torch.Tensor:
        if not self.supports_all_reduce_residual(x, residual, norm=norm):
            return super().all_reduce_residual(x, residual, norm=norm)
        return self.provider.all_reduce_residual(x, residual, norm=norm)

    def try_all_reduce_norm(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor | None:
        method = getattr(self.provider, "try_all_reduce_norm", None)
        return method(x, weight, eps) if method is not None else None

    def capture(self):
        capture = getattr(self.provider, "capture", None)
        return capture() if capture is not None else nullcontext()

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if close is not None:
            close()


def enable_mnnvl_distributed(provider: MNNVLProvider) -> MNNVLDistributedImpl:
    """Install an externally compiled MNNVL provider as the active backend."""

    backend = MNNVLDistributedImpl(provider)
    DistributedCommunicator.enable(backend)
    return backend


def _provider_factory(module_name: str):
    if not module_name:
        raise ValueError("An MNNVL provider module is required for the mnnvl backend")
    module = importlib.import_module(module_name)
    factory = getattr(module, "create_provider", None)
    if not callable(factory):
        raise AttributeError(f"MNNVL provider module {module_name!r} lacks create_provider")
    return module, factory


def prepare_mnnvl_provider(module_name: str, **kwargs) -> None:
    """Run optional local preparation before any process group is created."""
    module, _ = _provider_factory(module_name)
    prepare = getattr(module, "prepare", None)
    if prepare is not None:
        if not callable(prepare):
            raise TypeError("MNNVL provider prepare must be callable")
        prepare(**kwargs)


def load_mnnvl_provider(module_name: str, **kwargs) -> MNNVLProvider:
    """Create a provider after bootstrap and device process groups are ready."""
    _, factory = _provider_factory(module_name)
    provider = factory(**kwargs)
    if not callable(getattr(provider, "all_reduce", None)) or not callable(
        getattr(provider, "all_gather", None)
    ):
        raise TypeError("MNNVL provider must implement all_reduce and all_gather")
    return provider


__all__ = [
    "MNNVLAllReduceProvider",
    "MNNVLDistributedImpl",
    "MNNVLProvider",
    "enable_mnnvl_distributed",
    "load_mnnvl_provider",
    "prepare_mnnvl_provider",
]
