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

from .impl import DistributedCommunicator, DistributedImpl, _all_reduce_options


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

    # Generic storage is unrestricted; concrete consumers declare their ABI.
    _reserved_output_dtypes = None

    def __init__(
        self,
        comm,
        *,
        group=None,
        execution_context=nullcontext,
        input_alignment=1,
        output_group=None,
        device=None,
    ):
        self.comm = comm
        self.group = group
        self.execution_context = execution_context
        if input_alignment < 1 or input_alignment & (input_alignment - 1):
            raise ValueError("Collective input alignment must be a positive power of two")
        self.input_alignment = input_alignment
        self.output_group = output_group
        self.device = torch.device(device) if device is not None else None
        self._outputs = {}
        self._capturing = False
        self._closed = False

    def reserve_output(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, *, enabled: bool = True
    ):
        """Collective preparation, including reuse and rejected requests."""
        if self.output_group is None or self.device is None or self.device.type != "cuda":
            return None

        error = None
        if not isinstance(enabled, bool):
            error = ("value", "Output reservation enabled must be a bool")
        elif not isinstance(dtype, torch.dtype):
            error = ("value", "Output reservation dtype must be a torch dtype")
        elif (
            not isinstance(key, str)
            or not key
            or not isinstance(shape, (tuple, list, torch.Size))
            or not shape
            or any(not isinstance(size, int) or size <= 0 for size in shape)
        ):
            error = ("value", "Output reservation needs a name and positive integer shape")
        elif self._closed:
            error = ("state", "Cannot reserve output on a closed communicator")
        elif self._capturing or torch.cuda.is_current_stream_capturing():
            error = ("state", "Output reservation must precede graph capture")

        spec = None if error else (key, tuple(shape), str(dtype))
        previous = self._outputs.get(key) if error is None else None
        if previous is not None:
            tensor, _ = previous
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                error = ("value", "Cannot resize or change dtype of a reserved output")
        action = "error" if error else "reuse" if previous is not None else "new"
        available = (
            enabled
            and not getattr(self.comm, "disabled", False)
            and getattr(self.comm, "has_multicast", False)
            and (self._reserved_output_dtypes is None or dtype in self._reserved_output_dtypes)
        )
        # Every call agrees before returning, including cache hits and local
        # validation errors. A rank must never reuse while a peer rendezvouses.
        peers = [None] * dist.get_world_size(self.output_group)
        dist.all_gather_object(
            peers, (spec, action, error, bool(available)), group=self.output_group
        )
        errors = [peer[2] for peer in peers if peer[2] is not None]
        if errors:
            kind, message = errors[0]
            raise (ValueError if kind == "value" else RuntimeError)(message)
        if any(peer[0] != peers[0][0] for peer in peers):
            raise ValueError("Output reservation differs across communicator ranks")
        if any(peer[1] != peers[0][1] for peer in peers):
            raise RuntimeError("Output reservation reuse/new state differs across ranks")
        if not all(peer[3] for peer in peers):
            return None
        if action == "reuse":
            return previous[0]
        shape = tuple(shape)

        # Allocation is local. Agree its outcome before any rank enters the
        # collective rendezvous, so one rank's OOM cannot strand its peers.
        tensor = handle = None
        error = None
        try:
            from torch._C._distributed_c10d import _SymmetricMemory

            # Persistent mutable storage even inside inference_mode.
            with torch.inference_mode(False):
                tensor = _SymmetricMemory.empty_strided_p2p(
                    (torch.Size(shape).numel(),),
                    [1],
                    dtype,
                    self.device,
                    self.output_group.group_name,
                ).view(shape)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        dist.all_gather_object(peers, error, group=self.output_group)
        if any(peer is not None for peer in peers):
            tensor = None
            details = next(f"rank {rank}: {peer}" for rank, peer in enumerate(peers) if peer)
            raise RuntimeError(f"Output reservation allocation failed on {details}")

        # Do not publish a slot on only some ranks if rendezvous or multicast
        # validation fails locally. Retain temporary mappings until agreement.
        error = None
        try:
            handle = _SymmetricMemory.rendezvous(tensor)
            if handle is None or not handle.multicast_ptr:
                raise RuntimeError("Reserved output has no multicast mapping")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        dist.all_gather_object(peers, error, group=self.output_group)
        if any(peer is not None for peer in peers):
            handle = tensor = None
            details = next(f"rank {rank}: {peer}" for rank, peer in enumerate(peers) if peer)
            raise RuntimeError(f"Output reservation mapping failed on {details}")
        self._outputs[key] = (tensor, handle)
        return tensor

    def _output_mc_ptr(self, x: torch.Tensor, *, key: str | None = None):
        """Resolve contiguous aliases within the explicitly named reservation."""
        if not x.is_contiguous():
            return None
        ptr = x.data_ptr()
        end = ptr + x.numel() * x.element_size()
        outputs = self._outputs.values() if key is None else (self._outputs.get(key),)
        for output in outputs:
            if output is None:
                continue
            tensor, handle = output
            if (
                x.device == tensor.device
                and x.dtype == tensor.dtype
                and tensor.data_ptr() <= ptr
                and end <= tensor.data_ptr() + tensor.nbytes
            ):
                return int(handle.multicast_ptr) + ptr - tensor.data_ptr()
        return None

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
            previous = self._capturing
            self._capturing = True
            try:
                yield
            finally:
                self._capturing = previous

    def close(self) -> None:
        if self._closed:
            return
        # Engine destroys its graphs before distributed teardown. Drain any
        # remaining use before closing communication and dropping mappings.
        # A device/close failure deliberately retains the reservations.
        if self._outputs:
            torch.cuda.synchronize(self.device)
        close = getattr(self.comm, "close", None)
        if close is not None:
            close()
        self._outputs.clear()
        self._closed = True


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

    def reserve_output(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, *, enabled: bool = True
    ):
        reserve = getattr(self.provider, "reserve_output", None)
        return reserve(key, shape, dtype, enabled=enabled) if reserve is not None else None

    def supports_all_reduce_residual(
        self,
        x: torch.Tensor,
        residual=None,
        *,
        norm=None,
        output_key: str | None = None,
        overlap: bool = False,
    ) -> bool:
        method = getattr(self.provider, "supports_all_reduce_residual", None)
        options = _all_reduce_options(norm, output_key, overlap)
        return bool(method(x, residual, **options)) if method is not None else False

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
        if not self.supports_all_reduce_residual(x, residual, **options):
            return super().all_reduce_residual(x, residual, **options)
        return self.provider.all_reduce_residual(x, residual, **options)

    def supports_weighted_all_reduce(self, out, *, norm, topk: int) -> bool:
        method = getattr(self.provider, "supports_weighted_all_reduce", None)
        return bool(method(out, norm=norm, topk=topk)) if method is not None else False

    def weighted_all_reduce(self, values, indices, weights, *, out, norm):
        method = getattr(self.provider, "weighted_all_reduce", None)
        if method is None:
            raise NotImplementedError("This provider has no weighted reduction consumer")
        return method(values, indices, weights, out=out, norm=norm)

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
