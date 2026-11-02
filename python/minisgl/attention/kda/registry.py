"""Pluggable KDA recurrence and decode kernels.

The model supplies projections and owns recurrent state.  A kernel backend may
replace only the numerical core, or may provide a complete one-token decode
path when it can consume the layer's state layout.  This keeps hardware
specific fusion out of model definitions while preserving a portable fallback.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
from minisgl.utils import Registry

if TYPE_CHECKING:
    from minisgl.layers.kda import KDALayer

    from .reference import KDAState

RecurrenceKernel = Callable[..., tuple["torch.Tensor", "torch.Tensor"]]
DecodeKernel = Callable[["KDALayer", "torch.Tensor", "KDAState | None"], "KDADecodeResult | None"]


@dataclass(frozen=True)
class KDADecodeResult:
    """Core decode output and updated state returned by a fused backend.

    ``normed`` is true when the backend already applied KDA's gated output
    normalization.  The layer still owns the output projection and collective,
    so a fused kernel never needs to know the model's tensor-parallel layout.
    """

    output: "torch.Tensor"
    state: "KDAState"
    normed: bool = False


@dataclass(frozen=True)
class KDAKernelSpec:
    name: str
    recurrence: RecurrenceKernel
    decode: DecodeKernel | None = None


KDA_KERNELS = Registry[KDAKernelSpec]("KDA kernel backend")


def register_kda_kernel(
    name: str, recurrence: RecurrenceKernel, decode: DecodeKernel | None = None
):
    """Register a recurrence backend without coupling it to a model class."""

    KDA_KERNELS.register(name)(KDAKernelSpec(name, recurrence, decode))


def register_kda_decode_kernel(name: str, decode: DecodeKernel):
    """Register a fused decode path while retaining portable prefill logic."""

    register_kda_kernel(name, _torch_recurrence, decode)


def load_kda_provider(module_name: str) -> None:
    """Load an external module that registers one or more KDA kernel specs."""

    if not module_name:
        raise ValueError("A KDA provider module is required for an external KDA backend")
    module = importlib.import_module(module_name)
    register = getattr(module, "register_kda_kernels", None)
    if register is None:
        raise AttributeError(f"KDA provider module {module_name!r} lacks register_kda_kernels")
    register()


def get_kda_kernel(name: str) -> KDAKernelSpec:
    return KDA_KERNELS[name]


def _torch_recurrence(*args):
    from .reference import kda_log_decay, kda_recurrence

    q, k, v, forget, beta, a, dt, state, lower = args
    decay = kda_log_decay(forget, a, dt, lower)
    return kda_recurrence(q, k, v, decay, beta.float().sigmoid(), state)


def _supports_cuda_recurrence(q):
    return q.is_cuda and q.dtype == torch.bfloat16 and q.shape[-1] == 128


def _triton_recurrence(*args):
    if not _supports_cuda_recurrence(args[0]):
        return _torch_recurrence(*args)
    from .triton import recurrent_kda

    return recurrent_kda(*args)


def _flashinfer_recurrence(*args):
    if not _supports_cuda_recurrence(args[0]):
        return _torch_recurrence(*args)
    from .flashinfer import recurrent_kda

    return recurrent_kda(*args)


register_kda_kernel("torch", _torch_recurrence)
register_kda_kernel("triton", _triton_recurrence)
register_kda_kernel("flashinfer", _flashinfer_recurrence)


__all__ = [
    "KDADecodeResult",
    "KDAKernelSpec",
    "KDA_KERNELS",
    "get_kda_kernel",
    "register_kda_kernel",
    "register_kda_decode_kernel",
    "load_kda_provider",
]
