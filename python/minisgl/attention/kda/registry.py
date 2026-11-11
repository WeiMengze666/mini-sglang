"""Pluggable KDA projection, recurrence and decode kernels.

The model supplies projections and owns recurrent state.  A kernel backend may
adapt projection dispatch, replace the numerical core, or provide a one-token decode
path when it can consume the layer's state layout.  This keeps hardware
specific fusion out of model definitions while preserving a portable fallback.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
from minisgl.utils import Registry

if TYPE_CHECKING:
    from minisgl.kvcache.recurrent_pool import RecurrentStatePool
    from minisgl.layers.kda import KDALayer

    from .backend import KDAMetadata
    from .reference import KDAState

RecurrenceKernel = Callable[..., tuple["torch.Tensor", "torch.Tensor"]]
ProjectionKernel = Callable[["KDALayer", "torch.Tensor"], "KDAProjection"]
DecodeKernel = Callable[["KDALayer", "torch.Tensor", "KDAState | None"], "KDADecodeResult | None"]
IndexedDecodeKernel = Callable[
    ["KDALayer", "torch.Tensor", "RecurrentStatePool", int, "torch.Tensor", "KDAMetadata"],
    "KDADecodeResult | None",
]


class KDAProjection(NamedTuple):
    """Core attention projections and an optional explicitly carried output gate."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    forget: torch.Tensor
    beta: torch.Tensor
    gate: torch.Tensor | None = None


@dataclass(frozen=True)
class KDADecodeResult:
    """Core decode output and updated state returned by a fused backend.

    ``normed`` is true when the backend already applied KDA's gated output
    normalization. The layer owns the output projection and collective.
    Functional decode returns state; indexed decode updates its owned pool and
    leaves ``state`` unset. The optional gate is carried to the layer when
    output normalization remains outside the kernel.
    """

    output: "torch.Tensor"
    state: "KDAState | None" = None
    normed: bool = False
    gate: torch.Tensor | None = None


@dataclass(frozen=True)
class KDAKernelSpec:
    name: str
    recurrence: RecurrenceKernel
    decode: DecodeKernel | None = None
    project: ProjectionKernel | None = None
    indexed_decode: IndexedDecodeKernel | None = None


KDA_KERNELS = Registry[KDAKernelSpec]("KDA kernel backend")


def register_kda_kernel(
    name: str,
    recurrence: RecurrenceKernel,
    decode: DecodeKernel | None = None,
    *,
    project: ProjectionKernel | None = None,
    indexed_decode: IndexedDecodeKernel | None = None,
):
    """Register a recurrence backend without coupling it to a model class."""

    KDA_KERNELS.register(name)(KDAKernelSpec(name, recurrence, decode, project, indexed_decode))


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


def _flashinfer_recurrence(*args):
    if not _supports_cuda_recurrence(args[0]):
        return _torch_recurrence(*args)
    from .flashinfer import recurrent_kda

    return recurrent_kda(*args)


def _sgl_recurrence(*args):
    if not _supports_cuda_recurrence(args[0]):
        return _torch_recurrence(*args)
    from .sgl import recurrent_kda

    return recurrent_kda(*args)


def _sgl_project(layer, x):
    from .sgl import project

    return project(layer, x)


def _sgl_decode(layer, x, state):
    from .sgl import decode

    return decode(layer, x, state)


def _sgl_indexed_decode(layer, x, pool, layer_id, indices, metadata):
    from .sgl import indexed_decode

    return indexed_decode(layer, x, pool, layer_id, indices, metadata)


register_kda_kernel("torch", _torch_recurrence)
register_kda_kernel("flashinfer", _flashinfer_recurrence)
register_kda_kernel(
    "sgl", _sgl_recurrence, _sgl_decode, project=_sgl_project, indexed_decode=_sgl_indexed_decode
)


__all__ = [
    "KDADecodeResult",
    "KDAKernelSpec",
    "KDA_KERNELS",
    "get_kda_kernel",
    "register_kda_kernel",
    "register_kda_decode_kernel",
    "load_kda_provider",
]
