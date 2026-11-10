"""Pluggable KDA projection, recurrence and decode kernels.

The model supplies projections and owns recurrent state.  A kernel backend may
adapt projection dispatch, replace the numerical core, or provide a one-token decode
path when it can consume the layer's state layout.  This keeps hardware
specific fusion out of model definitions while preserving a portable fallback.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

import torch
from minisgl.utils import Registry

if TYPE_CHECKING:
    from minisgl.kvcache.recurrent_pool import RecurrentStatePool
    from minisgl.layers.kda import KDALayer

    from .backend import KDAMetadata
    from .reference import KDAState


class RecurrenceKernel(Protocol):
    """Functional core; phase is explicit even when prefill contains one token.

    Adapters receive raw gates and K,V state and must accept ``is_decode``.
    Portable backends may use one algorithm for both phases; native backends
    must preserve their phase-specific precision and state-update contracts.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forget: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor | None,
        lower_bound: float | None,
        *,
        is_decode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


ProjectionKernel = Callable[["KDALayer", "torch.Tensor"], tuple["torch.Tensor", ...]]
DecodeKernel = Callable[["KDALayer", "torch.Tensor", "KDAState | None"], "KDADecodeResult | None"]
IndexedDecodeKernel = Callable[
    ["KDALayer", "torch.Tensor", "RecurrentStatePool", int, "torch.Tensor"],
    "KDADecodeResult | None",
]

IndexedPrefillKernel = Callable[
    ["KDALayer", "torch.Tensor", "RecurrentStatePool", int, "torch.Tensor", "KDAMetadata"],
    "torch.Tensor | None",
]


@dataclass(frozen=True)
class KDADecodeResult:
    """Core decode output and updated state returned by a fused backend.

    ``normed`` is true when the backend already applied KDA's gated output
    normalization. The layer owns the output projection and collective.
    Functional decode returns state; indexed decode updates its owned pool and
    leaves ``state`` unset.
    """

    output: "torch.Tensor"
    state: "KDAState | None" = None
    normed: bool = False


@dataclass(frozen=True)
class KDAKernelSpec:
    name: str
    recurrence: RecurrenceKernel
    decode: DecodeKernel | None = None
    project: ProjectionKernel | None = None
    indexed_decode: IndexedDecodeKernel | None = None
    indexed_prefill: IndexedPrefillKernel | None = None


KDA_KERNELS = Registry[KDAKernelSpec]("KDA kernel backend")


def register_kda_kernel(
    name: str,
    recurrence: RecurrenceKernel,
    decode: DecodeKernel | None = None,
    *,
    project: ProjectionKernel | None = None,
    indexed_decode: IndexedDecodeKernel | None = None,
    indexed_prefill: IndexedPrefillKernel | None = None,
):
    """Register a recurrence backend without coupling it to a model class."""

    KDA_KERNELS.register(name)(
        KDAKernelSpec(name, recurrence, decode, project, indexed_decode, indexed_prefill)
    )


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


def _torch_recurrence(*args, is_decode=False):
    from .reference import kda_log_decay, kda_recurrence

    q, k, v, forget, beta, a, dt, state, lower = args
    if is_decode and q.shape[0] != 1:
        raise ValueError("KDA decode requires one token per request")
    decay = kda_log_decay(forget, a, dt, lower)
    return kda_recurrence(q, k, v, decay, beta.float().sigmoid(), state)


def _supports_cuda_recurrence(q):
    return q.is_cuda and q.dtype == torch.bfloat16 and q.shape[-1] == 128


def _flashinfer_recurrence(*args, is_decode=False):
    if not _supports_cuda_recurrence(args[0]):
        return _torch_recurrence(*args, is_decode=is_decode)
    from .flashinfer import recurrent_kda

    return recurrent_kda(*args, is_decode=is_decode)


def _sgl_recurrence(*args, is_decode=False):
    if not _supports_cuda_recurrence(args[0]):
        return _torch_recurrence(*args, is_decode=is_decode)
    from .sgl import recurrent_kda

    return recurrent_kda(*args, is_decode=is_decode)


def _sgl_project(layer, x):
    from .sgl import project

    return project(layer, x)


def _sgl_decode(layer, x, state):
    from .sgl import decode

    return decode(layer, x, state)


def _sgl_indexed_decode(layer, x, pool, layer_id, indices):
    from .sgl import indexed_decode

    return indexed_decode(layer, x, pool, layer_id, indices)


def _flashinfer_indexed_prefill(*args):
    from .flashinfer import indexed_prefill

    return indexed_prefill(*args)


register_kda_kernel("torch", _torch_recurrence)
register_kda_kernel(
    "flashinfer", _flashinfer_recurrence, indexed_prefill=_flashinfer_indexed_prefill
)
register_kda_kernel(
    "sgl",
    _sgl_recurrence,
    _sgl_decode,
    project=_sgl_project,
    indexed_decode=_sgl_indexed_decode,
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
