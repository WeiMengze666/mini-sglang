"""Attention backend construction. Algorithm implementations live in subpackages."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

from .base import (
    BaseAttnBackend,
    BaseAttnMetadata,
    HybridAttnBackend,
    PrefillDecodeAttnBackend,
)

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


class BackendCreator(Protocol):
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...


@dataclass(frozen=True)
class AttentionBackendSpec:
    create: BackendCreator
    stateful_layers: bool = False
    cuda_graphs: bool = True

    def __call__(self, config: ModelConfig) -> BaseAttnBackend:
        return self.create(config)


SUPPORTED_ATTENTION_BACKENDS = Registry[AttentionBackendSpec]("Attention Backend")


def register_attention_backend(name, *, stateful_layers=False, cuda_graphs=True):
    """Declare startup capabilities without constructing GPU resources."""

    def register(create):
        SUPPORTED_ATTENTION_BACKENDS.register(name)(
            AttentionBackendSpec(create, stateful_layers, cuda_graphs)
        )
        return create

    return register


@cache
def load_attention_provider(module_name: str) -> None:
    """Load an optional module that registers complete attention backends."""
    if not module_name:
        raise ValueError("An attention provider module is required")
    module = importlib.import_module(module_name)
    register = getattr(module, "register_attention_backends", None)
    if not callable(register):
        raise TypeError(f"Attention provider {module_name!r} lacks register_attention_backends")
    register()


@register_attention_backend("torch", stateful_layers=True, cuda_graphs=False)
def create_torch_backend(config: ModelConfig, *, cache=None):
    from .kda.backend import KDAAttentionBackend
    from .mla.backend import MLAAttentionBackend

    if config.cache_config is None:
        raise ValueError("The Torch reference backend requires a KDA/MLA cache layout")
    return HybridAttnBackend(
        KDAAttentionBackend(config, cache=cache),
        MLAAttentionBackend(config, cache=cache),
        config.is_kda_layer,
    )


@register_attention_backend("fi", stateful_layers=True)
def create_fi_backend(config: ModelConfig):
    from minisgl.precision import use_strict_bf16

    from .fi import FlashInferBackend, FlashInferMLABackend
    from .kda.backend import KDAAttentionBackend
    from .kda.registry import get_kda_kernel

    if config.cache_config is None:
        return FlashInferBackend(config)
    kernel = get_kda_kernel("sgl" if use_strict_bf16() else "flashinfer")
    return HybridAttnBackend(
        KDAAttentionBackend(config, kernel),
        FlashInferMLABackend(config),
        config.is_kda_layer,
    )


@register_attention_backend("sgl", stateful_layers=True)
def create_sgl_backend(config: ModelConfig):
    """Compose the optional SGLang KDA kernels with FlashInfer's MLA portfolio."""
    from .fi import FlashInferBackend, FlashInferMLABackend
    from .kda.backend import KDAAttentionBackend
    from .kda.registry import get_kda_kernel

    if config.cache_config is None:
        return FlashInferBackend(config)
    return HybridAttnBackend(
        KDAAttentionBackend(config, get_kda_kernel("sgl")),
        FlashInferMLABackend(config),
        config.is_kda_layer,
    )


@register_attention_backend("trtllm")
def create_trtllm_backend(config: ModelConfig):
    from .trtllm import TensorRTLLMBackend

    return TensorRTLLMBackend(config)


@register_attention_backend("fa")
def create_fa_backend(config: ModelConfig):
    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    if backend != "auto":
        names = backend.split(",")
        assert len(names) <= 2, "Only one comma is allowed for prefill/decode selection"
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(names)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(backend: str, config: ModelConfig) -> BaseAttnBackend:
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        prefill, decode = backend.split(",")
        if prefill != decode:
            return PrefillDecodeAttnBackend(
                create_attention_backend(prefill, config),
                create_attention_backend(decode, config),
            )
        backend = prefill
    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "HybridAttnBackend",
    "PrefillDecodeAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "register_attention_backend",
    "load_attention_provider",
    "validate_attn_backend",
]
