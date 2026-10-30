from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)


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
    """Declare configuration-time capabilities without constructing GPU resources."""

    def register(create):
        SUPPORTED_ATTENTION_BACKENDS.register(name)(
            AttentionBackendSpec(create, stateful_layers, cuda_graphs)
        )
        return create

    return register


@register_attention_backend("recurrent", stateful_layers=True, cuda_graphs=False)
def create_recurrent_backend(config: ModelConfig):
    from .recurrent import RecurrentAttentionBackend

    return RecurrentAttentionBackend(config)


@register_attention_backend("recurrent-fi", stateful_layers=True)
def create_recurrent_fi_backend(config: ModelConfig):
    from .mla_fi import FlashInferMLA
    from .recurrent import RecurrentAttentionBackend

    return RecurrentAttentionBackend(
        config, kda_implementation="flashinfer", mla_attention=FlashInferMLA()
    )


@register_attention_backend("trtllm")
def create_trtllm_backend(config: ModelConfig):
    from .trtllm import TensorRTLLMBackend

    return TensorRTLLMBackend(config)


@register_attention_backend("fi")
def create_fi_backend(config: ModelConfig):
    from .fi import FlashInferBackend

    return FlashInferBackend(config)


@register_attention_backend("fa")
def create_fa_backend(config: ModelConfig):
    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    if backend != "auto":
        required_backends = backend.split(",") if "," in backend else [backend]
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(
    backend: str,
    config: ModelConfig,
) -> BaseAttnBackend:
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            p_backend = create_attention_backend(p_backend, config)
            d_backend = create_attention_backend(d_backend, config)
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # both are the same, fall through to single backend
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "register_attention_backend",
    "validate_attn_backend",
]
