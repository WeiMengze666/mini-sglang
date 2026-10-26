from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    max_running_req: int | None = None,
    max_cached_states: int = 0,
    unified_memory: bool = False,
) -> BaseKVCachePool:
    if model_config.cache_config is not None:
        from .hybrid_pool import HybridCache

        if max_running_req is None:
            raise ValueError("Hybrid cache requires max_running_req for bounded recurrent state")
        return HybridCache(
            model_config.cache_config,
            num_pages,
            page_size,
            max_running_req,
            dtype,
            device,
            max_cached_states=max_cached_states,
            unified_memory=unified_memory,
        )

    from .mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("hybrid")
def create_hybrid_cache(device: torch.device):
    from minisgl.core import get_global_ctx

    from .hybrid_cache import HybridPrefixCache

    return HybridPrefixCache(device, get_global_ctx().kv_cache.recurrent)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
