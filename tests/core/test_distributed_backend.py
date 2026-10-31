"""Capability routing for optional distributed backends."""

from types import SimpleNamespace

import torch
from minisgl.distributed import FallbackDistributedImpl


def test_fallback_backend_selects_by_tensor_capability():
    events = []
    primary = SimpleNamespace(
        supports=lambda tensor: tensor.numel() == 4,
        all_reduce=lambda tensor: events.append("primary") or tensor + 1,
        all_gather=lambda tensor: tensor,
        close=lambda: events.append("primary-close"),
    )
    fallback = SimpleNamespace(
        supports=lambda tensor: True,
        all_reduce=lambda tensor: events.append("fallback") or tensor + 2,
        all_gather=lambda tensor: tensor,
        close=lambda: events.append("fallback-close"),
    )
    backend = FallbackDistributedImpl(primary, fallback)

    assert backend.all_reduce(torch.zeros(4)).eq(1).all()
    assert backend.all_reduce(torch.zeros(3)).eq(2).all()
    backend.close()
    assert events == ["primary", "fallback", "primary-close", "fallback-close"]


def test_fallback_backend_routes_each_collective_by_capability():
    events = []
    primary = SimpleNamespace(
        supports=lambda tensor: True,
        supports_all_gather=lambda tensor: False,
        all_reduce=lambda tensor: events.append("primary-reduce") or tensor,
        all_gather=lambda tensor: events.append("primary-gather") or tensor,
        close=lambda: None,
    )
    fallback = SimpleNamespace(
        supports=lambda tensor: True,
        all_reduce=lambda tensor: events.append("fallback-reduce") or tensor,
        all_gather=lambda tensor: events.append("fallback-gather") or tensor,
        close=lambda: None,
    )
    backend = FallbackDistributedImpl(primary, fallback)

    backend.all_reduce(torch.zeros(2))
    backend.all_gather(torch.zeros(2))

    assert events == ["primary-reduce", "fallback-gather"]
