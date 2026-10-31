"""Capability routing for optional distributed backends."""

from types import SimpleNamespace

import torch
from minisgl.distributed.impl import FallbackDistributedImpl


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
