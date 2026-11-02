"""Capability routing for optional distributed backends."""

from types import SimpleNamespace

import torch
from minisgl.distributed import FallbackDistributedImpl
from minisgl.distributed.mnnvl import MNNVLAllReduceProvider, MNNVLDistributedImpl


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


def test_mnnvl_adapter_exposes_optional_fused_residual():
    class Provider:
        def all_reduce(self, tensor):
            return tensor + 1

        def all_gather(self, tensor):
            return tensor

        def supports_all_reduce_residual(self, tensor, residual, *, norm=None):
            return residual is not None and norm is not None

        def all_reduce_residual(self, tensor, residual=None, *, norm=None):
            return norm(tensor + 1 + residual)

    backend = MNNVLDistributedImpl(Provider())
    x = torch.zeros(2)
    assert backend.all_reduce(x).eq(1).all()
    assert backend.supports_all_reduce_residual(x, x, norm=torch.neg)
    assert backend.all_reduce_residual(x, x, norm=torch.neg).eq(-1).all()


def test_mnnvl_custom_allreduce_provider_uses_external_collective(monkeypatch):
    class Comm:
        disabled = False

        def should_custom_ar(self, tensor):
            return tensor.numel() == 2

        def custom_all_reduce(self, tensor):
            return tensor + 3

    provider = MNNVLAllReduceProvider(Comm())
    assert provider.all_reduce(torch.zeros(2)).eq(3).all()

    def unexpected(*args, **kwargs):
        raise AssertionError("Torch fallback must not run for supported input")

    monkeypatch.setattr("torch.distributed.all_reduce", unexpected)
