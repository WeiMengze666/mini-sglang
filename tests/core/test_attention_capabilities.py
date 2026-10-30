"""Registered capabilities, rather than backend names, govern hybrid startup."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention import (
    SUPPORTED_ATTENTION_BACKENDS,
    create_attention_backend,
    register_attention_backend,
)
from minisgl.distributed import DistributedInfo
from minisgl.engine import EngineConfig
from minisgl.engine.engine import _adjust_config


@pytest.fixture
def backends(monkeypatch):
    import minisgl.distributed.info as info

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr(
        SUPPORTED_ATTENTION_BACKENDS, "_registry", dict(SUPPORTED_ATTENTION_BACKENDS._registry)
    )
    calls = []

    def create(config):
        calls.append(config)
        return SimpleNamespace(config=config)

    register_attention_backend("portable-test", stateful_layers=True, cuda_graphs=False)(create)
    register_attention_backend("native-test", stateful_layers=True)(create)
    register_attention_backend("dense-test")(create)
    return calls


def config(backend, graphs):
    value = EngineConfig(
        "unused",
        DistributedInfo(0, 1),
        torch.float32,
        attention_backend=backend,
        cuda_graph_bs=[1] if graphs else [],
    )
    value.__dict__["model_config"] = SimpleNamespace(
        cache_config=SimpleNamespace(state_bytes_per_request=lambda *args: 0),
        num_qo_heads=8,
        has_recurrent_state=True,
        is_moe=False,
    )
    return value


@pytest.mark.parametrize(
    "backend", ["native-test", "portable-test,native-test", "native-test,native-test"]
)
@pytest.mark.parametrize("graphs", [False, True])
def test_new_stateful_backend_requires_no_engine_name_change(backends, backend, graphs):
    value = config(backend, graphs)
    _adjust_config(value)
    assert value.attention_backend == backend
    assert value.cuda_graph_bs == ([1] if graphs else [])
    assert not backends  # Capability resolution must allocate no backend resources.


@pytest.mark.parametrize("backend", ["portable-test", "native-test,portable-test"])
def test_graphs_follow_decode_capability(backends, backend):
    _adjust_config(config(backend, False))
    with pytest.raises(ValueError, match="decode.*CUDA graphs"):
        _adjust_config(config(backend, True))
    assert not backends


@pytest.mark.parametrize(
    "backend", ["dense-test", "dense-test,native-test", "native-test,dense-test"]
)
def test_each_stage_must_support_stateful_layers(backends, backend):
    with pytest.raises(ValueError, match="stateful layers"):
        _adjust_config(config(backend, False))


def test_registration_still_constructs_through_the_common_factory(backends):
    value = object()
    actual = create_attention_backend("native-test", value)
    assert actual.config is value and backends == [value]
