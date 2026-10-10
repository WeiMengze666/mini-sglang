"""CPU checks for strict incremental assignment and engine capacity preflight."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.engine.engine import Engine
from minisgl.layers.base import BaseOP, OPList


class Leaf(BaseOP):
    def __init__(self):
        self.weight = torch.empty(2, 3, device="meta")
        self.scale = torch.empty(2, 1, dtype=torch.uint8, device="meta")


class Model(BaseOP):
    def __init__(self):
        self.layers = OPList([Leaf(), Leaf()])


def test_stream_assigns_each_item_before_reading_next():
    model = Model()
    values = {name: torch.ones_like(t, device="cpu") for name, t in model.state_dict().items()}

    def stream():
        for name, tensor in values.items():
            yield name, tensor
            assert model.state_dict()[name] is tensor

    model.load_weights(stream())
    assert all(t.device.type == "cpu" for t in model.state_dict().values())
    assert len(values) == 4  # caller-owned dictionary is not consumed


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "shape", "dtype"])
def test_stream_rejects_invalid_checkpoint(failure):
    model = Model()
    values = [(name, torch.ones_like(t, device="cpu")) for name, t in model.state_dict().items()]
    if failure == "missing":
        values.pop()
    elif failure == "duplicate":
        values.append(values[0])
    elif failure == "unknown":
        values.append(("bad.weight", torch.ones(1)))
    elif failure == "shape":
        values[0] = (values[0][0], torch.ones(1))
    else:
        values[1] = (values[1][0], values[1][1].float())
    with pytest.raises(ValueError):
        model.load_weights(iter(values))


def test_engine_dummy_packed_weights_and_native_float_dtypes():
    engine = object.__new__(Engine)
    engine.model = Model()
    engine.device = torch.device("cpu")
    values = dict(engine._iter_kimi_weights(SimpleNamespace(use_dummy_weight=True)))
    assert values["layers.0.scale"].dtype == torch.uint8
    assert values["layers.0.weight"].dtype == torch.float32
    engine.model.load_weights(values.items())


def test_engine_stream_only_casts_floating_parameters(monkeypatch):
    import minisgl.engine.engine as module

    engine = object.__new__(Engine)
    engine.model = Model()
    engine.device = torch.device("cpu")
    raw = [
        ("layers.0.weight", torch.ones(2, 3, dtype=torch.float64)),
        ("layers.0.scale", torch.ones(2, 1, dtype=torch.uint8)),
    ]
    monkeypatch.setattr(module, "load_weight", lambda *args: iter(raw))
    result = dict(
        engine._iter_kimi_weights(SimpleNamespace(use_dummy_weight=False, model_path="x"))
    )
    assert result["layers.0.weight"].dtype == torch.float32
    assert result["layers.0.scale"] is raw[1][1]


def test_capacity_preflight_accounts_per_rank_fixed_state(monkeypatch):
    import minisgl.engine.engine as engine_module
    import minisgl.kvcache.kimi_pool as pool

    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args: None)

    engine = object.__new__(Engine)
    engine.model = Model()
    engine.dtype = torch.bfloat16
    seen = []
    monkeypatch.setattr(pool, "kimi_cache_bytes_per_token", lambda *args: 4)
    monkeypatch.setattr(
        pool, "kimi_state_bytes_per_request", lambda *args, tp_size: seen.append(tp_size) or 16
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(kimi=object()),
        max_running_req=2,
        tp_info=SimpleNamespace(rank=0, size=2),
        page_size=2,
        memory_ratio=0.9,
        num_page_override=None,
    )
    # 52 weight bytes +32 state +24 cache =108, budget90.
    with pytest.raises(MemoryError, match="before|needs at least"):
        engine._check_kimi_capacity(100, config)
    engine._check_kimi_capacity(1000, config)
    assert seen == [2, 2]


def test_kimi_cache_capacity_uses_least_free_tp_rank(monkeypatch):
    import minisgl.engine.engine as module
    import minisgl.kvcache.kimi_pool as pool

    monkeypatch.setattr(module.logger, "info", lambda *args: None)
    monkeypatch.setattr(pool, "kimi_cache_bytes_per_token", lambda *args: 10)
    monkeypatch.setattr(pool, "kimi_state_bytes_per_request", lambda *args, **kwargs: 20)
    engine = object.__new__(Engine)
    engine.dtype = torch.bfloat16
    engine._sync_get_memory = lambda: (700, 900)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_kimi=True, kimi=object(), head_dim=4, num_kv_heads=1, num_layers=1
        ),
        tp_info=SimpleNamespace(size=2),
        page_size=1,
        max_running_req=2,
        num_page_override=None,
        memory_ratio=0.9,
    )
    # Initial minimum1000 -> minimum700: 300 model +40 state, /10 minusdummy.
    assert engine._determine_num_pages(1000, config) == 55
