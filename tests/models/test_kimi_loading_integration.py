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
    values = dict(engine._iter_weights(SimpleNamespace(use_dummy_weight=True)))
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
    result = dict(engine._iter_weights(SimpleNamespace(use_dummy_weight=False, model_path="x")))
    assert result["layers.0.weight"].dtype == torch.float32
    assert result["layers.0.scale"] is raw[1][1]


def test_capacity_preflight_accounts_per_rank_fixed_state(monkeypatch):
    import minisgl.engine.engine as engine_module

    pool = SimpleNamespace(cache_bytes_per_token=None, state_bytes_per_request=None)

    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args: None)

    engine = object.__new__(Engine)
    engine.model = Model()
    engine.dtype = torch.bfloat16
    seen = []
    monkeypatch.setattr(pool, "cache_bytes_per_token", lambda *args: 4)
    monkeypatch.setattr(
        pool, "state_bytes_per_request", lambda *args, tp_size: seen.append(tp_size) or 16
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(cache_config=pool),
        max_running_req=2,
        tp_info=SimpleNamespace(rank=0, size=2),
        page_size=2,
        memory_ratio=0.9,
        num_page_override=None,
    )
    # 52 weight bytes +32 state +24 cache =108, budget90.
    with pytest.raises(MemoryError, match="before|needs at least"):
        engine._check_capacity(100, config)
    engine._check_capacity(1000, config)
    assert seen == [2, 2]


@pytest.mark.parametrize(
    "before,after,fixed,ratio",
    [
        ((100000, 101000), (60000, 59000), (1000, 1000), 0.9),
        ((100000, 101000), (59000, 60000), (1000, 1000), 0.9),
        ((100000, 101000), (60000, 60100), (1000, 3000), 0.9),
        ((100000, 101000), (60000, 59000), (1000, 3000), 1.0),
        ((100000,), (60000,), (1000,), 0.9),
    ],
)
def test_kimi_capacity_reduces_paired_rank_budgets(monkeypatch, before, after, fixed, ratio):
    import minisgl.engine.engine as module

    pool = SimpleNamespace(cache_bytes_per_token=None, state_bytes_per_request=None)

    unit = 1024**2
    before, after, fixed = [tuple(v * unit for v in values) for values in (before, after, fixed)]
    budgets = [
        int(ratio * old) - (old - new) - state for old, new, state in zip(before, after, fixed)
    ]
    shared = min(budgets)
    expected_pages = shared // unit - 1
    monkeypatch.setattr(module.logger, "info", lambda *args: None)
    monkeypatch.setattr(pool, "cache_bytes_per_token", lambda *args: unit)
    group = object()
    calls = []
    for rank in range(len(before)):
        engine = object.__new__(Engine)
        engine.dtype = torch.bfloat16
        engine.tp_cpu_group = group
        engine._initial_free_memory = before[rank]

        def snapshot():
            engine._local_free_memory = after[rank]
            return min(after), max(after)

        engine._sync_get_memory = snapshot
        monkeypatch.setattr(pool, "state_bytes_per_request", lambda *args, **kwargs: fixed[rank])

        def reduce_budget(tensor, op, group):
            assert tensor.device.type == "cpu" and tensor.dtype == torch.int64
            assert tensor.item() == budgets[rank]  # pairing is checked BEFORE reduction
            assert op == torch.distributed.ReduceOp.MIN and group is engine.tp_cpu_group
            calls.append(rank)
            tensor.fill_(shared)

        monkeypatch.setattr(torch.distributed, "all_reduce", reduce_budget)
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                cache_config=pool, head_dim=4, num_kv_heads=1, num_layers=1
            ),
            tp_info=SimpleNamespace(size=len(before)),
            page_size=1,
            max_running_req=1,
            num_page_override=None,
            memory_ratio=ratio,
        )
        assert engine._determine_num_pages(min(before), config) == expected_pages
        # Include the dummy page: every rank retains its own ratio reserve.
        used_cache = (expected_pages + 1) * unit
        assert after[rank] - fixed[rank] - used_cache >= before[rank] - int(ratio * before[rank])
        config.num_page_override = expected_pages
        assert engine._determine_num_pages(min(before), config) == expected_pages
        config.num_page_override += 1
        with pytest.raises(MemoryError, match="post-load memory budget"):
            engine._determine_num_pages(min(before), config)
    assert len(calls) == 3 * len(before)


def test_separate_minima_counterexample_and_error_bound():
    unit = 1024**2
    before = [100000 * unit, 101000 * unit]
    after = [60000 * unit, 59000 * unit]
    fixed = 1000 * unit
    old = int(0.9 * min(before)) - (min(before) - min(after)) - fixed
    safe = min(int(0.9 * a) - (a - b) - fixed for a, b in zip(before, after))
    assert old == 48000 * unit and safe == 47900 * unit
    assert old - safe == 100 * unit
    assert max(before) - min(before) < 2 * 1024**3
    assert max(after) - min(after) < 2 * 1024**3
    # Wrong reserve still does NOT exceed the currently measured free memory.
    assert old + fixed <= min(after)
    # For homogeneous fixed costs: 0 <= error <= (1-ratio)*initial spread +1 byte.
    for ratio in (0.25, 0.5, 0.9, 1.0):
        for a0 in range(20, 30):
            for a1 in range(20, 30):
                for b0, b1 in ((8, 9), (10, 7), (8, 8)):
                    a, b = (a0, a1), (b0, b1)
                    separate = int(ratio * min(a)) - min(a) + min(b)
                    paired = min(int(ratio * x) - x + y for x, y in zip(a, b))
                    assert 0 <= separate - paired <= (1 - ratio) * (max(a) - min(a)) + 1


def test_memory_snapshot_retains_local_value_before_minmax_reduce(monkeypatch):
    import minisgl.engine.engine as module

    engine = object.__new__(Engine)
    engine.device = torch.device("cpu")
    engine.tp_cpu_group = object()
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(module, "get_free_memory", lambda device: 1100)

    def reduce_extrema(tensor, **kwargs):
        assert tensor.tolist() == [1100, -1100]
        tensor.copy_(torch.tensor([1000, -1200]))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_extrema)
    assert engine._sync_get_memory() == (1000, 1200)
    assert engine._local_free_memory == 1100


def test_non_kimi_capacity_path_is_unchanged(monkeypatch):
    import minisgl.engine.engine as module

    engine = object.__new__(Engine)
    engine.dtype = torch.bfloat16
    engine._sync_get_memory = lambda: (700, 900)
    monkeypatch.setattr(module.logger, "info", lambda *args: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("Unexpected collective"),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(cache_config=None, head_dim=4, num_kv_heads=1, num_layers=1),
        tp_info=SimpleNamespace(size=2),
        page_size=1,
        num_page_override=None,
        memory_ratio=0.9,
    )
    assert engine._determine_num_pages(1000, config) == 50
    config.num_page_override = 999
    assert engine._determine_num_pages(1000, config) == 999
