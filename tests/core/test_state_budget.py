from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine
from minisgl.kvcache.config import state_slot_capacity
from minisgl.scheduler import SchedulerConfig


def test_state_ratio_budgets_one_total_physical_pool():
    unit = 1 << 20
    total = state_slot_capacity(1310 * unit + 4096, unit, ratio=0.31)
    assert total == 310
    assert total * unit <= (1310 * unit + 4096) * 0.31 / 1.31


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_state_ratio(ratio):
    with pytest.raises(ValueError, match="finite and positive"):
        state_slot_capacity(10000, 100, ratio=ratio)


def test_state_ratio_requires_a_physical_slot():
    with pytest.raises(MemoryError, match="physical state slot"):
        state_slot_capacity(100, 100, ratio=1)


@pytest.mark.parametrize("overlap,expected", [(False, 8), (True, 12)])
def test_automatic_preload_estimate_reserves_only_mandatory_state(overlap, expected):
    config = SchedulerConfig(
        "unused",
        DistributedInfo(0, 1),
        torch.bfloat16,
        max_running_req=4,
        state_memory_ratio=0.31,
        enable_cache_overlap=overlap,
    )
    assert config.state_slot_budget == expected


def plan(monkeypatch, *, ratio=1.0, overlap=False, unified=False, pages=None, collective=None):
    engine = object.__new__(Engine)
    engine.dtype = torch.bfloat16
    engine.tp_cpu_group = None
    engine._initial_free_memory = 10000
    engine._local_free_memory = 8000
    engine._sync_get_memory = lambda: (8000, 8000)
    monkeypatch.setattr(torch.distributed, "all_reduce", collective or (lambda budget, **kw: None))
    layout = SimpleNamespace(
        cache_bytes_per_page=lambda *a: 200, state_bytes_per_request=lambda *a, **kw: 100
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(cache_config=layout),
        num_page_override=pages,
        cache_dtype=torch.bfloat16,
        logical_page_size=64,
        tp_info=DistributedInfo(0, 1),
        recurrent_state_dtype=torch.bfloat16,
        memory_ratio=0.9,
        enable_unified_memory=unified,
        state_slot_budget=4 + ((8 if overlap else 4) if ratio is not None else 0),
        state_memory_ratio=ratio,
        max_running_req=4,
        enable_cache_overlap=overlap,
        cache_type="hybrid",
    )
    pages = engine._determine_num_pages(10000, config)
    return pages, engine.state_slot_capacity


@pytest.mark.parametrize("overlap", [False, True])
def test_static_engine_budget_uses_one_common_byte_limit(monkeypatch, overlap):
    pages, slots = plan(monkeypatch, overlap=overlap)
    assert (pages, slots) == (16, 35)
    assert (pages + 1) * 200 + slots * 100 <= 7000


def test_state_and_token_pools_follow_least_safe_rank(monkeypatch):
    expected_collectives = [(35, 31), (3900, 2800)]

    def reduce(tensor, **options):
        local, shared = expected_collectives.pop(0)
        assert tensor.item() == local
        tensor.fill_(shared)

    pages, slots = plan(monkeypatch, collective=reduce)
    assert not expected_collectives
    assert (pages, slots) == (13, 31)
    assert (pages + 1) * 200 + slots * 100 == 5900


def test_without_prefix_ratio_only_working_state_is_reserved(monkeypatch):
    assert plan(monkeypatch, ratio=None) == (32, None)


@pytest.mark.parametrize("override,expected", [(None, (26, 13)), (23, (23, 12))])
def test_unified_state_limit_uses_actual_aligned_arena(monkeypatch, override, expected):
    assert plan(monkeypatch, unified=True, pages=override) == expected


def test_explicit_pages_cannot_overspend_state_budget(monkeypatch):
    with pytest.raises(MemoryError, match="least-safe TP"):
        plan(monkeypatch, pages=17)


def test_cli_uses_ratio_derived_checkpoint_budget():
    from minisgl.server.args import parse_args

    config, _ = parse_args(
        ["--model-path", "unused", "--dtype", "bfloat16", "--mamba-full-memory-ratio", "0.31"]
    )
    assert config.state_memory_ratio == 0.31
    assert not hasattr(config, "max_cached_states")


@pytest.mark.parametrize("limit", [0, 8, 9])
@pytest.mark.parametrize("reverse", [False, True])
def test_cli_rejects_removed_fixed_capacity_option(limit, reverse):
    from minisgl.server.args import parse_args

    base = ["--model-path", "unused", "--dtype", "bfloat16"]
    ratio, explicit = ["--mamba-full-memory-ratio", "0.31"], ["--max-cached-states", str(limit)]
    options = [*explicit, *ratio] if reverse else [*ratio, *explicit]
    with pytest.raises(SystemExit):
        parse_args([*base, *options])


def test_state_footprints_can_differ_between_ranks(monkeypatch):
    # Local state uses100B, peer300B. Both get7000B, so the peer permits
    # eleven shared slots. Each rank deducts its own physical byte size.
    expected_collectives = [(35, 11), (5900, 3700)]

    def reduce(tensor, **options):
        local, shared = expected_collectives.pop(0)
        assert tensor.item() == local
        tensor.fill_(shared)

    pages, slots = plan(monkeypatch, collective=reduce)
    assert not expected_collectives
    assert (pages, slots) == (17, 11)
    assert (pages + 1) * 200 + slots * 300 <= 7000


def test_insufficient_peer_state_budget_fails_collectively(monkeypatch):
    calls = []

    def reduce(tensor, **options):
        calls.append(tensor.item())
        tensor.zero_()

    with pytest.raises(MemoryError, match="State budget on a TP rank"):
        plan(monkeypatch, collective=reduce)
    assert calls == [35]
