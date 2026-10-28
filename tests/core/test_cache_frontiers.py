"""A completed frontier stays immutable while the next batch mutates working state."""

import os
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.kvcache.unified import UnifiedMemory
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.getenv("MINISGL_RUN_CUDA_TESTS") != "1", reason="GPU tests require gpuq"
            ),
        ),
    ]
)
def device(request):
    return request.param


@pytest.fixture(params=[False, True])
def manager(monkeypatch, request, device):
    arena = UnifiedMemory(32, 256, 256, device) if request.param else None
    states = RecurrentStatePool(
        (0,),
        2,
        1,
        2,
        3,
        torch.float32,
        device,
        max_cached_states=2,
        max_staged_states=2,
        arena=arena,
    )
    pool = SimpleNamespace(
        arena=arena,
        recurrent=states,
        stage_state=lambda slot, uid, length: states.snapshot(
            slot, uid, staged=True, expected_len=length
        ),
        release_staged=states.release_staged,
        release_request=states.release_request,
    )
    ctx = core.Context(page_size=2)
    ctx.kv_cache = pool
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return CacheManager(
        32, 2, torch.full((2, 16), -1, dtype=torch.int32, device=device), "hybrid", pool
    )


def admit(manager, slot=0, uid=10):
    ids = torch.arange(8, dtype=torch.int32)
    handle = manager.prefix_cache.match_prefix(ids).cuda_handle
    manager.lock(handle)
    manager.restore_state(slot, uid, handle)
    req = Req(ids, slot, 0, 3, uid, SamplingParams(), handle)
    pages = manager._allocate(3)
    manager.page_table[slot, :6] = manager._page_to_token(pages)
    return req


def advance(manager, req, start, end, value):
    states = manager.pool.recurrent
    states.begin_request(req.table_idx, req.uid, start)
    state = states.get_kda_state(0, req.table_idx)
    state.conv.fill_(value)
    state.recurrent.fill_(value * 2)
    states.finish_request(req.table_idx, req.uid, end)
    req.cached_len = end


def test_old_frontier_is_donated_without_recopying_advanced_state(manager):
    req = admit(manager)
    advance(manager, req, 0, 4, 7)
    staged = manager.stage_states([req], [4], is_prefill=True)
    checkpoint = staged[req.uid]
    saved = manager.pool.recurrent._checkpoints[checkpoint.slot]
    advance(manager, req, 4, 5, 99)
    manager.cache_req(req, finished=True, frontier=4, checkpoint=checkpoint)
    manager.release_staged(staged)
    manager.release_state(req.table_idx, req.uid)
    assert len(manager.pool.recurrent._checkpoints) == 1
    assert manager.pool.recurrent._checkpoints[checkpoint.slot] is saved
    matched = manager.prefix_cache.match_prefix(req.input_ids).cuda_handle
    assert matched.cached_len == 4
    manager.lock(matched)
    manager.restore_state(1, 11, matched)
    manager.pool.recurrent.begin_request(1, 11, 4)
    restored = manager.pool.recurrent.get_kda_state(0, 1)
    assert restored.conv.eq(7).all() and restored.recurrent.eq(14).all()
    manager.release_state(1, 11)
    manager.unlock(matched)
    # The speculative fifth token occupied another private page, now reclaimed.
    manager.check_integrity()
    if manager.arena is not None:
        manager.arena.check_integrity()


def test_missing_snapshot_never_labels_advanced_state_as_an_old_frontier(manager):
    req = admit(manager)
    advance(manager, req, 0, 5, 9)
    manager.cache_req(req, finished=True, frontier=4)
    manager.release_state(req.table_idx, req.uid)
    assert not manager.prefix_cache.checkpoints
    assert manager.prefix_cache.match_prefix(req.input_ids).cuda_handle.cached_len == 0
    manager.check_integrity()


def test_next_chunk_uses_the_latest_lease_even_if_constructed_before_donation(manager):
    req = admit(manager)
    next_chunk = Req(req.input_ids, 0, 4, 3, req.uid, SamplingParams(), req.cache_handle)
    advance(manager, req, 0, 4, 7)
    staged = manager.stage_states([req], [4], is_prefill=True)
    advance(manager, next_chunk, 4, 6, 9)
    manager.cache_req(req, finished=False, frontier=4, checkpoint=staged[req.uid])
    manager.release_staged(staged)
    assert next_chunk.cache_handle.cached_len == 0
    assert manager.current_handle(next_chunk).cached_len == 4
    manager.cache_req(next_chunk, finished=True)
    manager.release_state(0, req.uid)
    assert manager.prefix_cache.match_prefix(req.input_ids).cuda_handle.cached_len == 6
    manager.check_integrity()


def test_staging_failure_releases_earlier_copies_and_capacity_is_reusable(manager):
    first, second = admit(manager), admit(manager, 1, 11)
    advance(manager, first, 0, 4, 1)
    advance(manager, second, 0, 5, 2)
    with pytest.raises(RuntimeError, match="frontier"):
        manager.stage_states([first, second], [4, 4], is_prefill=True)
    assert not manager.pool.recurrent._staged
    staged = manager.stage_states([first], [4], is_prefill=True)
    manager.release_staged(staged)
    assert not manager.pool.recurrent._checkpoints


def test_staged_frontier_does_not_survive_cancellation_or_slot_reuse(manager):
    req = admit(manager)
    advance(manager, req, 0, 4, 7)
    staged = manager.stage_states([req], [4], is_prefill=True)
    manager.release_state(0, req.uid)
    manager.pool.recurrent.reserve_request(0, req.uid + 1)
    manager.release_staged(staged)
    manager.pool.recurrent.begin_request(0, req.uid + 1, 0)
    assert not manager.pool.recurrent.get_kda_state(0, 0).recurrent.any()
    assert not manager.pool.recurrent._checkpoints


@pytest.mark.parametrize(
    "overlap,cached,cache_type,expected",
    [
        (False, 2, "hybrid", 6),
        (True, 2, "hybrid", 10),
        (True, 0, "hybrid", 4),
        (True, 2, "naive", 4),
    ],
)
def test_memory_budget_reserves_one_staged_frontier_per_running_request(
    overlap, cached, cache_type, expected
):
    from minisgl.distributed import DistributedInfo
    from minisgl.scheduler import SchedulerConfig

    config = SchedulerConfig(
        "unused",
        DistributedInfo(0, 1),
        torch.bfloat16,
        max_running_req=4,
        max_cached_states=cached,
        cache_type=cache_type,
        enable_cache_overlap=overlap,
    )
    assert config.state_slot_budget == expected
