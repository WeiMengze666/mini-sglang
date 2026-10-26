from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.kvcache.hybrid_cache import HybridPrefixCache
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.kvcache.unified import UnifiedMemory


@pytest.fixture(params=[False, True])
def cache(monkeypatch, request):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=2))
    arena = UnifiedMemory(64, 256, 576, "cpu") if request.param else None
    states = RecurrentStatePool(
        (0, 2), 3, 2, 3, 4, torch.float32, "cpu", max_cached_states=3, arena=arena
    )
    return HybridPrefixCache(torch.device("cpu"), states)


def commit(cache, slot, uid, length, value):
    states = cache.states
    states.begin_request(slot, uid, 0)
    for layer in (0, 2):
        state = states.get_kda_state(layer, slot)
        state.conv.fill_(value)
        state.recurrent.fill_(value * 2)
    states.finish_request(slot, uid, length)
    return SimpleNamespace(table_idx=slot, uid=uid, cached_len=length)


def test_restore_is_private_and_snapshot_survives_working_slot_reuse(cache):
    req = commit(cache, 0, 100, 6, 3)
    ids = torch.arange(6, dtype=torch.int32)
    handle = cache.insert_prefix(ids, ids).handle
    cache.cache_state(req, handle)
    cache.states.release_request(0, 100)
    matched = cache.match_prefix(ids).cuda_handle
    assert matched.cached_len == 6
    cache.lock_handle(matched)
    cache.restore_state(1, 101, matched)
    cache.states.begin_request(1, 101, 6)
    cache.states.get_kda_state(0, 1).recurrent.zero_()
    cache.states.finish_request(1, 101, 7)
    cache.restore_state(2, 102, matched)
    cache.states.begin_request(2, 102, 6)
    assert cache.states.get_kda_state(0, 2).recurrent.eq(6).all()
    cache.lock_handle(matched, unlock=True)


def test_mid_edge_fork_replays_from_nearest_checkpoint_and_plants_one(cache):
    ids = torch.arange(6, dtype=torch.int32)
    cache.cache_state(commit(cache, 0, 100, 6, 3), cache.insert_prefix(ids, ids).handle)
    fork = torch.tensor([0, 1, 2, 3, 99], dtype=torch.int32)
    matched = cache.match_prefix(fork).cuda_handle
    assert matched.cached_len == 0 and matched.replay_until == 4
    # Replaying to the aligned fork creates a checkpoint on the newly split node.
    handle = cache.insert_prefix(fork[:4], torch.arange(16, 20, dtype=torch.int32)).handle
    cache.cache_state(commit(cache, 1, 101, 4, 7), handle)
    matched = cache.match_prefix(fork).cuda_handle
    assert matched.cached_len == 4 and matched.replay_until == 0
    torch.testing.assert_close(matched.get_matched_indices(), ids[:4])
    # The old descendant checkpoint is still at length 6 after the radix split.
    assert cache.match_prefix(ids).cuda_handle.cached_len == 6


def test_eviction_releases_state_and_protected_checkpoints_are_not_recycled(cache):
    for slot in range(3):
        ids = torch.tensor([slot * 10, 1], dtype=torch.int32)
        handle = cache.insert_prefix(ids, ids).handle
        cache.cache_state(commit(cache, slot, slot, 2, slot + 1), handle)
    protected = cache.match_prefix(torch.tensor([0, 1], dtype=torch.int32)).cuda_handle
    cache.lock_handle(protected)
    cache.evict(4)
    assert len(cache.checkpoints) == len(cache.states._checkpoints) == 1
    cache.restore_state(0, 0, protected)
    cache.lock_handle(protected, unlock=True)
    cache.evict(2)
    assert not cache.checkpoints and not cache.states._checkpoints


def test_snapshot_requires_committed_state_and_alignment(cache):
    states = cache.states
    states.begin_request(0, 1, 0)
    with pytest.raises(RuntimeError, match="committed"):
        states.snapshot(0, 1)
    states.finish_request(0, 1, 3)
    ids = torch.arange(3, dtype=torch.int32)
    cache.cache_state(
        SimpleNamespace(table_idx=0, uid=1, cached_len=3), cache.insert_prefix(ids, ids).handle
    )
    assert not cache.checkpoints


def test_unified_pressure_preserves_a_locked_checkpoint_until_private_restore(cache):
    if cache.states.arena is None:
        pytest.skip("Unified-memory pressure only")
    ids = torch.arange(2, dtype=torch.int32)
    handle = cache.insert_prefix(ids, ids).handle
    cache.cache_state(commit(cache, 0, 100, 2, 3), handle)
    matched = cache.match_prefix(ids).cuda_handle
    cache.lock_handle(matched)
    cache.release_checkpoint_memory(cache.states.arena.buffer.numel())
    assert len(cache.checkpoints) == 1
    cache.restore_state(1, 101, matched)
    cache.release_checkpoint_memory(cache.states.arena.buffer.numel())
    assert not cache.checkpoints  # KV stays locked; private recurrent state is independent.
    cache.states.begin_request(1, 101, 2)
    assert cache.states.get_kda_state(0, 1).recurrent.eq(6).all()
    cache.lock_handle(matched, unlock=True)
