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


def test_path_cap_refreshes_checkpoints_while_request_keeps_kv_locked(cache):
    cache.max_states_per_path = 2
    ids = torch.arange(8, dtype=torch.int32)
    req = commit(cache, 0, 100, 2, 1)
    handle = cache.insert_prefix(ids[:2], ids[:2]).handle
    cache.cache_state(req, handle)
    cache.lock_handle(handle)
    for length in (4, 6, 8):
        cache.states.begin_request(0, 100, req.cached_len)
        for layer in (0, 2):
            cache.states.get_kda_state(layer, 0).recurrent.fill_(length)
        cache.states.finish_request(0, 100, length)
        req.cached_len = length
        advanced = cache.insert_prefix(ids[:length], ids[:length]).handle
        for node, checkpoint in cache.checkpoints.values():
            node.timestamp = checkpoint.length  # deterministic recency
        cache.cache_state(req, advanced)
        cache.lock_handle(handle, unlock=True)
        cache.lock_handle(advanced)
        handle = advanced
    assert sorted(state.length for _, state in cache.checkpoints.values()) == [6, 8]
    assert cache.match_prefix(ids).cuda_handle.cached_len == 8
    cache.lock_handle(handle, unlock=True)


def test_capacity_recycles_restored_state_but_preserves_pending_restore(cache):
    handles = []
    for slot in range(3):
        ids = torch.tensor([slot * 10, slot * 10 + 1], dtype=torch.int32)
        handle = cache.insert_prefix(ids, ids).handle
        cache.cache_state(commit(cache, slot, 100 + slot, 2, 10 + slot), handle)
        handles.append(handle)
    restored = cache.match_prefix(torch.tensor([0, 1], dtype=torch.int32)).cuda_handle
    pending = cache.match_prefix(torch.tensor([10, 11], dtype=torch.int32)).cuda_handle
    cache.lock_handle(restored)
    cache.lock_handle(pending)
    cache.lock_handle(handles[2])
    cache.restore_state(0, 100, restored)
    cache.states.begin_request(2, 102, 2)
    cache.states.get_kda_state(0, 2).recurrent.fill_(99)
    cache.states.finish_request(2, 102, 4)
    ids = torch.tensor([20, 21, 22, 23], dtype=torch.int32)
    advanced = cache.insert_prefix(ids, ids).handle
    for handle, timestamp in zip(handles, (1, 0, 2)):
        handle.node.timestamp = timestamp
    cache.cache_state(SimpleNamespace(table_idx=2, uid=102, cached_len=4), advanced)
    assert len(cache.checkpoints) == 3
    assert restored.node.uuid not in cache.checkpoints
    assert pending.node.uuid in cache.checkpoints
    assert cache.match_prefix(ids).cuda_handle.cached_len == 4
    # Dropping a tree checkpoint cannot change a request's private working state.
    cache.states.begin_request(0, 100, 2)
    assert cache.states.get_kda_state(0, 0).recurrent.eq(20).all()
    cache.states.finish_request(0, 100, 3)
    cache.restore_state(1, 101, pending)
    cache.states.begin_request(1, 101, 2)
    assert cache.states.get_kda_state(0, 1).recurrent.eq(22).all()
    cache.states.finish_request(1, 101, 3)
    for handle in (restored, pending, handles[2]):
        cache.lock_handle(handle, unlock=True)
