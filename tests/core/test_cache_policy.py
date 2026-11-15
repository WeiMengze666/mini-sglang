"""State eviction follows LRU while queued restores retain their private copy source."""

import torch
from test_checkpoint_writes import admit, allocate_pages, make_manager
from test_hybrid_prefix import cache as cache
from test_hybrid_prefix import commit


def populated_manager(monkeypatch):
    manager = make_manager(monkeypatch, False)
    states = manager.pool.recurrent
    assert states.checkpoint_enabled
    req = admit(manager, end=4)
    allocate_pages(manager, req)
    states.begin_request(req.table_idx, req.uid, 0)
    states.finish_request(req.table_idx, req.uid, 4)
    req.cached_len = 4
    manager.cache_req(req, finished=False)
    return manager, req, next(iter(states._checkpoints))


def test_continuing_reader_keeps_state_component_anchor_protected(monkeypatch):
    manager, first, old_slot = populated_manager(monkeypatch)
    states = manager.pool.recurrent
    assert manager.current_handle(first).node.ref_count > 0
    assert manager.prefix_cache._state_locks
    assert not manager.prefix_cache.evict_checkpoint()
    assert old_slot in states._checkpoints
    manager.unlock(manager.current_handle(first))
    assert manager.prefix_cache.evict_checkpoint()
    assert old_slot not in states._checkpoints


def test_two_receipts_require_two_matching_unlocks(monkeypatch):
    manager, first, old_slot = populated_manager(monkeypatch)
    protected = manager.prefix_cache.match_prefix(torch.arange(4, dtype=torch.int32)).cuda_handle
    manager.lock(protected)
    manager.unlock(manager.current_handle(first))
    assert not manager.prefix_cache.evict_checkpoint()
    manager.unlock(protected)
    assert manager.prefix_cache.evict_checkpoint()
    assert old_slot not in manager.pool.recurrent._checkpoints


def test_state_pressure_uses_lru_including_branching_nodes(cache):
    assert cache.states.checkpoint_enabled
    prefixes = ([0, 1], [0, 1, 2, 3], [0, 1, 4, 5])
    nodes = []
    for slot, prefix in enumerate(prefixes):
        ids = torch.tensor(prefix, dtype=torch.int32)
        handle = cache.insert_prefix(ids, ids).handle
        cache.cache_state(commit(cache, slot, slot + 1, len(ids), slot + 1), handle)
        nodes.append(handle.node)
    assert len(nodes[0].children) == 2
    for age, node in enumerate(nodes, 1):
        node.timestamp = age
    assert cache.evict_checkpoint()
    assert nodes[0].uuid not in cache.checkpoints
    assert all(node.uuid in cache.checkpoints for node in nodes[1:])
