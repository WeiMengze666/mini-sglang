"""State eviction follows LRU while queued restores retain their private copy source."""

from types import SimpleNamespace

import torch
from minisgl.core import Batch
from test_checkpoint_writes import admit, allocate_pages, make_manager
from test_hybrid_prefix import cache as cache
from test_hybrid_prefix import commit


def populated_manager(monkeypatch):
    manager = make_manager(monkeypatch, False)
    states = manager.pool.recurrent
    states.max_cached_states = 1
    req = admit(manager, end=4)
    allocate_pages(manager, req)
    states.begin_request(req.table_idx, req.uid, 0)
    states.finish_request(req.table_idx, req.uid, 4)
    req.cached_len = 4
    manager.cache_req(req, finished=False)
    return manager, req, next(iter(states._checkpoints))


def test_continuing_reader_does_not_prevent_lru_checkpoint_reclamation(monkeypatch):
    manager, first, old_slot = populated_manager(monkeypatch)
    states = manager.pool.recurrent
    assert manager.current_handle(first).node.ref_count > 0  # KV remains protected.
    assert not manager.prefix_cache._pending_restores
    req = admit(manager, slot=1, uid=20, end=11)
    assert req.cached_len == 4
    batch = Batch([req], "prefill")
    backend = SimpleNamespace(checkpoint_quantum=lambda batch: 4)
    (plan,) = manager.plan_checkpoints(batch, backend)
    assert plan.start_len == 4 and plan.checkpoint.length == 8
    assert old_slot not in states._checkpoints
    assert plan.checkpoint.slot != old_slot  # Generation changes even if storage is reused.
    manager.release_checkpoint_plan((plan,))
    assert not states._pending


def test_queued_restore_is_not_evicted_to_make_an_optional_checkpoint(monkeypatch):
    manager, _, old_slot = populated_manager(monkeypatch)
    states = manager.pool.recurrent
    protected = manager.prefix_cache.match_prefix(torch.arange(4, dtype=torch.int32)).cuda_handle
    manager.lock(protected)
    req = admit(manager, slot=1, uid=20, end=11)
    backend = SimpleNamespace(checkpoint_quantum=lambda batch: 4)
    assert manager.plan_checkpoints(Batch([req], "prefill"), backend) == ()
    assert old_slot in states._checkpoints
    manager.unlock(protected)
    (plan,) = manager.plan_checkpoints(Batch([req], "prefill"), backend)
    assert old_slot not in states._checkpoints
    manager.release_checkpoint_plan((plan,))


def test_state_pressure_uses_lru_including_branching_nodes(cache):
    cache.states.max_cached_states = 3
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
