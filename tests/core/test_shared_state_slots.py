"""Shared-pool lifecycle counterexamples; CPU tensors, no kernel imitation."""

from dataclasses import replace
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.kvcache.hybrid_cache import HybridPrefixCache
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.kvcache.unified import UnifiedMemory
from minisgl.scheduler.cache import CacheManager


def pool(total=8, requests=4, tracking=2, *, arena=None, abort=None):
    return RecurrentStatePool(
        (0, 2),
        requests,
        1,
        2,
        3,
        torch.float32,
        "cpu",
        state_slot_capacity=total,
        tracking_buffers=tracking,
        arena=arena,
        device_abort=abort,
    )


def committed(p, uid=1, length=4, table=0, start=0):
    p.begin_request(table, uid, start)
    for layer in (0, 2):
        state = p.get_kda_state(layer, table)
        state.conv.fill_(length)
        state.recurrent.fill_(2 * length)
    p.finish_request(table, uid, length)
    return SimpleNamespace(table_idx=table, uid=uid, cached_len=length)


@pytest.mark.parametrize("arena_pressure", [False, True])
def test_three_slot_admission_is_atomic_under_metadata_and_arena_pressure(arena_pressure):
    arena = UnifiedMemory(2, 256, 256, "cpu") if arena_pressure else None
    p = pool(total=8 if arena_pressure else 2, arena=arena)
    before = p.state_usage.copy()
    with pytest.raises(MemoryError):
        p.reserve_request(0, 7)
    assert p.state_usage == before
    assert p._owners == [None] * 4 and not any(p._tracking_slots)
    if arena is not None:
        assert not arena._states
        arena.check_integrity()


def test_continuing_owner_needs_no_second_three_slot_admission():
    p = pool(total=3)
    p.reserve_request(0, 7)
    before = p.state_usage.copy()
    assert before["working"] == 1 and before["tracking"] == 2 and before["free"] == 0
    assert p.request_slot_demand(0, 7) == 0
    p.reserve_request(0, 7)
    assert p.state_usage == before
    p.release_request(0, 6)  # stale request UID cannot free the current owner
    assert p.state_usage == before


@pytest.mark.parametrize("unified", [False, True])
def test_idle_request_capacity_is_lent_to_tree_and_cow_keeps_source_receipt(monkeypatch, unified):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=2))
    arena = UnifiedMemory(32, 256, 256, "cpu") if unified else None
    p = pool(total=8, arena=arena)
    tree = HybridPrefixCache(torch.device("cpu"), p)
    handles = []
    # Six checkpoints survive with a four-entry request table. There is no
    # startup 4-working + 8-tracking bank or independently fixed cache quota.
    for uid in range(6):
        req = committed(p, uid=uid)
        ids = torch.tensor([uid * 10, 1, 2, 3], dtype=torch.int32)
        handle = tree.insert_prefix(ids, ids).handle
        tree.cache_state(req, handle, finished=True)
        p.release_request(0, uid)
        handles.append(handle)
    assert p.state_usage == {
        "total": 8,
        "free": 2,
        "working": 0,
        "tracking": 0,
        "private_checkpoint": 0,
        "tree_checkpoint": 6,
        "deferred_release": 0,
        "pending_writes": 0,
    }
    source = tree.match_prefix(torch.tensor([0, 1, 2, 3], dtype=torch.int32)).cuda_handle
    source.node.timestamp = -1  # would be evicted first without its receipt
    source_cp = tree.checkpoints[source.node.uuid][1]
    tree.lock_handle(source)
    tree.restore_state(1, 20, source)  # needs3; evicts another tree slot first
    assert source.node.uuid in tree.checkpoints
    assert p.state_usage["tree_checkpoint"] == 5 and p.state_usage["free"] == 0
    p.begin_request(1, 20, 4)
    assert p.get_kda_state(0, 1).recurrent.eq(8).all()
    p.finish_request(1, 20, 5)
    while tree.evict_checkpoint():
        pass
    assert list(tree.checkpoints) == [source.node.uuid]  # COW does not end lock lifetime
    forged_receipt = replace(source)
    with pytest.raises(RuntimeError, match="receipt"):
        tree.lock_handle(forged_receipt, unlock=True)
    assert tree.checkpoints[source.node.uuid][1] == source_cp
    tree.lock_handle(source, unlock=True)
    assert tree.evict_checkpoint()
    p.release_request(1, 20)
    assert p.state_usage["free"] == 8


def test_replacement_failure_preserves_donor_then_finished_transfer_needs_no_slot():
    p = pool(total=3)
    committed(p)
    cp = p.snapshot(0, 1, staged=True)
    physical = p._checkpoint_storage[cp.slot]
    before = p.state_usage.copy()
    with pytest.raises(MemoryError):
        p.adopt_staged(cp)
    assert p.state_usage == before
    assert p._request_checkpoints[cp.slot] == (0, 1)
    assert physical in p._tracking_slots[0]
    p.adopt_staged(cp, finished=True)
    assert p._checkpoint_storage[cp.slot] == physical
    assert physical not in p._tracking_slots[0]
    assert cp.slot not in p._request_checkpoints
    p.release_request(0, 1)
    assert p.state_usage["tree_checkpoint"] == 1 and p.state_usage["free"] == 2
    p.drop_checkpoint(cp)
    assert p.state_usage["free"] == 3


def test_ongoing_transfer_keeps_generation_and_pointer_and_replaces_tracking_first():
    p = pool(total=4)
    committed(p)
    cp = p.snapshot(0, 1, staged=True)
    physical = p._checkpoint_storage[cp.slot]
    pointer = p._checkpoints[cp.slot][0].data_ptr()
    p.adopt_staged(cp)
    assert p._checkpoint_storage[cp.slot] == physical
    assert p._checkpoints[cp.slot][0].data_ptr() == pointer
    assert len(p._tracking_slots[0]) == 2 and physical not in p._tracking_slots[0]
    assert p.state_usage["free"] == 0 and p.state_usage["tree_checkpoint"] == 1


def test_duplicate_insert_releases_unused_donor_exactly_once(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=2))
    p = pool()
    tree = HybridPrefixCache(torch.device("cpu"), p)
    req = committed(p)
    ids = torch.arange(4, dtype=torch.int32)
    handle = tree.insert_prefix(ids, ids).handle
    tree.cache_state(req, handle)
    existing = tree.checkpoints[handle.node.uuid][1]
    unused = p.snapshot(0, 1, staged=True)
    tree.cache_state(req, handle, checkpoint=unused)
    p.release_staged(unused)
    before = p.state_usage.copy()
    p.release_staged(unused)
    assert p.state_usage == before and tree.checkpoints[handle.node.uuid][1] == existing
    p.release_request(0, 1)
    assert p.state_usage["tree_checkpoint"] == 1 and p.state_usage["free"] == 7


def test_tombstone_anchor_lock_protects_later_insert_until_matching_unlock(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=2))
    p = pool()
    tree = HybridPrefixCache(torch.device("cpu"), p)
    ids = torch.arange(4, dtype=torch.int32)
    handle = tree.insert_prefix(ids, ids).handle  # KV exists; no state yet
    tree.lock_handle(handle, for_restore=False)
    tree.cache_state(committed(p), handle)
    assert not tree.evict_checkpoint()
    tree.lock_handle(handle, unlock=True)
    assert tree.evict_checkpoint()


def test_latest_track_beyond_finished_frontier_uses_only_previous_overlap_track():
    p = pool()
    committed(p, length=4)
    old = p.snapshot(0, 1, staged=True)
    committed(p, length=8, start=4)
    new = p.snapshot(0, 1, staged=True)
    manager = object.__new__(CacheManager)
    manager._tracking = {(0, 1): [old, new]}
    req = SimpleNamespace(table_idx=0, uid=1)
    assert manager.finished_checkpoint(req, 9) == new
    assert manager.finished_checkpoint(req, 7) == old
    assert manager.finished_checkpoint(req, 3) is None
    assert manager.finished_checkpoint(SimpleNamespace(table_idx=0, uid=2), 9) is None


def test_partial_layer_cannot_seal_and_generation_prevents_aba():
    p = pool()
    p.reserve_request(0, 1)
    plan = p.reserve_checkpoint(0, 1, 2, end_len=4, staged=True)
    p.begin_request(0, 1, 0)
    with p.borrow_checkpoint(plan, 0) as (conv, state):
        conv.fill_(2)
        state.fill_(4)
    p.finish_request(0, 1, 4)
    with pytest.raises(RuntimeError, match="state layers"):
        p.seal_checkpoint(plan)
    with pytest.raises(RuntimeError, match="unsealed"):
        p.adopt_staged(plan.checkpoint)
    p.release_request(0, 1)
    p.reserve_request(0, 2)
    fresh = p.reserve_checkpoint(0, 2, 2, end_len=4, staged=True)
    assert fresh.generation > plan.generation
    p.abort_checkpoint(plan)
    p.release_request(0, 1)
    assert fresh.generation in p._pending and p._owners[0] == 2
    with pytest.raises(RuntimeError, match="stale"):
        p.restore(1, 3, plan.checkpoint)


def test_nested_arena_abort_waits_for_last_global_borrow_before_slot_reuse():
    arena = UnifiedMemory(8, 256, 256, "cpu")
    p = pool(total=8, tracking=0, arena=arena)
    p.reserve_request(0, 1)
    outer = p.reserve_checkpoint(0, 1, 2, end_len=4)
    inner = p.reserve_checkpoint(0, 1, 3, end_len=4)
    inner_slot = p._checkpoint_storage[inner.generation]
    p.begin_request(0, 1, 0)
    with p.borrow_checkpoint(outer, 0):
        with pytest.raises(RuntimeError, match="stale"):
            with p.borrow_checkpoint(inner, 0):
                p.abort_checkpoint(inner)
                assert inner_slot not in p._free_state_slots
        assert arena.has_state(p._state_key(inner_slot))
        assert inner_slot not in p._free_state_slots
        assert p.state_usage["deferred_release"] == 1
        with pytest.raises(RuntimeError, match="borrowed"):
            p.release_request(0, 1)
        assert p._owners[0] == 1
    assert not arena.has_state(p._state_key(inner_slot))
    assert p.state_usage["deferred_release"] == 0
    assert inner_slot in p._free_state_slots
    arena.check_integrity()


def test_unknown_copy_fault_calls_owner_abort_before_gpu_cleanup_or_recycle(monkeypatch):
    class FatalOwnerExit(BaseException):
        pass

    errors = []

    def abort(error):
        errors.append(error)
        raise FatalOwnerExit

    p = pool(abort=abort)
    committed(p)

    def failed_copy(*args, **kwargs):
        raise RuntimeError("unknown device fault")

    monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
    monkeypatch.setattr(p, "_release_slot", lambda *a: pytest.fail("recycled after unknown fault"))
    with pytest.raises(FatalOwnerExit):
        p.snapshot(0, 1, staged=True)
    assert len(errors) == 1 and "unknown device fault" in str(errors[0])
    assert p._owners[0] == 1 and p.state_usage["private_checkpoint"] == 0
    assert len(p._request_checkpoints) == 1  # owned worker exits with ledger intact


def test_checkpoint_strategy_capability_does_not_depend_on_current_free_slots(monkeypatch):
    from minisgl.attention.kda.backend import KDAAttentionBackend

    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=2))
    p = pool(total=3)
    p.reserve_request(0, 1)
    assert p.available_checkpoint_slots == 0 and p.checkpoint_enabled
    backend = object.__new__(KDAAttentionBackend)
    backend.kvcache = SimpleNamespace(recurrent=p, arena=None, supports_overlap=False)
    backend.checkpoint_layer_configs = {0: object()}
    backend.config = SimpleNamespace(cache_config=object())
    backend.prefill_kernel = SimpleNamespace(
        prepare_checkpoints=object(),
        checkpoint_quantum=lambda *args: 64,
    )
    assert backend.checkpoint_quantum() == 64
