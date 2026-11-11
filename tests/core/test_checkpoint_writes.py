"""CPU protocol gates for optional checkpoint writes; no numerical kernel adapter."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.kvcache.unified import UnifiedMemory
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.scheduler import ForwardData

LAYERS = (0, 2)
SUPPORTED = SimpleNamespace(checkpoint_quantum=lambda batch: 4)


def make_manager(monkeypatch, unified):
    arena = UnifiedMemory(64, 256, 256, "cpu") if unified else None
    states = RecurrentStatePool(
        LAYERS, 3, 1, 2, 3, torch.float32, "cpu", max_cached_states=3, arena=arena
    )
    pool = SimpleNamespace(
        recurrent=states,
        arena=arena,
        release_staged=states.release_staged,
        release_request=states.release_request,
    )
    ctx = core.Context(page_size=2)
    ctx.kv_cache = pool
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return CacheManager(32, 2, torch.full((3, 32), -1, dtype=torch.int32), "hybrid", pool)


@pytest.fixture(params=[False, True])
def manager(monkeypatch, request):
    return make_manager(monkeypatch, request.param)


@pytest.fixture
def arena_manager(monkeypatch):
    return make_manager(monkeypatch, True)


def admit(manager, slot=0, uid=10, end=11):
    ids = torch.arange(end, dtype=torch.int32)
    handle = manager.prefix_cache.match_prefix(ids).cuda_handle
    manager.lock(handle)
    manager.restore_state(slot, uid, handle)
    if manager.pool.recurrent._owners[slot] is None:
        manager.pool.recurrent.reserve_request(slot, uid)
    return Req(ids, slot, handle.cached_len, 3, uid, SamplingParams(), handle)


def allocate_pages(manager, req):
    # Exercise real page ownership without the CUDA-only pinned write launcher.
    count = (req.device_len + manager.page_size - 1) // manager.page_size
    locations = manager._page_to_token(manager._allocate(count))
    manager.page_table[req.table_idx, : len(locations)].copy_(locations)


def write_all(states, plan, *, value=None):
    for layer in LAYERS:
        with states.borrow_checkpoint(plan, layer) as (conv, recurrent):
            conv.fill_(plan.checkpoint.length if value is None else value)
            recurrent.fill_(2 * plan.checkpoint.length if value is None else value * 2)


def finish(states, req):
    states.finish_request(req.table_idx, req.uid, req.device_len)
    req.cached_len = req.device_len


def test_non_aligned_tail_is_paired_sealed_and_reused_across_uid(manager):
    req = admit(manager)
    allocate_pages(manager, req)
    states = manager.pool.recurrent
    batch = Batch([req], "prefill")
    with manager.checkpoint_writes(batch, SUPPORTED) as plans:
        (plan,) = plans
        assert plan.checkpoint.length == 8 and req.cached_len == 0 and req.device_len == 11
        states.begin_request(0, 10, 0)
        write_all(states, plan)
        finish(states, req)
    assert req.cached_len == 11  # checkpoint frontier never rewinds executed state
    checkpoint = states.seal_checkpoint(plan)
    saved = states._checkpoints[checkpoint.slot]
    manager.cache_req(req, finished=False, frontier=8, checkpoint=checkpoint)
    assert req.cached_len == 11 and req.device_len == 11
    manager.release_checkpoint_plan(plans)  # donation transferred ownership; no free
    manager.release_state(0, 10)
    assert states._checkpoints[checkpoint.slot] is saved
    assert checkpoint.slot not in states._pending
    matched = manager.prefix_cache.match_prefix(req.input_ids).cuda_handle
    assert matched.cached_len == 8
    manager.lock(matched)
    manager.restore_state(1, 11, matched)
    states.begin_request(1, 11, 8)
    restored = states.get_kda_state(2, 1)
    assert restored.conv.eq(8).all() and restored.recurrent.eq(16).all()
    states.finish_request(1, 11, 9)
    manager.release_state(1, 11)
    manager.unlock(matched)
    manager.unlock(manager.current_handle(req))


def test_known_fork_uses_lcm_grid_and_chunk_continuation(manager):
    req = admit(manager)
    states = manager.pool.recurrent
    manager._handles[0, 10] = SimpleNamespace(replay_until=4)
    batch = Batch([req], "prefill")
    (plan,) = manager.plan_checkpoints(batch, SUPPORTED)
    assert plan.checkpoint.length == 4
    states.begin_request(0, 10, 0)
    write_all(states, plan)
    finish(states, req)
    states.seal_checkpoint(plan)
    manager.release_checkpoint_plan(batch.checkpoint_plan)
    # Next chunk continues at11, not at the previous cached frontier4.
    req.input_ids = torch.arange(19, dtype=torch.int32)
    req.device_len = 19
    manager._handles[0, 10] = SimpleNamespace(replay_until=0)
    assert manager.plan_checkpoints(Batch([req], "prefill"), SUPPORTED) == ()
    assert req.cached_len == 11  # relative kernel chunk is not representable
    states.release_request(0, 10)


def test_aligned_continuation_and_dcp_page_geometry(manager):
    req = admit(manager, end=8)
    states = manager.pool.recurrent
    states.begin_request(0, 10, 0)
    states.finish_request(0, 10, 8)
    req.cached_len = 8
    req.input_ids = torch.arange(19, dtype=torch.int32)
    req.device_len = 19
    (plan,) = manager.plan_checkpoints(Batch([req], "prefill"), SUPPORTED)
    assert plan.checkpoint.length == 16 and plan.start_len == 8 and plan.end_len == 19
    manager.release_checkpoint_plan((plan,))
    states.release_request(0, 10)
    req = admit(manager, uid=20, end=1050)
    manager.page_size = 512
    backend = SimpleNamespace(checkpoint_quantum=lambda batch: 64)
    (plan,) = manager.plan_checkpoints(Batch([req], "prefill"), backend)
    assert plan.checkpoint.length == 1024
    manager.release_checkpoint_plan((plan,))
    states.release_request(0, 20)


def test_missing_layer_cannot_publish_or_restore(manager):
    req = admit(manager)
    states = manager.pool.recurrent
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    with states.borrow_checkpoint(plan, 0) as (conv, recurrent):
        conv.zero_()
        recurrent.zero_()
    finish(states, req)
    with pytest.raises(RuntimeError, match="state layers"):
        states.seal_checkpoint(plan)
    with pytest.raises(RuntimeError, match="unsealed"):
        states.adopt_staged(plan.checkpoint)
    with pytest.raises(RuntimeError, match="unsealed"):
        states.restore(1, 11, plan.checkpoint)
    assert not manager.prefix_cache.checkpoints
    states.abort_checkpoint(plan)
    assert not states._pending and not states._checkpoints


def test_late_model_failure_and_partial_batch_seal_abort_every_plan(manager):
    reqs = [admit(manager, slot=slot, uid=10 + slot) for slot in range(2)]
    states = manager.pool.recurrent
    batch = Batch(reqs, "prefill")
    with pytest.raises(RuntimeError, match="state layers"):
        with manager.checkpoint_writes(batch, SUPPORTED) as plans:
            for req, plan in zip(reqs, plans):
                states.begin_request(req.table_idx, req.uid, 0)
                if req.table_idx == 0:
                    write_all(states, plan)
                finish(states, req)
            # The generic scope no longer seals; exercise the explicit all-layer guard here.
            for plan in plans:
                states.seal_checkpoint(plan)
    assert not states._pending and not states._checkpoints
    # A failure after all per-layer writes also aborts, not publishes.
    batch = Batch([admit(manager, slot=2, uid=30)], "prefill")
    with pytest.raises(ValueError, match="late model failure"):
        with manager.checkpoint_writes(batch, SUPPORTED) as plans:
            states.begin_request(2, 30, 0)
            write_all(states, plans[0])
            raise ValueError("late model failure")
    assert not states._pending and not states._checkpoints
    states.release_request(2, 30)


def test_layer_exception_aborts_and_uid_reuse_cannot_borrow_or_abort_new_lease(manager):
    admit(manager)
    states = manager.pool.recurrent
    old = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    with pytest.raises(ValueError, match="kernel stub"):
        with states.borrow_checkpoint(old, 0) as (conv, recurrent):
            conv.fill_(99)  # fail between the paired conv and SSM writes
            raise ValueError("kernel stub")
    assert not states._pending
    states.release_request(0, 10)
    states.reserve_request(0, 20)
    new = states.reserve_checkpoint(0, 20, 8, end_len=11)
    assert new.generation > old.generation
    states.begin_request(0, 20, 0)
    with pytest.raises(RuntimeError, match="stale"):
        with states.borrow_checkpoint(old, 0):
            pytest.fail("stale lease borrowed")
    states.abort_checkpoint(old)
    states.release_request(0, 10)  # late release never frees the new owner
    assert new.generation in states._pending
    with pytest.raises(RuntimeError, match="stale"):
        with states.borrow_checkpoint(replace(new, uid=10), 0):
            pytest.fail("forged owner borrowed")
    write_all(states, new)
    states.finish_request(0, 20, 11)
    states.seal_checkpoint(new)
    states.release_request(0, 20)  # cancellation releases even a sealed undonated plan
    assert not states._pending and not states._checkpoints


def test_same_uid_reuse_has_new_generation_and_old_frontier_cannot_seal(manager):
    states = manager.pool.recurrent
    req = admit(manager)
    old = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.release_request(0, 10)
    states.reserve_request(0, 10)
    new = states.reserve_checkpoint(0, 10, 8, end_len=11)
    assert new.generation != old.generation
    with pytest.raises(RuntimeError, match="stale"):
        states.seal_checkpoint(old)
    states.begin_request(0, 10, 0)
    write_all(states, new)
    finish(states, req)
    states.begin_request(0, 10, 11)
    with pytest.raises(RuntimeError, match="completed forward"):
        states.seal_checkpoint(new)
    states.release_request(0, 10)


def test_capacity_is_cached_budget_not_additional_staging(manager):
    admit(manager)
    states = manager.pool.recurrent
    assert states.max_staged_states == 0
    states.max_cached_states = 1
    first = states.reserve_checkpoint(0, 10, 4, end_len=11)
    assert first is not None
    assert states.reserve_checkpoint(0, 10, 8, end_len=11) is None
    assert len(states._checkpoints) == 1
    states.abort_checkpoint(first)
    states.max_cached_states = 0
    assert states.reserve_checkpoint(0, 10, 8, end_len=11) is None
    assert not states._checkpoints


def test_pressure_never_evicts_locked_tree_checkpoint(manager):
    req = admit(manager, end=4)
    allocate_pages(manager, req)
    states = manager.pool.recurrent
    states.max_cached_states = 1
    states.begin_request(0, 10, 0)
    finish(states, req)
    manager.cache_req(req, finished=False)
    tree_slot = next(iter(states._checkpoints))
    protected = manager.prefix_cache.match_prefix(req.input_ids).cuda_handle
    manager.lock(protected)
    states.reserve_request(1, 20)
    assert states.reserve_checkpoint(1, 20, 4, end_len=7) is None
    manager.prefix_cache.release_checkpoint_memory(1 << 30)
    assert tree_slot in states._checkpoints
    manager.unlock(protected)
    manager.unlock(manager.current_handle(req))
    manager.prefix_cache.evict(4)
    assert tree_slot not in states._checkpoints
    assert states.reserve_checkpoint(1, 20, 4, end_len=7) is not None
    states.release_request(1, 20)


def test_plan_is_immutable_and_current_backends_have_no_default_plan(manager):
    req = admit(manager)
    batch = Batch([req], "prefill")
    assert manager.plan_checkpoints(batch, SimpleNamespace(name="sgl")) == ()
    assert manager.plan_checkpoints(Batch([req], "decode"), SUPPORTED) == ()
    (plan,) = manager.plan_checkpoints(batch, SUPPORTED)
    data = ForwardData(None, None, (11,), batch.checkpoint_plan)
    with pytest.raises(FrozenInstanceError):
        plan.uid = 999
    req.cached_len = 11
    batch.checkpoint_plan = ()
    assert data.checkpoint_plan == (plan,) and plan.start_len == 0 and plan.end_len == 11
    manager.release_checkpoint_plan(data.checkpoint_plan)
    assert not manager.pool.recurrent._pending


def test_nondivisible_page_and_quantum_use_lcm_and_skip_existing_endpoint(manager):
    req = admit(manager, end=20)
    manager.page_size = 6
    states = manager.pool.recurrent
    (plan,) = manager.plan_checkpoints(Batch([req], "prefill"), SUPPORTED)
    assert plan.checkpoint.length == 12  # lcm(6, 4), not either grid alone
    manager.release_checkpoint_plan((plan,))
    req.input_ids = torch.arange(24, dtype=torch.int32)
    req.device_len = 24
    assert manager.plan_checkpoints(Batch([req], "prefill"), SUPPORTED) == ()
    assert not states._pending  # aligned endpoint uses the existing snapshot path


def test_known_fork_requires_global_lcm_and_relative_kernel_alignment(manager):
    req = admit(manager, end=20)
    states = manager.pool.recurrent
    states.begin_request(0, 10, 0)
    states.finish_request(0, 10, 2)
    req.cached_len = 2
    manager.page_size = 6
    manager._handles[0, 10] = SimpleNamespace(replay_until=6)
    # Fork6 is page aligned and6-2 is kernel aligned, but6 is not on lcm(6,4).
    assert manager.plan_checkpoints(Batch([req], "prefill"), SUPPORTED) == ()
    assert not states._pending
    states.release_request(0, 10)


def test_arena_pressure_skips_without_leaking_or_releasing_working_state():
    arena = UnifiedMemory(1, 256, 256, "cpu")
    states = RecurrentStatePool(
        LAYERS, 1, 1, 2, 3, torch.float32, "cpu", max_cached_states=3, arena=arena
    )
    states.reserve_request(0, 10)
    original = arena.state_view(("request", 0)).clone()
    assert arena.free_bytes == 0
    assert states.reserve_checkpoint(0, 10, 4, end_len=7) is None
    assert not states._pending and not states._checkpoints
    assert states._owners == [10] and arena.has_state(("request", 0))
    assert torch.equal(arena.state_view(("request", 0)), original)
    arena.check_integrity()
    states.release_request(0, 10)
    assert arena.free_bytes == 256


def state_pool(arena, *, capacity=2):
    return RecurrentStatePool(
        LAYERS,
        capacity,
        1,
        2,
        3,
        torch.float32,
        "cpu",
        max_cached_states=4,
        arena=arena,
    )


def test_borrowed_checkpoint_cannot_silently_relocate_and_publish_wrong_state():
    arena = UnifiedMemory(16, 256, 256, "cpu")
    arena.buffer.zero_()
    states = state_pool(arena)
    states.reserve_request(0, 10)
    states.begin_request(0, 10, 0)
    states.finish_request(0, 10, 1)
    old = states.snapshot(0, 10)
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 1)
    original_row = arena.state_index(("checkpoint", plan.generation))
    refused = False
    with states.borrow_checkpoint(plan, 0) as (conv, recurrent):
        try:
            states.drop_checkpoint(old)
        except RuntimeError as error:
            assert "borrow" in str(error)
            refused = True
        conv.fill_(8)
        recurrent.fill_(16)
    if refused:
        assert old.slot in states._checkpoints  # rejection precedes pool bookkeeping
        assert arena.state_index(("checkpoint", plan.generation)) == original_row
        states.drop_checkpoint(old)  # compaction becomes safe after the borrow
    with states.borrow_checkpoint(plan, 2) as (conv, recurrent):
        conv.fill_(8)
        recurrent.fill_(16)
    states.finish_request(0, 10, 11)
    checkpoint = states.seal_checkpoint(plan)
    states.adopt_staged(checkpoint)
    states.restore(1, 20, checkpoint)
    states.begin_request(1, 20, 8)
    restored = states.get_kda_state(0, 1)
    # Frozen201 permits row2->row1 and silently restores0/0 here.
    assert restored.conv.eq(8).all() and restored.recurrent.eq(16).all()
    assert refused
    arena.check_integrity()


@pytest.mark.parametrize(
    "operation", ["allocate_state", "free_state", "allocate_pages", "free_pages"]
)
def test_borrow_guard_covers_direct_arena_mutation_before_any_bookkeeping(operation):
    arena = UnifiedMemory(8, 256, 256, "cpu")
    arena.buffer.zero_()
    arena.allocate_state("pinned")
    arena.allocate_state("other")
    arena.allocate_pages([0, 1])
    before = (
        dict(arena._states),
        list(arena._state_ids),
        dict(arena._pages),
        list(arena._page_ids),
        arena.page_map.clone(),
        arena.buffer.clone(),
    )
    mutations = {
        "allocate_state": lambda: arena.allocate_state("new"),
        "free_state": lambda: arena.free_state("other"),
        "allocate_pages": lambda: arena.allocate_pages([2]),
        "free_pages": lambda: arena.free_pages_by_id([0, 1]),
    }
    with arena.borrow_state("pinned"):
        with pytest.raises(RuntimeError, match="borrow"):
            mutations[operation]()
        assert arena._states == before[0] and arena._state_ids == before[1]
        assert arena._pages == before[2] and arena._page_ids == before[3]
        assert torch.equal(arena.page_map, before[4])
        assert torch.equal(arena.buffer, before[5])
    mutations[operation]()  # guard restored after successful exit
    arena.check_integrity()


def test_shared_arena_guard_constrains_other_pool_release_and_admission():
    arena = UnifiedMemory(12, 256, 256, "cpu")
    states, other = state_pool(arena), state_pool(arena)
    states.reserve_request(0, 10)
    other.reserve_request(1, 20)
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    with states.borrow_checkpoint(plan, 0):
        with pytest.raises(RuntimeError, match="borrow"):
            other.release_request(1, 20)
        assert other._owners[1] == 20 and arena.has_state(("request", 1))
        with pytest.raises(RuntimeError, match="borrow"):
            arena.free_state(("request", 1))
    other.release_request(1, 20)
    with states.borrow_checkpoint(plan, 2):
        with pytest.raises(RuntimeError, match="borrow"):
            other.reserve_request(1, 30)
        assert other._owners[1] is None and not arena.has_state(("request", 1))
    other.reserve_request(1, 30)
    other.release_request(1, 30)
    states.abort_checkpoint(plan)
    states.release_request(0, 10)
    assert arena.free_bytes == arena.buffer.numel()


def test_nested_layer_borrows_hold_guard_until_outermost_exit():
    arena = UnifiedMemory(8, 256, 256, "cpu")
    states = state_pool(arena)
    states.reserve_request(0, 10)
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    with states.borrow_checkpoint(plan, 0) as (conv0, recurrent0):
        with states.borrow_checkpoint(plan, 2) as (conv2, recurrent2):
            conv2.fill_(8)
            recurrent2.fill_(16)
            with pytest.raises(RuntimeError, match="borrow"):
                arena.allocate_pages([0])
        with pytest.raises(RuntimeError, match="borrow"):
            arena.allocate_state("still-borrowed")
        conv0.fill_(8)
        recurrent0.fill_(16)
    arena.allocate_pages([0])
    arena.free_pages_by_id([0])
    states.finish_request(0, 10, 11)
    states.seal_checkpoint(plan)
    states.abort_checkpoint(plan)
    states.release_request(0, 10)
    assert arena.free_bytes == arena.buffer.numel()


def test_nested_different_plan_failure_defers_only_aborted_state_free():
    arena = UnifiedMemory(12, 256, 256, "cpu")
    states = state_pool(arena)
    for slot, uid in [(0, 10), (1, 20)]:
        states.reserve_request(slot, uid)
    failed = states.reserve_checkpoint(1, 20, 8, end_len=11)
    first = states.reserve_checkpoint(0, 10, 8, end_len=11)
    for slot, uid in [(0, 10), (1, 20)]:
        states.begin_request(slot, uid, 0)
    failed_key = ("checkpoint", failed.generation)
    before = arena.free_bytes
    with states.borrow_checkpoint(first, 0) as (conv, recurrent):
        with pytest.raises(ValueError, match="inner kernel"):
            with states.borrow_checkpoint(failed, 0) as (failed_conv, _):
                failed_conv.fill_(99)
                raise ValueError("inner kernel")
        assert failed.generation not in states._pending
        assert arena.has_state(failed_key) and arena.free_bytes == before
        with pytest.raises(RuntimeError, match="stale"):
            states.seal_checkpoint(failed)
        with pytest.raises(RuntimeError, match="released"):
            states.adopt_staged(failed.checkpoint)
        with pytest.raises(RuntimeError, match="borrow"):
            arena.free_state(("request", 1))
        conv.fill_(8)
        recurrent.fill_(16)
    assert not arena.has_state(failed_key)
    assert arena.free_bytes == before + arena.state_bytes
    assert not arena._deferred_state_frees
    with states.borrow_checkpoint(first, 2) as (conv, recurrent):
        conv.fill_(8)
        recurrent.fill_(16)
    states.release_request(1, 20)
    states.finish_request(0, 10, 11)
    states.seal_checkpoint(first)
    states.adopt_staged(first.checkpoint)
    states.restore(1, 30, first.checkpoint)
    states.begin_request(1, 30, 8)
    restored = states.get_kda_state(0, 1)
    assert restored.conv.eq(8).all() and restored.recurrent.eq(16).all()
    states.release_request(0, 10)
    states.release_request(1, 30)
    states.drop_checkpoint(first.checkpoint)
    assert arena.free_bytes == arena.buffer.numel()
    arena.check_integrity()


def test_nested_same_plan_exception_unwinds_guard_and_recovers_cached_budget():
    arena = UnifiedMemory(8, 256, 256, "cpu")
    states = state_pool(arena)
    states.reserve_request(0, 10)
    working_only = arena.free_bytes
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    with pytest.raises(ValueError, match="nested failure"):
        with states.borrow_checkpoint(plan, 0):
            with states.borrow_checkpoint(plan, 2) as (conv, _):
                conv.fill_(99)
                raise ValueError("nested failure")
    assert not states._pending and not states._checkpoints
    assert not arena._deferred_state_frees and arena.free_bytes == working_only
    # A failed guard entry must also balance its lifetime counter.
    with pytest.raises(KeyError):
        with arena.borrow_state("missing"):
            pytest.fail("missing state borrowed")
    arena.allocate_pages([0])
    arena.free_pages_by_id([0])
    states.release_request(0, 10)
    assert arena.free_bytes == arena.buffer.numel()


def test_hybrid_eviction_rejection_preserves_tree_and_pool_bookkeeping(arena_manager):
    manager = arena_manager
    req = admit(manager, end=4)
    allocate_pages(manager, req)
    states = manager.pool.recurrent
    states.begin_request(0, 10, 0)
    finish(states, req)
    manager.cache_req(req, finished=False)
    manager.unlock(manager.current_handle(req))
    old = dict(manager.prefix_cache.checkpoints)
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 4)
    with states.borrow_checkpoint(plan, 0):
        with pytest.raises(RuntimeError, match="borrow"):
            manager.prefix_cache.release_checkpoint_memory(manager.arena.buffer.numel())
        assert manager.prefix_cache.checkpoints == old
        assert all(cp.slot in states._checkpoints for _, cp in old.values())
    manager.prefix_cache.release_checkpoint_memory(manager.arena.buffer.numel())
    assert not manager.prefix_cache.checkpoints
    states.release_request(0, 10)


@pytest.mark.parametrize("occupied", [False, True])
def test_optional_checkpoint_never_admits_or_consumes_working_budget(occupied):
    arena = UnifiedMemory(1, 256, 256, "cpu")
    if occupied:
        arena.allocate_state("external")
    states = state_pool(arena, capacity=1)
    free = arena.free_bytes
    assert states.reserve_checkpoint(0, 10, 4, end_len=7) is None
    assert states._owners == [None] and not states._checkpoints and not states._pending
    assert not arena.has_state(("request", 0)) and arena.free_bytes == free
    arena.check_integrity()


@pytest.mark.parametrize("operation", ["allocate", "free", "release", "publish", "cancel"])
def test_cache_manager_preflight_rejects_before_its_logical_mutations(arena_manager, operation):
    manager = arena_manager
    req = admit(manager)
    allocate_pages(manager, req)
    states = manager.pool.recurrent
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    handles = dict(manager._handles)
    slots = manager.free_slots.clone()
    checkpoints = dict(manager.prefix_cache.checkpoints)
    pending = dict(states._pending)
    mutations = {
        "allocate": lambda: manager._allocate(1),
        "free": lambda: manager._free(manager.page_table[0, :2]),
        "release": lambda: manager.release_state(0, 10),
        "publish": lambda: manager.cache_req(
            req, finished=False, frontier=8, checkpoint=plan.checkpoint
        ),
        "cancel": lambda: states.release_request(0, 10),
    }
    with states.borrow_checkpoint(plan, 0):
        with pytest.raises(RuntimeError, match="borrow"):
            mutations[operation]()
        assert manager._handles == handles
        assert torch.equal(manager.free_slots, slots)
        assert manager.prefix_cache.checkpoints == checkpoints
        assert states._pending == pending
        assert states._owners[0] == 10
    states.release_request(0, 10)
    assert not states._pending


def test_explicit_abort_inside_borrow_cannot_publish_and_reclaims_after_exit():
    arena = UnifiedMemory(8, 256, 256, "cpu")
    states = state_pool(arena)
    states.reserve_request(0, 10)
    free = arena.free_bytes
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    with pytest.raises(RuntimeError, match="stale"):
        with states.borrow_checkpoint(plan, 0) as (conv, _):
            states.abort_checkpoint(plan)
            assert arena.has_state(("checkpoint", plan.generation))
            assert plan.generation not in states._pending
            with pytest.raises(RuntimeError, match="released"):
                states.adopt_staged(plan.checkpoint)
            conv.fill_(99)  # still pinned storage, never reused until exit
    assert not arena.has_state(("checkpoint", plan.generation))
    assert not arena._deferred_state_frees and arena.free_bytes == free
    states.release_request(0, 10)
    arena.check_integrity()
