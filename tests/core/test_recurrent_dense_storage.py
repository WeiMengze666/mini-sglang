import pytest
import torch
from minisgl.kvcache.recurrent_pool import RecurrentStatePool


def pool(*, cached=2, staged=2):
    return RecurrentStatePool(
        (2, 5),
        1,
        1,
        4,
        4,
        torch.bfloat16,
        "cpu",
        max_cached_states=cached,
        max_staged_states=staged,
        state_dtype=torch.bfloat16,
    )


def write(p, uid, start, end, frontier, value):
    plan = p.reserve_checkpoint(0, uid, frontier, end_len=end, staged=True)
    assert plan is not None
    p.begin_request(0, uid, start)
    for layer in (2, 5):
        with p.borrow_checkpoint(plan, layer) as (conv, state):
            conv.fill_(value)
            state.fill_(value + 1)
    p.finish_request(0, uid, end)
    p.seal_checkpoint(plan)
    return plan


def test_two_inflight_tracking_leases_donate_without_copy_or_working_alias():
    p = pool()
    p.reserve_request(0, 7)
    first = write(p, 7, 0, 5, 4, 3)
    second = write(p, 7, 5, 9, 8, 6)
    assert p.available_tracking_slots == 0 and p.available_checkpoint_slots == 2
    first_pointer = p._checkpoints[first.generation][0].data_ptr()
    p.adopt_staged(first.checkpoint)
    assert p._checkpoints[first.generation][0].data_ptr() == first_pointer
    p.adopt_staged(second.checkpoint)
    assert p.available_tracking_slots == 2 and p.available_checkpoint_slots == 0
    p.abort_checkpoint(first)
    p.release_request(0, 7)
    p.restore(0, 8, first.checkpoint)
    assert p._processed_lens[0] == 4
    conv, state = p._working(0)
    assert torch.all(conv == 3) and torch.all(state == 4)
    p.drop_checkpoint(first.checkpoint)
    p.drop_checkpoint(second.checkpoint)
    assert p.available_checkpoint_slots == 2


def test_dense_snapshot_and_lease_reuse_do_not_allocate(monkeypatch):
    p = pool(cached=1, staged=1)
    p.reserve_request(0, 7)
    p.begin_request(0, 7, 0)
    p.finish_request(0, 7, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime tensor allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    old = p.snapshot(0, 7, staged=True, expected_len=2)
    pointer = p._checkpoints[old.slot][0].data_ptr()
    p.release_staged(old)
    fresh = p.reserve_checkpoint(0, 7, 3, end_len=4, staged=True)
    assert fresh.generation != old.slot
    assert p._checkpoints[fresh.generation][0].data_ptr() == pointer
    p.drop_checkpoint(old)
    assert fresh.generation in p._checkpoints
    p.abort_checkpoint(fresh)
    assert p.available_tracking_slots == 1


def test_borrowed_dense_row_not_reused_until_last_borrow_exits():
    p = pool(cached=1, staged=0)
    p.reserve_request(0, 7)
    plan = p.reserve_checkpoint(0, 7, 1, end_len=2)
    p.begin_request(0, 7, 0)
    with pytest.raises(RuntimeError, match="stale"):
        with p.borrow_checkpoint(plan, 2):
            p.abort_checkpoint(plan)
            assert not p._free_checkpoint_slots
    assert len(p._free_checkpoint_slots) == 1
    p.finish_request(0, 7, 2)
    fresh = p.reserve_checkpoint(0, 7, 3, end_len=4)
    assert fresh is not None and fresh.generation != plan.generation


def test_failed_donation_keeps_pending_lease_and_tracking_quota():
    p = pool(cached=0, staged=2)
    p.reserve_request(0, 7)
    plan = write(p, 7, 0, 3, 2, 1)
    with pytest.raises(MemoryError, match="tree"):
        p.adopt_staged(plan.checkpoint)
    assert plan.generation in p._pending and plan.generation in p._staged
    p.abort_checkpoint(plan)
    assert p.available_tracking_slots == 2


def test_allocated_bytes_counts_all_dense_payload_and_native_strides():
    p = pool(cached=3, staged=2)
    per_slot = sum(x.numel() * x.element_size() for x in p._working(0))
    assert p.allocated_bytes == (1 + 3 + 2) * per_slot
    conv, state = p.indexed_kda_state(2)
    assert conv.shape == (1, 12, 3) and conv.stride(1) == 1
    assert state.shape == (1, 1, 4, 4) and state.stride(-1) == 1
    assert p._checkpoint_conv[0].stride(-2) == 1
    assert p._checkpoint_recurrent[0].transpose(-1, -2).stride(-1) == 1


@pytest.mark.parametrize("overlap,tracking", [(False, 1), (True, 2)])
def test_hybrid_extra_buffer_capacity_and_full_payload_accounting(overlap, tracking):
    from minisgl.kvcache.config import HybridCacheConfig
    from minisgl.kvcache.hybrid_pool import HybridCache

    config = HybridCacheConfig(2, (1,), (0,), 8, 4, 1, 4, 4)
    cache = HybridCache(
        config,
        2,
        8,
        1,
        torch.bfloat16,
        torch.device("cpu"),
        max_cached_states=2,
        enable_cache_overlap=overlap,
    )
    assert cache.recurrent.max_staged_states == tracking
    assert cache.supports_checkpoint_overlap is False  # No fatal worker owner.
    assert cache.allocated_bytes == (
        cache.mla._mla.numel() * cache.mla._mla.element_size() + cache.recurrent.allocated_bytes
    )


def test_optional_end_snapshot_skips_when_two_tracking_generations_are_live():
    from types import SimpleNamespace

    from minisgl.scheduler.cache import CacheManager

    p = pool(cached=2, staged=2)
    p.reserve_request(0, 7)
    old = write(p, 7, 0, 4, 2, 3)  # Replay fork F2, aligned end E4.
    new = p.reserve_checkpoint(0, 7, 6, end_len=8, staged=True)
    assert new is not None and p.available_tracking_slots == 0
    manager = object.__new__(CacheManager)
    manager.pool = SimpleNamespace(
        stage_state=lambda table, uid, length: p.snapshot(
            table, uid, staged=True, expected_len=length
        )
    )
    manager.page_size = 2
    manager.prefix_cache = SimpleNamespace(checkpoint_interval=2)
    req = SimpleNamespace(uid=7, table_idx=0)
    # This is the actual optional staging caller, before enqueueing new.
    assert manager.stage_states([req], [4], is_prefill=True) == {}
    p.begin_request(0, 7, 4)
    for layer in (2, 5):
        with p.borrow_checkpoint(new, layer) as (conv, state):
            conv.fill_(6)
            state.fill_(7)
    p.finish_request(0, 7, 8)
    p.seal_checkpoint(new)
    p.adopt_staged(old.checkpoint)
    assert p._checkpoints[old.generation][0].eq(3).all()
    assert p._checkpoints[new.generation][0].eq(6).all()
    p.abort_checkpoint(old)  # No-op after donation, even with new generation live.
    assert old.generation in p._checkpoints and new.generation in p._pending
