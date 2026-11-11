"""CPU ownership checks for native-style private decode tracking."""

from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.scheduler import ForwardData, Scheduler


@pytest.fixture(params=[False, True], ids=["serial", "overlap"])
def manager(monkeypatch, request):
    overlap = request.param
    monkeypatch.setattr("minisgl.scheduler.scheduler.logger.debug_rank0", lambda *args: None)
    states = RecurrentStatePool(
        (0,), 2, 1, 2, 3, torch.float32, "cpu",
        max_cached_states=8, max_staged_states=4 if overlap else 2,
    )
    pool = SimpleNamespace(
        recurrent=states, arena=None, supports_overlap=overlap,
        stage_state=lambda table, uid, length: states.snapshot(
            table, uid, staged=True, expected_len=length,
        ),
        release_staged=states.release_staged, release_request=states.release_request,
    )
    ctx = core.Context(page_size=2)
    ctx.kv_cache = pool
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    manager = CacheManager(64, 2, torch.full((2, 128), -1, dtype=torch.int32), "hybrid", pool)
    yield manager
    assert not torch.cuda.is_initialized()


def ensure_pages(manager, req, length):
    old = getattr(req, "_allocated_pages", 0)
    new = (length + manager.page_size - 1) // manager.page_size
    if new > old:
        values = manager._page_to_token(manager._allocate(new - old))
        manager.page_table[req.table_idx, old * manager.page_size:new * manager.page_size] = values
    req._allocated_pages = new


def commit(manager, req, length):
    states = manager.pool.recurrent
    old = states._processed_lens[req.table_idx]
    states.begin_request(req.table_idx, req.uid, old)
    states.get_kda_state(0, req.table_idx).recurrent.fill_(length)
    states.finish_request(req.table_idx, req.uid, length)
    ensure_pages(manager, req, length)
    req.cached_len, req.device_len = length, length + 1
    req.input_ids = torch.arange(length + 1, dtype=torch.int32) + req.uid * 100


def admit(manager, slot=0, uid=1, end=8):
    ids = torch.arange(end, dtype=torch.int32) + uid * 100
    handle = manager.prefix_cache.match_prefix(ids).cuda_handle
    manager.lock(handle)
    manager.restore_state(slot, uid, handle)
    req = Req(ids, slot, 0, 30, uid, SamplingParams(), handle)
    commit(manager, req, end)
    manager.cache_req(req, finished=False)
    return req


def track(manager, req, length):
    commit(manager, req, length)
    assert manager.stage_states([req], [length], is_prefill=False) == {}


def scheduler_for(manager, req):
    scheduler = object.__new__(Scheduler)
    scheduler.cache_manager = manager
    scheduler.finished_reqs, scheduler.aborted_reqs = set(), set()
    scheduler.inflight_reqs = {req.uid: req}
    scheduler.stop_token_ids = set()
    scheduler.decode_manager = SimpleNamespace(remove_req=lambda req: None, abort_req=lambda uid: req)
    scheduler.prefill_manager = SimpleNamespace(abort_req=lambda uid: None)
    scheduler.table_manager = SimpleNamespace(free=lambda slot: None)
    scheduler.send_result = lambda result: None
    return scheduler


def test_decode_private_ring_rotates_at_full_scratch_quota(manager):
    reqs = [admit(manager, slot, slot + 1) for slot in range(2)]
    original = set(manager.prefix_cache.checkpoints)
    for length in (10, 12, 14, 16):
        for req in reqs:
            track(manager, req, length)
        expected = [length] if not manager.pool.supports_overlap else [length - 2, length]
        if length == 10:
            expected = [10]
        for req in reqs:
            assert [cp.length for cp in manager._tracking[req.table_idx, req.uid]] == expected
        assert set(manager.prefix_cache.checkpoints) == original
        assert len(manager.pool.recurrent._staged) <= manager.pool.recurrent.max_staged_states
    assert not manager.should_cache(False, 16)
    assert manager.should_cache(True, 16)


def test_finished_selector_latest_previous_and_uid_isolation(manager):
    req = admit(manager)
    track(manager, req, 10)
    track(manager, req, 12)
    assert manager.finished_checkpoint(req, 12).length == 12
    previous = manager.finished_checkpoint(req, 11)
    assert (previous.length if previous else None) == (10 if manager.pool.supports_overlap else None)
    assert manager.finished_checkpoint(req, 9) is None
    assert manager.finished_checkpoint(SimpleNamespace(table_idx=0, uid=999), 20) is None


def test_finish_donates_once_and_retains_prefill_prefix(manager):
    req = admit(manager)
    track(manager, req, 10)
    track(manager, req, 12)
    cp = manager.finished_checkpoint(req, 12)
    req.max_device_len = 13
    scheduler = scheduler_for(manager, req)
    data = ForwardData(
        SimpleNamespace(batch=Batch([req], "decode")),
        (None, torch.zeros(1, dtype=torch.int32), SimpleNamespace(synchronize=lambda: None)),
        (12,), (),
    )
    scheduler._process_last_data(data, staged={})
    assert not manager.has_tracking and not manager.pool.recurrent._staged
    assert manager.pool.recurrent._owners[0] is None
    assert sorted(state.length for _, state in manager.prefix_cache.checkpoints.values()) == [8, 12]
    assert cp.slot in manager.pool.recurrent._checkpoints
    manager.check_integrity()


def test_retract_discards_private_tracks_without_new_tree_state(manager):
    req = admit(manager)
    track(manager, req, 10)
    original = set(manager.prefix_cache.checkpoints)
    manager.discard_req(req)
    assert set(manager.prefix_cache.checkpoints) == original
    assert not manager.has_tracking and not manager.pool.recurrent._staged
    assert manager.pool.recurrent._owners[0] is None
    replacement = admit(manager, uid=9)
    manager.release_state(req.table_idx, req.uid)
    assert manager.pool.recurrent._owners[0] == replacement.uid
    assert manager.finished_checkpoint(replacement, 20) is None


def test_abort_does_not_publish_private_tracks(manager):
    req = admit(manager)
    track(manager, req, 10)
    original = set(manager.prefix_cache.checkpoints)
    scheduler = scheduler_for(manager, req)
    scheduler._process_one_msg(AbortBackendMsg(uid=req.uid))
    assert set(manager.prefix_cache.checkpoints) == original
    assert not manager.has_tracking and not manager.pool.recurrent._staged
    manager.check_integrity()


def test_normal_decode_loop_uses_private_stage_and_completion(manager):
    req = admit(manager)
    commit(manager, req, 10)
    scheduler = scheduler_for(manager, req)
    scheduler.prefill_manager.runnable = False
    scheduler.decode_manager.runnable = True
    scheduler.receive_msg = lambda **kwargs: []
    forward = SimpleNamespace(batch=Batch([req], "decode"))
    scheduler._schedule_next_batch = lambda: forward
    scheduler._forward = lambda value: (
        None, torch.zeros(1, dtype=torch.int32), SimpleNamespace(synchronize=lambda: None),
    )
    scheduler.normal_loop()
    assert [cp.length for cp in manager._tracking[req.table_idx, req.uid]] == [10]
    assert sorted(state.length for _, state in manager.prefix_cache.checkpoints.values()) == [8]


def test_overlap_copy_failure_keeps_private_state_until_fatal(manager):
    req = admit(manager)
    track(manager, req, 10)
    scheduler = scheduler_for(manager, req)
    cp = manager._tracking[req.table_idx, req.uid][-1]
    events = []
    def fatal(error):
        events.append("fatal")
        raise error
    manager.pool.recurrent.fail_device = fatal
    scheduler.engine = SimpleNamespace(kv_cache=manager.pool)
    def broken():
        raise RuntimeError("injected completion failure")
    data = ForwardData(
        SimpleNamespace(batch=Batch([req], "decode")),
        (None, torch.zeros(1, dtype=torch.int32), SimpleNamespace(synchronize=broken)),
        (10,), (),
    )
    with pytest.raises(RuntimeError, match="completion"):
        scheduler._process_last_data(data, staged={})
    assert events == ["fatal"]
    assert cp.slot in manager.pool.recurrent._staged


def test_known_fork_survives_short_chunks_until_it_is_computed(manager):
    from minisgl.kvcache.radix_cache import RadixCacheHandle
    from minisgl.scheduler.prefill import PrefillAdder

    # Shared KV reaches 16; the only restorable state initially reaches 4.
    seed = admit(manager, end=4)
    commit(manager, seed, 16)
    manager.cache_req(seed, finished=True, frontier=16)
    manager.release_state(seed.table_idx, seed.uid)
    ids = torch.arange(40, dtype=torch.int32) + seed.uid * 100
    ids[16:] += 1000
    handle = manager.prefix_cache.match_prefix(ids[:-1]).cuda_handle
    assert (handle.cached_len, handle.replay_until) == (4, 16)
    manager.lock(handle)
    manager.restore_state(0, 2, handle)
    req = Req(ids, 0, 4, 3, 2, SamplingParams(), handle)
    manager.page_table[0, :4] = handle.get_matched_indices()
    req._allocated_pages = 2
    states = manager.pool.recurrent
    states.begin_request(0, 2, 4)
    states.finish_request(0, 2, 8)
    ensure_pages(manager, req, 8)
    req.cached_len = 8
    manager.cache_req(req, finished=False)
    handle = manager.current_handle(req)
    adder = object.__new__(PrefillAdder)
    adder.token_budget = adder.chunk_budget = 32
    adder.compute_page_size = adder.chunk_alignment = 2
    pending = SimpleNamespace(input_len=40)
    assert adder._chunk_size(pending, handle, 8) == 8
    # A bare insertion handle reproduces the old lost-fork jump from 8 to 40.
    bare = RadixCacheHandle(handle.cached_len, handle.node)
    assert adder._chunk_size(pending, bare, 8) == 32
    states.begin_request(0, 2, 8)
    states.finish_request(0, 2, 16)
    ensure_pages(manager, req, 16)
    req.cached_len = 16
    manager.cache_req(req, finished=False)
    assert manager.current_handle(req).replay_until == 0
    assert manager.prefix_cache.match_prefix(ids[:-1]).cuda_handle.cached_len == 16


def test_owned_decode_forward_failure_aborts_before_working_release(manager):
    from minisgl.kvcache.hybrid_pool import HybridCache

    req = admit(manager)
    track(manager, req, 10)
    pool = object.__new__(HybridCache)
    pool.recurrent = manager.pool.recurrent
    events = []
    def fatal(error):
        events.append("fatal")
        raise error
    pool.recurrent.device_abort = fatal
    with pytest.raises(RuntimeError, match="model failure"):
        with pool.forward_batch(Batch([req], "decode")):
            raise RuntimeError("model failure")
    assert events == ["fatal"]
    assert pool.recurrent._owners[req.table_idx] == req.uid
    assert pool.recurrent._active[req.table_idx]
    assert manager.has_tracking


def test_serial_scheduling_failure_does_not_reclaim_live_tracking(manager):
    req = admit(manager)
    track(manager, req, 10)
    scheduler = scheduler_for(manager, req)
    scheduler.prefill_manager.runnable = False
    scheduler.decode_manager.runnable = True
    scheduler.receive_msg = lambda **kwargs: []
    scheduler.engine = SimpleNamespace(kv_cache=manager.pool)
    events = []
    def fatal(error):
        events.append("fatal")
        raise error
    manager.pool.recurrent.fail_device = fatal
    def broken():
        raise RuntimeError("scheduling failure")
    scheduler._schedule_next_batch = broken
    with pytest.raises(RuntimeError, match="scheduling failure"):
        scheduler.normal_loop()
    assert events == ["fatal"]
    assert manager.has_tracking
    assert manager.pool.recurrent._staged


@pytest.mark.parametrize("owned", [False, True], ids=["cpu-cleanup", "owned-fatal"])
def test_lazy_free_error_respects_worker_ownership(owned):
    class FatalSentinel(BaseException):
        pass

    events = []
    def fatal(error):
        events.append("fatal")
        raise FatalSentinel()
    states = SimpleNamespace(device_abort=fatal if owned else None, fail_device=fatal)
    manager = object.__new__(CacheManager)
    manager.pool = SimpleNamespace(recurrent=states)
    manager.page_size = 2
    manager.free_slots = torch.tensor([0, 2], dtype=torch.int32)
    manager.arena = SimpleNamespace(free_pages_by_id=lambda pages: events.append(("free", pages)))
    expected = FatalSentinel if owned else RuntimeError
    with pytest.raises(expected):
        with manager.lazy_free_region():
            manager._free(torch.tensor([4, 5], dtype=torch.int32))
            raise RuntimeError("owned-state operation failed")
    if owned:
        assert events == ["fatal"]
        assert manager.free_slots.tolist() == [0, 2]
    else:
        assert events == [("free", [2])]
        assert manager.free_slots.tolist() == [0, 2, 4]
    assert "_free" not in manager.__dict__
    assert not torch.cuda.is_initialized()
