from types import SimpleNamespace as S

import minisgl.core as core
import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import ForwardData, ForwardInput, Scheduler
from minisgl.scheduler.table import TableManager


@pytest.fixture(autouse=True)
def unpinned_CPU_mapping(monkeypatch):
    # Test the real mapping algorithm on CPU without requesting CUDA pinned memory.
    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    original = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)


def request(prompt, generated, uid, table, *, append_latest=True, output=10000):
    req = Req(
        torch.arange(prompt, dtype=torch.int32),
        table,
        0,
        output,
        uid,
        SamplingParams(ignore_eos=True, max_tokens=output),
        None,
        cached_prompt_tokens=prompt // 2,
    )
    for n in range(generated):
        req.complete_one()
        if append_latest or n < generated - 1:
            req.append_host(torch.tensor([30 + n], dtype=torch.int32))
    return req


def setup(monkeypatch, *, count=2, pages=4, append_latest=True, output=8):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=4))
    table = torch.full((3, 20000), -777, dtype=torch.int32)
    states = RecurrentStatePool((0,), 3, 1, 4, 4, torch.bfloat16, "cpu", max_cached_states=2)
    pool = S(release_request=states.release_request)
    cache = CacheManager(pages, 4, table, "naive", pool=pool)
    tm = TableManager(3, table)
    dm = DecodeManager(4)
    reqs = []
    for i in range(count):
        slot = tm.allocate()
        req = request(
            6 if i == 0 else 4,
            3 if i == 0 else 5,
            10 + i,
            slot,
            append_latest=append_latest,
            output=output,
        )
        handle = cache.match_req(
            S(input_len=len(req.input_ids), input_ids=req.input_ids)
        ).cuda_handle
        cache.lock(handle)
        cache._handles[slot, req.uid] = handle
        req.cache_handle = handle
        states.reserve_request(slot, req.uid)
        states.begin_request(slot, req.uid, 0)
        states.finish_request(slot, req.uid, req.cached_len)
        cache.allocate_paged([S(table_idx=slot, cached_len=0, device_len=req.cached_len)])
        reqs.append(req)
    dm.filter_reqs(reqs)
    scheduler = object.__new__(Scheduler)
    scheduler.cache_manager, scheduler.table_manager = (cache, tm)
    scheduler.decode_manager = dm
    scheduler.prefill_manager = PrefillManager(cache, tm, dm)
    scheduler.prefill_budget = 16
    scheduler.inflight_reqs = {r.uid: r for r in reqs}
    scheduler.finished_reqs, scheduler.aborted_reqs = (set(), set())
    scheduler.stop_token_ids = set()
    scheduler.replies = []
    scheduler.send_result = lambda rows: scheduler.replies.extend(rows)
    return (scheduler, reqs, states)


def test_next_decode_page_accounting_and_native_length_order():
    dm = DecodeManager(4)
    a = request(6, 3, 10, 0)
    b = request(4, 5, 11, 1)
    dm.filter_reqs([a, b])
    assert dm.next_page_tokens == 8
    assert dm.retraction_candidate() is a
    dm.retract_req(a)
    assert dm._prompt_lens[a.uid] == 6
    assert dm.next_page_tokens == 4
    b.cached_len, b.device_len = 9, 10
    assert dm.next_page_tokens == 0
    assert dm.inflight_tokens == 4096 + 3


def test_pressure_defers_until_completion_and_requeues_no_duplicate_output(monkeypatch):
    s, reqs, states = setup(monkeypatch, append_latest=False)
    assert s._schedule_next_batch(allow_retract=False) is None
    assert len(s.decode_manager.running_reqs) == 2 and not s.replies
    events = []
    batch = Batch(reqs, "decode")
    done = S(synchronize=lambda: events.append("completed"))
    last = ForwardData(
        ForwardInput(batch, None, None, None),
        (None, torch.tensor([90, 91], dtype=torch.int32), done),
        tuple(r.cached_len for r in reqs),
    )
    s._process_last_data(last)
    assert events == ["completed"] and len(s.replies) == 2
    history = reqs[0].input_ids.clone()
    limit = reqs[0].max_device_len
    assert s._prepare_decode_capacity(allow_retract=True)
    assert len(s.replies) == 2
    assert len(s.prefill_manager.pending_list) == 1
    pending = s.prefill_manager.pending_list[0]
    assert pending.uid == reqs[0].uid
    assert torch.equal(pending.input_ids, history)
    assert len(history) + pending.output_len == limit
    assert pending.cached_prompt_tokens == reqs[0].cached_prompt_tokens
    assert states._owners[reqs[0].table_idx] is None
    assert s.table_manager.available_size == 2
    assert len(s.decode_manager.running_reqs) == 1
    restored = Req(
        pending.input_ids,
        reqs[0].table_idx,
        0,
        pending.output_len,
        pending.uid,
        pending.sampling_params,
        None,
    )
    s.decode_manager.filter_reqs([restored])
    assert s.decode_manager._prompt_lens[restored.uid] == 6
    s.decode_manager.abort_req(restored.uid)
    assert restored.uid not in s.decode_manager._prompt_lens


def test_single_unfit_request_finishes_with_error_instead_of_requeue(monkeypatch):
    s, reqs, states = setup(monkeypatch, count=1, pages=2)
    assert not s._prepare_decode_capacity(allow_retract=True)
    assert not s.prefill_manager.pending_list and not s.decode_manager.runnable
    assert len(s.replies) == 1 and s.replies[0].finished and s.replies[0].error
    assert states._owners[reqs[0].table_idx] is None
    assert s.table_manager.available_size == 3
    assert not s.decode_manager._prompt_lens


def test_discard_only_allocated_private_pages_and_refuses_stale_uid(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=4))
    table = torch.full((1, 16), -777, dtype=torch.int32)
    cache = CacheManager(4, 4, table, "radix")
    req = request(4, 5, 7, 0)
    cache.allocate_paged([S(table_idx=0, cached_len=0, device_len=8)])
    handle = cache.prefix_cache.insert_prefix(req.input_ids[:4], table[0, :4]).handle
    cache.lock(handle)
    cache._handles[0, 7] = handle
    req.cache_handle = handle
    shared = int(table[0, 0])
    private = int(table[0, 4])
    cache.discard_req(req)
    free = cache.free_slots.tolist()
    assert private in free and shared not in free and -777 not in free
    assert len(free) == 3
    before = cache.free_slots.clone()
    with pytest.raises(RuntimeError, match="ownership"):
        cache.discard_req(req)
    assert torch.equal(cache.free_slots, before)


def test_reclaim_evicts_unlocked_prefix_before_live_retraction(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=4))
    table = torch.empty((1, 16), dtype=torch.int32)
    cache = CacheManager(2, 4, table, "radix")
    cache.allocate_paged([S(table_idx=0, cached_len=0, device_len=8)])
    cache.prefix_cache.insert_prefix(torch.arange(8, dtype=torch.int32), table[0, :8])
    assert len(cache.free_slots) == 0
    assert cache.reclaim_capacity(4)
    assert len(cache.free_slots) >= 1


def test_unadmittable_retracted_request_finishes_without_an_infinite_wait(monkeypatch):
    s, _, _ = setup(monkeypatch, output=10000)
    assert s._prepare_decode_capacity(allow_retract=True)
    assert not s.prefill_manager.pending_list
    assert len(s.replies) == 1 and s.replies[0].error
    assert len(s.decode_manager.running_reqs) == 1
