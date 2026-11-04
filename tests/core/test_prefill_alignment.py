"""Prefill budgeting must preserve reusable frontiers without leaking admission slots."""

from types import SimpleNamespace

import pytest
from minisgl.scheduler.prefill import PrefillAdder


@pytest.mark.parametrize(
    "start,budget,length,replay,expected",
    [
        (0, 23, 80, 0, 16),
        (8, 23, 80, 0, 16),
        (8, 23, 30, 0, 22),
        (8, 23, 80, 24, 16),
        (8, 7, 80, 0, 0),
        (8, 7, 15, 0, 7),
    ],
)
def test_partial_chunks_align_absolute_frontiers_and_final_tails_fit(
    start, budget, length, replay, expected
):
    adder = PrefillAdder(budget, 0, None, None, chunk_alignment=8)
    handle = SimpleNamespace(cached_len=start, replay_until=replay)
    assert adder._chunk_size(SimpleNamespace(input_len=length), handle, start) == expected
    assert adder.token_budget == budget


def test_insufficient_page_remainder_does_not_acquire_a_request_slot():
    handle = SimpleNamespace(cached_len=8, replay_until=0)
    cache = SimpleNamespace(match_req=lambda req: SimpleNamespace(cuda_handle=handle))
    # Neither object implements mutation methods: alignment must reject before
    # locking state or allocating a table slot.
    adder = PrefillAdder(7, 0, cache, SimpleNamespace(available_size=1), chunk_alignment=8)
    assert adder._try_allocate_one(SimpleNamespace(input_len=80)) is None


def test_active_chunk_waits_without_consuming_its_lease_or_budget():
    adder = PrefillAdder(7, 0, None, None, chunk_alignment=8)
    handle = SimpleNamespace(cached_len=8, replay_until=0)
    assert adder._add_one_req(SimpleNamespace(input_len=80), handle, 0, 8) is None
    assert adder.token_budget == 7 and adder.reserved_size == 0


def test_subpage_budget_can_advance_when_alignment_is_disabled():
    adder = PrefillAdder(3, 0, None, None)
    assert adder._chunk_size(SimpleNamespace(input_len=80), SimpleNamespace(), 8) == 3


def test_mixed_prefill_budget_keeps_restorable_intermediate_states(monkeypatch):
    import minisgl.core as core
    import torch
    from minisgl.core import SamplingParams
    from minisgl.kvcache.recurrent_pool import RecurrentStatePool
    from minisgl.message import UserMsg
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
    from minisgl.scheduler.table import TableManager

    # Exercise host scheduling and real cache ownership on CPU.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    empty = torch.empty

    def cpu_empty(*args, pin_memory=False, **kwargs):
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    states = RecurrentStatePool((0,), 3, 1, 2, 3, torch.float32, "cpu", max_cached_states=32)
    pool = SimpleNamespace(arena=None, recurrent=states, release_request=states.release_request)
    ctx = core.Context(page_size=8)
    ctx.kv_cache = pool
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    page_table = torch.full((3, 80), -1, dtype=torch.int32)
    cache = CacheManager(64, 8, page_table, "hybrid", pool)
    table = TableManager(3, page_table)
    prefill = PrefillManager(cache, table, SimpleNamespace(inflight_tokens=0))
    inputs = [
        torch.arange(length, dtype=torch.int32) + 100 * uid
        for uid, length in enumerate((3, 65, 65), 1)
    ]
    for uid, ids in enumerate(inputs, 1):
        prefill.add_one_req(UserMsg(uid, ids, SamplingParams(max_tokens=1)))
    frontiers = []
    rounds = 0
    while prefill.runnable:
        batch = prefill.schedule_next_batch(24)
        assert batch is not None
        assert sum(req.extend_len for req in batch.reqs) <= 24
        if rounds == 0:
            assert [req.extend_len for req in batch.reqs] == [3, 16]
            assert table.available_size == 1  # Five leftover tokens admit no long request.
        cache.allocate_paged(batch.reqs)
        for req in batch.reqs:
            states.begin_request(req.table_idx, req.uid, req.cached_len)
            states.finish_request(req.table_idx, req.uid, req.device_len)
            req.complete_one()
            finished = not isinstance(req, ChunkedReq)
            if not finished:
                assert req.cached_len % 8 == 0
                frontiers.append(req.cached_len)
            cache.cache_req(req, finished=finished)
            if finished:
                cache.release_state(req.table_idx, req.uid)
                table.free(req.table_idx)
        rounds += 1
        assert rounds < 20
    assert frontiers and states.is_idle and table.available_size == 3
    for ids in inputs[1:]:
        assert cache.prefix_cache.match_prefix(ids).cuda_handle.cached_len == 64
    cache.check_integrity()
