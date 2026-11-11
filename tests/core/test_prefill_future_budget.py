"""Future output admission and replay retain the native scheduler contract."""
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder, PrefillManager
from minisgl.scheduler.utils import CLIP_MAX_NEW_TOKENS, PendingReq


@pytest.fixture(autouse=True)
def host_copy(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda t: t)


def pending(length, output=10000, *, cached_prompt_tokens=None):
    return PendingReq(
        7, torch.arange(length, dtype=torch.int32), SamplingParams(max_tokens=output),
        cached_prompt_tokens=cached_prompt_tokens,
    )


def adder(*, budget=128, available=100000):
    cache = SimpleNamespace(
        page_size=512, available_size=available,
        match_req=lambda req: SimpleNamespace(cuda_handle=SimpleNamespace(cached_len=0)),
        lock=lambda handle: None, unlock=lambda handle: None,
        restore_state=lambda *args: None, release_state=lambda *args: None,
    )
    table = SimpleNamespace(
        available_size=1, allocate=lambda: 0, free=lambda index: None,
        token_pool=torch.empty(1, 1024, dtype=torch.int32),
    )
    return PrefillAdder(
        budget, 0, cache, table, chunk_alignment=64,
        chunk_budget=budget, compute_page_size=64,
    )


def test_new_request_full_prompt_gate_clips_future_at_4096():
    request = pending(128)
    assert request.output_len > CLIP_MAX_NEW_TOKENS
    assert adder(available=5000)._try_allocate_one(request) is not None
    # An unfinished chunk does not erase the initial whole-request feasibility gate.
    assert adder(budget=64, available=4200)._try_allocate_one(request) is None


@pytest.mark.parametrize("start,length,budget,expected_reservation,chunked", [
    (0, 10000, 128, 512, True),
    (0, 65, 128, 512 + 4096, False),
    (64, 128, 128, 4096, False),
])
def test_current_forward_reserves_actual_new_kv_pages_and_only_final_chunk_future(
    start, length, budget, expected_reservation, chunked
):
    add = adder(budget=budget)
    request = pending(length)
    actual = add._add_one_req(request, SimpleNamespace(), 0, start)
    assert isinstance(actual, ChunkedReq) == chunked
    assert add.reserved_size == expected_reservation
    assert add.token_budget == budget - -(-actual.extend_len // 64) * 64


@pytest.mark.parametrize("explicit,old_chunk,expected", [(7, 3, 7), (None, 3, 3), (None, None, 64)])
def test_replay_keeps_original_cached_prompt_usage(explicit, old_chunk, expected):
    request = pending(128, cached_prompt_tokens=explicit)
    if old_chunk is not None:
        request.chunked_req = SimpleNamespace(cached_prompt_tokens=old_chunk)
    actual = adder()._add_one_req(request, SimpleNamespace(), 0, 64)
    assert actual.cached_prompt_tokens == expected


def test_requeue_preserves_host_history_uid_remaining_budget_and_sampling_contract():
    params = SamplingParams(max_tokens=10, ignore_eos=True, temperature=.7, top_k=3)
    req = Req(torch.arange(5, dtype=torch.int32), 0, 0, 10, 77, params, None, cached_prompt_tokens=3)
    req.append_host(torch.tensor([101, 102], dtype=torch.int32))
    manager = PrefillManager(None, None, None)
    manager.requeue(req)
    queued = manager.pending_list[0]
    assert queued.uid == req.uid and queued.input_ids is req.input_ids
    assert queued.output_len == 8 and queued.cached_prompt_tokens == 3
    assert queued.chunked_req is None
    assert queued.sampling_params is not params and params.max_tokens == 10
    assert queued.sampling_params.ignore_eos and queued.sampling_params.temperature == .7
    assert queued.sampling_params.top_k == 3


def test_finished_request_cannot_be_requeued():
    req = Req(torch.arange(2, dtype=torch.int32), 0, 0, 1, 7, SamplingParams(), None)
    req.append_host(torch.tensor([9], dtype=torch.int32))
    manager = PrefillManager(None, None, None)
    with pytest.raises(ValueError, match="unfinished"):
        manager.requeue(req)
    assert not manager.pending_list
