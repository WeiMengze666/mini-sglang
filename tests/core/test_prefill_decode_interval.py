"""Generic prefill/decode cadence; no model or CUDA execution."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as S

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.scheduler import ForwardInput, Scheduler
from minisgl.server.args import parse_args


def request(uid, cls=Req):
    return cls(
        torch.arange(4, dtype=torch.int32), uid, 3, 8, uid, SamplingParams(max_tokens=8), object()
    )


def selector(interval, *, decode=True, capacity=True):
    scheduler = object.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_budget = 16
    pending = deque(Batch([request(i, ChunkedReq if i < 2 else Req)], "prefill") for i in range(3))
    events = []

    def prefill(budget):
        events.append(("admit", budget))
        return pending.popleft() if pending else None

    scheduler.prefill_manager = S(schedule_next_batch=prefill, runnable=True)
    scheduler.decode_manager = DecodeManager(4)
    scheduler.decode_manager.filter_reqs([request(11, ChunkedReq)])
    if decode:
        scheduler.decode_manager.filter_reqs([request(10)])

    def reclaim(needed):
        events.append(("capacity", needed))
        return capacity

    scheduler.cache_manager = S(reclaim_capacity=reclaim)

    def prepare(batch):
        events.append(("prepare", batch.phase))
        return ForwardInput(batch, None, (None, None), (None, None))

    scheduler._prepare_batch = prepare
    return scheduler, pending, events


def phase(forward):
    return None if forward is None else forward.batch.phase


@pytest.mark.parametrize(
    "options,value",
    [([], 0), (["--prefill-decode-interval", "0"], 0), (["--prefill-decode-interval", "2"], 2)],
)
def test_cli_defaults_and_explicit_interval(options, value):
    config, _ = parse_args(["--model-path", "unused", "--dtype", "bfloat16", *options])
    assert config.prefill_decode_interval == value


@pytest.mark.parametrize("value", [-1, True, False, 1.5, None, "1"])
def test_config_rejects_negative_or_noninteger_interval(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        SchedulerConfig(
            "unused", DistributedInfo(0, 1), torch.bfloat16, prefill_decode_interval=value
        )


def test_cli_rejects_negative_and_abbreviated_option():
    base = ["--model-path", "unused", "--dtype", "bfloat16"]
    with pytest.raises(ValueError, match="non-negative integer"):
        parse_args([*base, "--prefill-decode-interval", "-1"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--prefill-decode-interv", "1"])


@pytest.mark.parametrize("interval", [0, 1, 2])
def test_selected_prefill_arms_exactly_n_opportunities(interval):
    scheduler, pending, events = selector(interval)
    sequence = [phase(scheduler._schedule_next_batch()) for _ in range(3 * (interval + 1))]
    assert sequence == (["prefill"] + ["decode"] * interval) * 3
    assert not pending and sum(e[0] == "admit" for e in events) == 3
    # Real unfinished chunks are never admitted as decoders by DecodeManager.
    assert all(type(r) is Req and r.uid == 10 for r in scheduler.decode_manager.running_reqs)


@pytest.mark.parametrize("interval", [1, 2])
def test_idle_opportunities_do_not_wait_for_a_decoder(interval):
    scheduler, pending, events = selector(interval, decode=False)
    assert phase(scheduler._schedule_next_batch()) == "prefill"
    before = tuple(pending)
    for _ in range(interval):
        assert scheduler._schedule_next_batch() is None
        assert tuple(pending) == before
    assert sum(e[0] == "admit" for e in events) == 1
    assert phase(scheduler._schedule_next_batch()) == "prefill"


@pytest.mark.parametrize("interval", [1, 2])
def test_capacity_blocked_opportunities_preserve_real_request(interval):
    scheduler, pending, events = selector(interval, capacity=False)
    assert phase(scheduler._schedule_next_batch()) == "prefill"
    req = next(iter(scheduler.decode_manager.running_reqs))
    saved = req.input_ids.clone(), req.cached_len, req.device_len, req.cache_handle
    for _ in range(interval):
        assert scheduler._schedule_next_batch(allow_retract=False) is None
    assert scheduler._prefill_decode_interval_remaining == 0
    assert torch.equal(req.input_ids, saved[0])
    assert (req.cached_len, req.device_len, req.cache_handle) == saved[1:]
    assert req in scheduler.decode_manager.running_reqs
    assert len(pending) == 2 and sum(e[0] == "admit" for e in events) == 1
    assert phase(scheduler._schedule_next_batch(allow_retract=False)) == "prefill"


def test_decode_and_none_never_rearm_interval():
    scheduler, pending, _ = selector(2)
    assert phase(scheduler._schedule_next_batch()) == "prefill"
    pending.clear()
    for expected in (1, 0, 0):
        assert phase(scheduler._schedule_next_batch()) == "decode"
        assert scheduler._prefill_decode_interval_remaining == expected
    scheduler.decode_manager.remove_req(next(iter(scheduler.decode_manager.running_reqs)))
    assert scheduler._schedule_next_batch() is None
    assert scheduler._prefill_decode_interval_remaining == 0


def test_failed_batch_preparation_does_not_arm_interval():
    scheduler, _, _ = selector(2)

    def fail(batch):
        raise RuntimeError("metadata preparation failed")

    scheduler._prepare_batch = fail
    with pytest.raises(RuntimeError, match="metadata preparation failed"):
        scheduler._schedule_next_batch()
    assert scheduler._prefill_decode_interval_remaining == 0


@pytest.mark.parametrize("overlap", [False, True])
def test_both_loops_use_the_same_cadence_entry_once(overlap):
    scheduler, _, _ = selector(1)
    scheduler.finished_reqs, scheduler.aborted_reqs = set(), set()
    scheduler.stream = object()
    scheduler.engine = S(stream=S(wait_stream=lambda stream: None))
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.receive_msg = lambda **kwargs: []
    scheduler.cache_manager.stage_states = lambda *args, **kwargs: {}
    scheduler.cache_manager.release_staged = lambda staged: None
    forwards, completions, selections = [], [], []

    def forward(inputs):
        forwards.append(inputs.batch.phase)
        return object()

    scheduler._forward = forward
    scheduler._process_last_data = lambda data, **kwargs: completions.append(data)
    original = scheduler._schedule_next_batch

    def select(**kwargs):
        selections.append(kwargs)
        return original(**kwargs)

    scheduler._schedule_next_batch = select
    previous = None
    for _ in range(3):
        if overlap:
            previous = scheduler.overlap_loop(previous)
        else:
            scheduler.normal_loop()
    assert forwards == ["prefill", "decode", "prefill"]
    assert len(selections) == len(completions) == 3
