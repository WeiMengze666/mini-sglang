"""Owned-worker failure must precede releasing any in-flight producer lease."""
import json
import multiprocessing
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.engine.engine import WorkerOwner
from minisgl.scheduler.scheduler import Scheduler


def scheduler_with_fault(log_path, case):
    def record(event):
        with Path(log_path).open("a") as f:
            f.write(json.dumps({"event": event, "cuda_initialized": torch.cuda.is_initialized()}) + "\n")

    def error(*args, **kwargs):
        raise RuntimeError("injected producer operation failure")

    owner = WorkerOwner(False, lambda exc: record("fatal"))
    scheduler = object.__new__(Scheduler)
    scheduler.prefill_manager = SimpleNamespace(runnable=False)
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.finished_reqs = set()
    scheduler.aborted_reqs = set()
    scheduler.receive_msg = error if case == "message-with-old-plan" else lambda **kwargs: []
    scheduler._process_one_msg = error
    scheduler.stream = object()
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.engine = SimpleNamespace(
        stream=SimpleNamespace(wait_stream=lambda stream: None),
        kv_cache=SimpleNamespace(recurrent=SimpleNamespace(fail_device=owner.abort)),
    )
    plans = (object(),) if case in ("new-plan", "message-with-old-plan", "old-plan") else ()
    batch = SimpleNamespace(reqs=[], checkpoint_plan=plans, is_prefill=True)
    forward = SimpleNamespace(batch=batch)
    scheduler._schedule_next_batch = lambda **kwargs: forward if case == "new-plan" else None
    scheduler._forward = error
    scheduler.cache_manager = SimpleNamespace(
        stage_states=lambda *args, **kwargs: {"lease": object()} if case == "staged" else {},
        release_staged=lambda staged: record("released"),
    )
    scheduler._process_last_data = error
    last = None if case in ("new-plan", "no-live") else SimpleNamespace(
        inputs=SimpleNamespace(batch=batch), processed_lens=(), checkpoint_plan=plans,
    )
    return scheduler, last, record


def producer_fault_worker(log_path, case):
    scheduler, last, record = scheduler_with_fault(log_path, case)
    try:
        scheduler.overlap_loop(last)
    finally:
        record("python-unwound")


@pytest.mark.parametrize("case", ["new-plan", "old-plan", "staged", "message-with-old-plan"])
def test_live_producer_resources_abort_before_cleanup(tmp_path, case):
    path = tmp_path / "events.jsonl"
    process = multiprocessing.get_context("spawn").Process(
        target=producer_fault_worker, args=(str(path), case)
    )
    process.start()
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail("owned CPU fault probe did not exit")
    assert process.exitcode == 1
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"event": "fatal", "cuda_initialized": False}
    ]


def test_no_live_lease_keeps_ordinary_exception_path(tmp_path):
    path = tmp_path / "events.jsonl"
    scheduler, last, _ = scheduler_with_fault(path, "no-live")
    with pytest.raises(RuntimeError, match="injected producer"):
        scheduler.overlap_loop(last)
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == ["released"]
