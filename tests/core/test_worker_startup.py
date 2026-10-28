"""A failed TP worker must fail startup and release only its own peers."""

import multiprocessing as mp

import pytest
from minisgl.server.launch import _stop_workers, _wait_for_workers


def _ack_then_wait(queue, event):
    queue.put("first ready")
    queue.put("second ready")
    event.wait()


def _wait(event):
    event.wait()


def _fail():
    raise SystemExit(7)


def test_worker_readiness_while_process_remains_alive():
    context = mp.get_context("spawn")
    queue, stop = context.Queue(), context.Event()
    worker = context.Process(target=_ack_then_wait, args=(queue, stop))
    worker.start()
    try:
        assert list(_wait_for_workers(queue, [worker], 2)) == ["first ready", "second ready"]
        assert worker.is_alive()
    finally:
        _stop_workers([worker])
        queue.close()


def test_nonprimary_failure_is_detected_and_cleanup_is_scoped():
    context = mp.get_context("spawn")
    queue, stop = context.Queue(), context.Event()
    failed = context.Process(target=_fail, name="failed-TP-rank")
    peer = context.Process(target=_wait, args=(stop,))
    unrelated = context.Process(target=_wait, args=(stop,))
    for process in (failed, peer, unrelated):
        process.start()
    try:
        failed.join(timeout=10)
        assert failed.exitcode == 7
        with pytest.raises(RuntimeError, match="failed-TP-rank.*7"):
            list(_wait_for_workers(queue, [failed, peer], 2))
        _stop_workers([failed, peer])
        assert not peer.is_alive()
        assert unrelated.is_alive()
    finally:
        # A terminated Event waiter cannot acknowledge notify_all(). Do not
        # signal a synchronization primitive shared with an exited process.
        _stop_workers([failed, peer, unrelated])
        queue.close()
