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


def _exit_after_ready(ready, stop):
    ready.set()
    stop.wait()
    raise SystemExit(7)


def test_runtime_worker_failure_finishes_pending_requests_and_rejects_new_ones():
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException
    from minisgl.server.api_server import FrontendManager

    context = mp.get_context("spawn")
    ready, stop = context.Event(), context.Event()
    worker = context.Process(target=_exit_after_ready, args=(ready, stop), name="runtime-TP7")
    worker.start()
    try:
        assert ready.wait(timeout=10)

        async def exercise():
            queue = asyncio.Queue()

            async def send(msg):
                pass

            state = FrontendManager(
                SimpleNamespace(model_path="fixture"),
                SimpleNamespace(put=send, stop=lambda: None),
                SimpleNamespace(get=queue.get, stop=lambda: None),
                workers=[worker],
            )
            uids = [state.new_user(), state.new_user()]
            state._create_listener_once()

            async def consume(uid):
                return [reply async for reply in state.wait_for_ack(uid)]

            pending = [asyncio.create_task(consume(uid)) for uid in uids]
            stop.set()
            try:
                replies = await asyncio.wait_for(asyncio.gather(*pending), timeout=10)
                assert all(len(row) == 1 and row[0].finished for row in replies)
                assert all("runtime-TP7" in row[0].error and "7" in row[0].error for row in replies)
                assert state.failed_requests == 2 and state.completed_requests == 0
                assert not state.ack_map and not state.event_map
                with pytest.raises(HTTPException) as error:
                    state.new_user()
                assert error.value.status_code == 503
            finally:
                state.shutdown()
                await asyncio.gather(*state._tasks, return_exceptions=True)

        asyncio.run(exercise())
    finally:
        _stop_workers([worker])


def test_dead_backend_is_not_advertised_as_ready(monkeypatch):
    from types import SimpleNamespace

    import minisgl.server.api_server as api
    from fastapi.testclient import TestClient

    queue = SimpleNamespace(stop=lambda: None)
    state = api.FrontendManager(
        SimpleNamespace(model_path="fixture"),
        queue,
        queue,
        workers=[SimpleNamespace(name="failed-TP", exitcode=7)],
    )
    monkeypatch.setattr(api, "_GLOBAL_STATE", state)
    with TestClient(api.app) as client:
        assert client.get("/v1/models").status_code == 503
        for stream in [False, True]:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fixture",
                    "prompt": "p",
                    "max_tokens": 1,
                    "stream": stream,
                },
            )
            assert response.status_code == 503 and "failed-TP" in response.text
    assert state.uid_counter == 0 and not state.event_map


@pytest.mark.parametrize("frontend_fails", [False, True])
def test_frontend_lifetime_keeps_backend_context_owned(monkeypatch, frontend_fails):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import minisgl.server.api_server as api

    events = []
    owned = [SimpleNamespace(name="owned", exitcode=None)]
    queue = SimpleNamespace(stop=lambda: None)
    config = SimpleNamespace(
        server_host="127.0.0.1",
        server_port=18080,
        zmq_frontend_addr="frontend",
        zmq_tokenizer_addr="tokenizer",
        frontend_create_tokenizer_link=False,
    )
    monkeypatch.setattr(api, "_GLOBAL_STATE", None)
    monkeypatch.setattr(api, "ZmqAsyncPullQueue", lambda *a, **kw: queue)
    monkeypatch.setattr(api, "ZmqAsyncPushQueue", lambda *a, **kw: queue)

    @contextmanager
    def backend():
        events.append("backend-start")
        try:
            yield owned
        finally:
            events.append("backend-cleanup")

    def frontend(*a, **kw):
        assert api.get_global_state().workers is owned
        events.append("frontend")
        if frontend_fails:
            raise RuntimeError("frontend failed")

    monkeypatch.setattr(api.uvicorn, "run", frontend)
    if frontend_fails:
        with pytest.raises(RuntimeError, match="frontend failed"):
            api.run_api_server(config, backend, run_shell=False)
    else:
        api.run_api_server(config, backend, run_shell=False)
    assert events == ["backend-start", "frontend", "backend-cleanup"]


def test_backend_failure_stream_reports_error_and_finishes():
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException
    from minisgl.server.api_server import FrontendManager

    async def exercise():
        queue = SimpleNamespace(stop=lambda: None)
        worker = SimpleNamespace(name="failed-TP", exitcode=None)
        state = FrontendManager(
            SimpleNamespace(model_path="fixture"), queue, queue, workers=[worker], initialized=True
        )
        uid = state.new_user()
        worker.exitcode = 7
        with pytest.raises(HTTPException):
            state.check_backend()
        events = [chunk async for chunk in state.stream_chat_completions(uid)]
        assert len(events) == 2 and b"server_error" in events[0]
        assert events[1] == b"data: [DONE]\n\n"
        assert state.failed_requests == 1 and state.completed_requests == 0
        assert not state.event_map and not state.ack_map

    asyncio.run(exercise())
