"""Real CPU owned-process exits; no CUDA fault is executed."""

import asyncio
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.engine.engine import WorkerOwner, _prepare_checkpoint_model
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.scheduler.scheduler import ForwardData, Scheduler


def _child(kind, folder):
    folder = Path(folder)
    states = RecurrentStatePool(
        (0,),
        1,
        1,
        2,
        3,
        torch.bfloat16,
        "cpu",
        max_cached_states=1,
        state_dtype=torch.bfloat16,
    )
    states.reserve_request(0, 10)
    plan = states.reserve_checkpoint(0, 10, 64, end_len=67)

    def owner(error):
        (folder / "owner.json").write_text(
            json.dumps(
                {
                    "error": str(error),
                    "pending": len(states._pending),
                    "tensor_storage_present": plan.generation in states._checkpoints,
                    "working_owner": states._owners[0],
                    "cuda_initialized": torch.cuda.is_initialized(),
                }
            )
        )
        if kind == "callback_returns":
            return
        if kind == "callback_raises":
            raise ValueError("callback itself failed")
        import os

        os._exit(9)

    states.device_abort = WorkerOwner(True, owner).abort
    original = states.abort_checkpoint

    def abort(p):
        (folder / "ABORTED").write_text("unsafe cleanup before known completion")
        original(p)

    states.abort_checkpoint = abort

    def fault(*args, **kwargs):
        raise RuntimeError("CPU simulated unknown device operation failure")

    if kind in ("borrow", "callback_returns", "callback_raises"):
        pool = object.__new__(HybridCache)
        pool.recurrent = states
        req = SimpleNamespace(table_idx=0, uid=10, cached_len=0, device_len=67)
        batch = SimpleNamespace(reqs=[req], checkpoint_plan=(plan,))
        with pool.forward_batch(batch):
            with states.borrow_checkpoint(plan, 0):
                fault()
    elif kind == "allocation":
        states.abort_checkpoint = original
        original(plan)
        torch.empty_like = fault
        states.reserve_checkpoint(0, 10, 64, end_len=67)
    else:
        scheduler = object.__new__(Scheduler)
        scheduler.engine = SimpleNamespace(
            kv_cache=SimpleNamespace(recurrent=states),
            attn_backend=SimpleNamespace(),
            graph_runner=SimpleNamespace(pad_batch=fault),
        )

        class Cache:
            def coordinates_checkpoints(self, batch):
                return True

            def plan_checkpoints(self, *args):
                return (plan,)

            def checkpoint_metadata_ready(self, *args):
                (folder / "GLOO").write_text("unsafe error collective")

            def release_checkpoint_plan(self, *args, **kwargs):
                (folder / "RELEASED").write_text("unsafe event cleanup")
                abort(plan)

        scheduler.cache_manager = Cache()
        batch = SimpleNamespace(checkpoint_plan=(plan,))
        if kind == "metadata":
            scheduler._prepare_batch(batch)
        elif kind == "event":
            data = ForwardData(
                SimpleNamespace(batch=batch),
                (None, None, SimpleNamespace(synchronize=fault)),
                (),
                (plan,),
            )
            scheduler._process_last_data(data)
    (folder / "CONTINUED").write_text("fatal callback returned to runtime")


@pytest.mark.parametrize("kind", ["borrow", "metadata", "event", "allocation", "callback_returns", "callback_raises"])
def test_real_owned_CPU_exit_precedes_inner_lease_unwind_and_collectives(tmp_path, kind):
    proc = mp.get_context("spawn").Process(target=_child, args=(kind, str(tmp_path)))
    proc.start()
    proc.join(20)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("owned child failed to exit")
    assert proc.exitcode == (1 if kind.startswith("callback_") else 9)
    result = json.loads((tmp_path / "owner.json").read_text())
    assert result["cuda_initialized"] is False
    if kind != "allocation":
        assert result["pending"] == 1 and result["tensor_storage_present"]
        assert result["working_owner"] == 10
    assert not any(
        (tmp_path / name).exists() for name in ["ABORTED", "GLOO", "RELEASED", "CONTINUED"]
    )


def test_embedding_no_owner_disables_automatic_capability_but_still_binds_model():
    events = []
    backend = SimpleNamespace(
        prepare_model=lambda records: events.append("bound"),
        checkpoint_quantum=lambda: pytest.fail("noowner cannot advertise auto tracking"),
    )
    assert _prepare_checkpoint_model(backend, (), None, (), enabled=False) is None
    assert events == ["bound"]
    assert WorkerOwner(fresh_worker=False, abort_callback=lambda error: None).fresh_worker is False


def test_failed_event_without_owner_retains_explicit_manual_lease():
    states = RecurrentStatePool((0,), 1, 1, 2, 3, torch.bfloat16, "cpu", max_cached_states=1)
    states.reserve_request(0, 10)
    plan = states.reserve_checkpoint(0, 10, 64, end_len=67)
    scheduler = object.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(kv_cache=SimpleNamespace(recurrent=states))
    scheduler.cache_manager = SimpleNamespace(
        release_checkpoint_plan=lambda *a, **kw: pytest.fail("unconfirmed lease reclaimed"),
    )

    def fault():
        raise RuntimeError("completion unconfirmed")

    batch = SimpleNamespace(checkpoint_plan=(plan,))
    data = ForwardData(
        SimpleNamespace(batch=batch), (None, None, SimpleNamespace(synchronize=fault)), (), (plan,)
    )
    with pytest.raises(RuntimeError, match="completion unconfirmed"):
        scheduler._process_last_data(data)
    assert plan.generation in states._pending and plan.generation in states._checkpoints


def test_runtime_watcher_stops_exact_borrowed_owned_worker_list(monkeypatch):
    import minisgl.server.api_server as api
    import minisgl.server.launch as launch
    from fastapi import HTTPException

    owned = [object(), object()]
    unowned = object()
    calls = []
    monkeypatch.setattr(launch, "_stop_workers", lambda workers: calls.append(list(workers)))

    def failed():
        raise HTTPException(503, "owned worker exited")

    fake = SimpleNamespace(check_backend=failed, workers=owned)
    # Locate the existing class method; no API server or worker is started.
    cls = next(
        value
        for value in vars(api).values()
        if isinstance(value, type) and "_watch_workers" in vars(value)
    )
    asyncio.run(cls._watch_workers(fake))
    assert calls == [owned] and all(unowned not in workers for workers in calls)
