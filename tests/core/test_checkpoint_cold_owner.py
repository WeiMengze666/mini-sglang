"""CPU cold-admission fault flow; does not inject an actual CUDA failure."""

import json
import multiprocessing as mp
from pathlib import Path

import pytest
import torch
from minisgl.engine.engine import WorkerOwner
from minisgl.kvcache.recurrent_pool import RecurrentStatePool


def pool():
    return RecurrentStatePool((0,), 1, 1, 2, 3, torch.bfloat16, "cpu", max_cached_states=1)


def _cold_child(stage, folder):
    p = Path(folder)
    states = pool()
    events = []
    recurrent_ptr = states._recurrent[:, 0].data_ptr()
    conv_ptr = states._conv[:, 0].data_ptr()
    original = torch.Tensor.zero_

    def zero(tensor):
        ptr = tensor.data_ptr()
        kind = "recurrent" if ptr == recurrent_ptr else "conv" if ptr == conv_ptr else "other"
        events.append(kind)
        if kind == stage:
            raise RuntimeError("CPU simulated cold device zero failure")
        return original(tensor)

    def owner(error):
        (p / "owner.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "events": events,
                    "working_owner": states._owners[0],
                    "active": states._active[0],
                    "processed": states._processed_lens[0],
                    "pending": len(states._pending),
                    "cuda_initialized": torch.cuda.is_initialized(),
                }
            )
        )
        import os

        os._exit(9)

    states.device_abort = WorkerOwner(True, owner).abort
    states.release_request = lambda *a: (p / "RELEASED").write_text("unexpected cleanup")
    torch.Tensor.zero_ = zero
    try:
        states.reserve_request(0, 10)
    except BaseException:
        (p / "CAUGHT").write_text("ordinary exception escaped before owner")
    (p / "CONTINUED").write_text("admission continued")


@pytest.mark.parametrize("stage", ["recurrent", "conv"])
def test_actual_worker_owner_exits_before_cold_admission_publication(tmp_path, stage):
    proc = mp.get_context("spawn").Process(target=_cold_child, args=(stage, str(tmp_path)))
    proc.start()
    proc.join(20)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("owned cold-failure process did not exit")
    assert proc.exitcode == 9
    result = json.loads((tmp_path / "owner.json").read_text())
    assert result["events"] == (["recurrent"] if stage == "recurrent" else ["recurrent", "conv"])
    assert result["working_owner"] is None and result["active"] is False
    assert result["processed"] == result["pending"] == 0 and result["cuda_initialized"] is False
    assert not any((tmp_path / n).exists() for n in ["RELEASED", "CAUGHT", "CONTINUED"])


def test_no_owner_preserves_embedding_exception(monkeypatch):
    states = pool()

    def fail(tensor):
        raise RuntimeError("ordinary embedded allocation")

    monkeypatch.setattr(torch.Tensor, "zero_", fail)
    with pytest.raises(RuntimeError, match="ordinary embedded"):
        states.reserve_request(0, 10)
    assert states._owners == [None] and states._active == [False]


def test_CPU_parameter_errors_precede_device_guard(monkeypatch):
    states = pool()
    calls = []
    states.device_abort = lambda error: calls.append(error)

    def unexpected(tensor):
        pytest.fail("invalid CPU arguments reached a device write")

    monkeypatch.setattr(torch.Tensor, "zero_", unexpected)
    with pytest.raises(IndexError):
        states.reserve_request(1, 10)
    with pytest.raises(ValueError, match="negative"):
        states.begin_request(0, 10, -1)
    assert calls == [] and states._owners == [None]


def _warm_setup(patch):
    from test_checkpoint_writes import admit, allocate_pages, make_manager, write_all

    manager = make_manager(patch, False)
    req = admit(manager, end=11)
    allocate_pages(manager, req)
    states = manager.pool.recurrent
    plan = states.reserve_checkpoint(0, 10, 8, end_len=11)
    states.begin_request(0, 10, 0)
    write_all(states, plan)
    states.finish_request(0, 10, 11)
    states.seal_checkpoint(plan)
    req.cached_len = 11
    manager.cache_req(req, finished=False, frontier=8, checkpoint=plan.checkpoint)
    manager.release_state(0, 10)
    handle = manager.prefix_cache.match_prefix(req.input_ids).cuda_handle
    assert handle.cached_len == 8
    manager.lock(handle)
    return manager, states, handle


def _warm_child(stage, folder):
    p = Path(folder)
    events = []
    with pytest.MonkeyPatch.context() as patch:
        manager, states, handle = _warm_setup(patch)
        target_conv, target_state = states._working(1)
        conv_ptr, state_ptr = target_conv.data_ptr(), target_state.data_ptr()
        original = torch.Tensor.copy_

        def copy(tensor, *args, **kwargs):
            ptr = tensor.data_ptr()
            kind = "conv" if ptr == conv_ptr else "recurrent" if ptr == state_ptr else "other"
            events.append(kind)
            if kind == stage:
                raise RuntimeError("CPU simulated warm restore device copy failure")
            return original(tensor, *args, **kwargs)

        def owner(error):
            (p / "owner.json").write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "events": events,
                        "working_owner": states._owners[1],
                        "active": states._active[1],
                        "processed": states._processed_lens[1],
                        "tree_checkpoints": len(manager.prefix_cache.checkpoints),
                        "handle_published": (1, 11) in manager._handles,
                        "cuda_initialized": torch.cuda.is_initialized(),
                    }
                )
            )
            import os

            os._exit(9)

        states.device_abort = WorkerOwner(True, owner).abort
        states.release_request = lambda *a: (p / "RELEASED").write_text("unexpected cleanup")
        torch.Tensor.copy_ = copy
        try:
            manager.restore_state(1, 11, handle)
        except BaseException:
            (p / "CAUGHT").write_text("ordinary exception escaped before owner")
        (p / "CONTINUED").write_text("admission continued")


@pytest.mark.parametrize("stage", ["conv", "recurrent"])
def test_actual_worker_owner_exits_before_warm_admission_publication(tmp_path, stage):
    proc = mp.get_context("spawn").Process(target=_warm_child, args=(stage, str(tmp_path)))
    proc.start()
    proc.join(20)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("owned warm-failure process did not exit")
    assert proc.exitcode == 9
    result = json.loads((tmp_path / "owner.json").read_text())
    assert result["events"] == (["conv"] if stage == "conv" else ["conv", "recurrent"])
    assert result["working_owner"] is None and result["active"] is False
    assert result["processed"] == 0 and result["tree_checkpoints"] == 1
    assert result["handle_published"] is False and result["cuda_initialized"] is False
    assert not any((tmp_path / n).exists() for n in ["RELEASED", "CAUGHT", "CONTINUED"])


def test_no_owner_preserves_warm_admission_exception(monkeypatch):
    manager, states, handle = _warm_setup(monkeypatch)

    def fail(tensor, *args, **kwargs):
        raise RuntimeError("ordinary embedded restore")

    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="ordinary embedded restore"):
        manager.restore_state(1, 11, handle)
    assert states._owners[1] is None and (1, 11) not in manager._handles


def test_warm_CPU_parameter_errors_precede_device_guard(monkeypatch):
    manager, states, handle = _warm_setup(monkeypatch)
    calls = []
    states.device_abort = lambda error: calls.append(error)

    def unexpected(tensor, *a, **k):
        pytest.fail("invalid CPU arguments reached a copy")

    monkeypatch.setattr(torch.Tensor, "copy_", unexpected)
    # Use the actual valid handle; invalid destination must fail before a copy.
    with pytest.raises(IndexError):
        manager.restore_state(99, 11, handle)
    assert calls == []
