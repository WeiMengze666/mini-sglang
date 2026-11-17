"""CPU proof that unreachable cycles are released before allocator sampling."""

import gc
import weakref
from types import SimpleNamespace

import torch
from minisgl.engine import Engine


def test_memory_sample_collects_cycles_before_flushing_allocator(monkeypatch):
    import minisgl.engine.engine as module

    engine = object.__new__(Engine)
    engine.device = torch.device("cpu")
    engine.tp_cpu_group = object()
    events = []
    allocator = {"live": 64, "cached": 0}

    class TemporaryAllocation:
        pass

    def release():
        events.append("release")
        allocator["live"] -= 64
        allocator["cached"] += 64

    def empty_cache():
        events.append("empty_cache")
        allocator["cached"] = 0

    def sample(device):
        events.append("sample")
        return 1000 - allocator["live"] - allocator["cached"]

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset_peak")
    )
    monkeypatch.setattr(module, "get_free_memory", sample)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, **kwargs: events.append("reduce")
    )

    was_enabled = gc.isenabled()
    gc.disable()  # Only explicit collection may release the real Python cycle.
    try:
        temporary = TemporaryAllocation()
        temporary.cycle = temporary
        reference = weakref.ref(temporary)
        finalizer = weakref.finalize(temporary, release)
        del temporary
        assert reference() is not None and finalizer.alive

        # Omit only the new collection point to reproduce the old sampling order.
        with monkeypatch.context() as old_order:
            old_order.setattr(module, "gc", SimpleNamespace(collect=lambda: None))
            assert engine._sync_get_memory() == (936, 936)
        assert reference() is not None and finalizer.alive
        assert allocator == {"live": 64, "cached": 0}

        events.clear()
        assert engine._sync_get_memory() == (1000, 1000)
        assert reference() is None and not finalizer.alive
        assert allocator == {"live": 0, "cached": 0}
        assert engine._local_free_memory == 1000
        assert events == ["synchronize", "release", "empty_cache", "reset_peak", "sample", "reduce"]
        assert not torch.cuda.is_initialized()
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()
