"""Communication graph registration must outlive capture and unwind on failure."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed.impl import (
    DistributedCommunicator,
    FallbackDistributedImpl,
    TorchDistributedImpl,
)
from minisgl.distributed.mnnvl import MNNVLAllReduceProvider, MNNVLDistributedImpl
from minisgl.engine.graph import GraphRunner


def test_fallback_registers_each_backend_once_and_unwinds_in_reverse_order():
    events = []

    class Backend(TorchDistributedImpl):
        def __init__(self, name):
            self.name = name

        @contextmanager
        def capture(self):
            events.append(self.name + "-enter")
            try:
                yield
            finally:
                events.append(self.name + "-exit")

    first, second = Backend("primary"), Backend("fallback")
    with FallbackDistributedImpl(first, second).capture():
        events.append("captured")
    assert events == [
        "primary-enter",
        "fallback-enter",
        "captured",
        "fallback-exit",
        "primary-exit",
    ]
    events.clear()
    with FallbackDistributedImpl(first, first).capture():
        pass
    assert events == ["primary-enter", "primary-exit"]


def test_partial_capture_entry_releases_already_entered_backend():
    events = []

    class First(TorchDistributedImpl):
        @contextmanager
        def capture(self):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    class Second(TorchDistributedImpl):
        def capture(self):
            raise RuntimeError("registration unavailable")

    with pytest.raises(RuntimeError, match="registration unavailable"):
        with FallbackDistributedImpl(First(), Second()).capture():
            pytest.fail("capture body must not run")
    assert events == ["enter", "exit"]


@pytest.mark.parametrize("fail", [False, True])
def test_graph_runner_registers_provider_after_cuda_capture(monkeypatch, fail):
    import minisgl.engine.graph as module

    events = []
    mode = {"capture": False, "provider": False}

    class Vendor:
        @contextmanager
        def capture(self):
            assert not mode["capture"]
            mode["provider"] = True
            events.append("provider-enter")
            try:
                yield
            finally:
                assert not mode["capture"], "peer registration cannot run inside CUDA capture"
                mode["provider"] = False
                events.append("provider-exit")

    provider = MNNVLDistributedImpl(MNNVLAllReduceProvider(Vendor()))
    monkeypatch.setattr(DistributedCommunicator, "plugins", [provider])

    class FakeGraph:
        def pool(self):
            return None

    @contextmanager
    def capture_graph(*args, **kwargs):
        assert mode["provider"]
        mode["capture"] = True
        events.append("cuda-enter")
        try:
            yield
        finally:
            mode["capture"] = False
            events.append("cuda-exit")

    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", capture_graph)
    monkeypatch.setattr(module, "get_free_memory", lambda device: 0)
    monkeypatch.setattr(module, "logger", SimpleNamespace(info_rank0=lambda message: None))
    monkeypatch.setattr(module, "get_tp_info", lambda: SimpleNamespace(is_primary=lambda: True))
    monkeypatch.setattr(
        module,
        "get_global_ctx",
        lambda: SimpleNamespace(forward_batch=lambda batch: nullcontext()),
    )
    buffer = SimpleNamespace(
        set_batch=lambda batch: None, logits=torch.zeros(1, 2), out_loc=torch.zeros(1)
    )
    monkeypatch.setattr(module.GraphCaptureBuffer, "init", lambda *args: buffer)

    def forward():
        assert mode["provider"]
        events.append("captured" if mode["capture"] else "warmup")
        if fail and mode["capture"]:
            raise RuntimeError("capture failed")
        return torch.ones(1, 2)

    runner = object.__new__(GraphRunner)
    runner.max_graph_bs = 1
    runner.graph_bs_list = [1]
    runner.device = torch.device("cpu")
    runner.stream = None
    runner.dummy_req = object()
    runner.attn_backend = SimpleNamespace(
        init_capture_graph=lambda **kwargs: None,
        prepare_for_capture=lambda batch: None,
        finish_capture=lambda batch: events.append("state-release"),
    )
    model = SimpleNamespace(apply=lambda fn: None, forward=forward)
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            runner._capture_graphs(8, 2, model)
    else:
        runner._capture_graphs(8, 2, model)
        assert 1 in runner.graph_map
    assert events == [
        "provider-enter",
        "warmup",
        "cuda-enter",
        "captured",
        "cuda-exit",
        "provider-exit",
        "state-release",
    ]
    assert mode == {"capture": False, "provider": False}
