"""Communication resource lifetime contracts without GPUs."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def test_communication_resources_close_in_reverse_order(monkeypatch):
    from minisgl.distributed.impl import DistributedCommunicator, destroy_distributed

    events = []
    monkeypatch.setattr(
        DistributedCommunicator,
        "plugins",
        [
            SimpleNamespace(close=lambda: events.append(1)),
            SimpleNamespace(close=lambda: events.append(2)),
        ],
    )
    destroy_distributed()
    assert events == [2, 1] and DistributedCommunicator.plugins == []


@pytest.mark.parametrize("fail_bootstrap", [False, True])
def test_engine_closes_graphs_and_communication_before_bootstrap(monkeypatch, fail_bootstrap):
    import minisgl.engine.engine as module

    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("synchronize"))

    def destroy_bootstrap():
        events.append("bootstrap")
        if fail_bootstrap:
            raise RuntimeError("bootstrap teardown failed")

    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy_bootstrap)
    monkeypatch.setattr(module, "destroy_distributed", lambda: events.append("communication"))
    engine = SimpleNamespace(
        device="cuda",
        graph_runner=SimpleNamespace(destroy_cuda_graphs=lambda: events.append("graphs")),
        _precision_scope=SimpleNamespace(close=lambda: events.append("precision")),
    )
    with (
        pytest.raises(RuntimeError, match="bootstrap teardown failed")
        if fail_bootstrap
        else nullcontext()
    ):
        module.Engine.shutdown(engine)
    assert events == ["synchronize", "graphs", "communication", "bootstrap", "precision"]
