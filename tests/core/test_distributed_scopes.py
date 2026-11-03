"""Tensor-sharded KDA state and scoped mature context collectives."""

import os
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.attention.config import KDAConfig
from minisgl.distributed import DistributedInfo
from minisgl.distributed.context import ContextParallel, init_context_parallel
from minisgl.distributed.impl import (
    DistributedCommunicator,
    DistributedImpl,
    FallbackDistributedImpl,
    destroy_distributed,
)
from minisgl.distributed.layout import configure_parallel
from minisgl.distributed.parallel import TensorParallel
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.layers.kda import KDALayer
from minisgl.models.kimi_parallel import shard_spec


def test_scoped_backend_selection_capture_and_single_owner_cleanup(monkeypatch):
    events = []

    class Backend(DistributedImpl):
        def __init__(self, name, value):
            self.name, self.value = name, value

        def all_reduce(self, x):
            return x + self.value

        def all_gather(self, x):
            return x + self.value * 2

        def supports_all_to_all(self, x):
            return self.value > 1

        def all_to_all(self, x):
            return x + self.value * 3

        @contextmanager
        def capture(self):
            events.append((self.name, "enter"))
            yield
            events.append((self.name, "exit"))

        def close(self):
            events.append((self.name, "close"))

    tensor, context = Backend("tensor", 1), Backend("context", 5)
    monkeypatch.setattr(DistributedCommunicator, "plugins", [tensor])
    monkeypatch.setattr(DistributedCommunicator, "groups", {})
    DistributedCommunicator.enable(context, group="context")
    # One provider may serve several scopes; it must own its resources once.
    DistributedCommunicator.enable(context, group="experts")
    x = torch.tensor([3])
    assert DistributedCommunicator().all_reduce(x).item() == 4
    assert DistributedCommunicator("context").all_reduce(x).item() == 8
    assert DistributedCommunicator("experts").all_gather(x).item() == 13
    assert DistributedCommunicator("context").all_to_all(x).item() == 18
    assert FallbackDistributedImpl(tensor, context).all_to_all(x).item() == 18
    with DistributedCommunicator().capture():
        events.append(("body", "capture"))
    assert events == [
        ("tensor", "enter"),
        ("context", "enter"),
        ("body", "capture"),
        ("context", "exit"),
        ("tensor", "exit"),
    ]
    destroy_distributed()
    assert events[-2:] == [("context", "close"), ("tensor", "close")]
    assert DistributedCommunicator.groups == {}
    with pytest.raises(RuntimeError, match="not initialized"):
        DistributedCommunicator("context").all_reduce(x)


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
@pytest.mark.parametrize("rank", range(8))
def test_kda_parameter_state_and_capacity_are_independent_of_dcp(monkeypatch, cp, rank):
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout_module

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 8))
    monkeypatch.setattr(layout_module, "_DCP_SIZE", cp)
    heads, dim = 96, 4
    expected_heads = heads // 8
    layer = KDALayer(KDAConfig(8, heads, dim, 3, full_rank_gate=True))
    assert layer._heads == expected_heads
    assert layer.q_proj.weight.shape == (expected_heads * dim, 8)
    assert layer.o_proj.weight.shape == (8, expected_heads * dim)
    config = HybridCacheConfig(1, (), (0,), 1, 0, heads, dim, 3)
    cache = HybridCache(config, 4, cp * 4, 1, torch.bfloat16, "cpu")
    cache.begin_request(0, 10, 0)
    assert cache.get_kda_state(0, 0).recurrent.shape == (expected_heads, dim, dim)
    assert cache.allocated_bytes == config.state_bytes_per_request(torch.bfloat16, 8)
    text = SimpleNamespace(
        linear_attn_config={"num_heads": heads, "head_dim": dim, "kda_layers": [1]}
    )
    params = [
        ("q_proj.weight", (heads * dim, 8), 0, dim),
        ("o_proj.weight", (8, heads * dim), 1, dim),
        ("A_log", (heads,), 0, 1),
    ]
    for name, shape, axis, block in params:
        spec = shard_spec("model.layers.0.self_attn." + name, text, rank, 8, dcp_size=cp)
        assert spec.axis == axis and spec.length == expected_heads * block
        assert spec.start == rank * expected_heads * block
        assert spec.slice(torch.empty(shape)).shape[axis] == expected_heads * block


def _worker(rank, init_file):
    import minisgl.distributed.info as info

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    info._TP_INFO = DistributedInfo(rank, 4)
    try:
        for cp in (1, 2, 4):
            configure_parallel(dcp_size=cp)
            init_context_parallel()
            tensor, context = TensorParallel(), ContextParallel()
            assert tensor.all_reduce(torch.tensor([rank + 1])).item() == 10
            local = torch.arange(context.rank, 8, context.size, dtype=torch.float32)
            torch.testing.assert_close(
                context.gather_interleaved(local, 8), torch.arange(8).float(), rtol=0, atol=0
            )
            partial = torch.full((1, context.size * 2, 3), float(context.rank + 1))
            lse = torch.zeros(1, context.size * 2)
            expected = torch.full((1, 2, 3), (context.size + 1) / 2)
            torch.testing.assert_close(context.exchange(partial, lse), expected, rtol=0, atol=0)
    finally:
        destroy_distributed()
        configure_parallel()
        info._TP_INFO = None
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in multiprocess gate"
)
def test_tensor_and_context_collective_scopes(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "attention-group"),), nprocs=4, join=True)
