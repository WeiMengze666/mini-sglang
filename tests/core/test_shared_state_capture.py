"""Capture ownership protocol on CPU; no CUDA graph or numerical kernel imitation."""

from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.attention.base import HybridAttnBackend
from minisgl.attention.kda.backend import KDAAttentionBackend
from minisgl.attention.kda.registry import KDAKernelSpec, KDAResult, get_kda_kernel
from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine.engine import _graph_batch_capacity
from minisgl.engine.graph import GraphRunner
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.kvcache.recurrent_pool import RecurrentStatePool


@pytest.mark.parametrize("explicit", [None, [1, 2, 4]])
def test_eight_shared_slots_cap_four_request_capture_to_two(monkeypatch, explicit):
    states = RecurrentStatePool(
        (0,),
        4,
        1,
        128,
        4,
        torch.bfloat16,
        "cpu",
        state_slot_capacity=8,
        tracking_buffers=2,
    )
    maximum = _graph_batch_capacity(4, 128, SimpleNamespace(recurrent=states))
    assert maximum == 2
    # Use the real existing GraphRunner filter; do not change its bucket policy.
    monkeypatch.setattr(GraphRunner, "_capture_graphs", lambda *args: None)
    runner = GraphRunner(
        stream=None,
        device="cpu",
        model=None,
        attn_backend=SimpleNamespace(supports_graph_padding=False),
        cuda_graph_bs=explicit,
        cuda_graph_max_bs=4,
        free_memory=90 << 30,
        max_seq_len=16,
        vocab_size=16,
        dummy_req=object(),
        max_batch_size=maximum,
    )
    assert runner.graph_bs_list == [1, 2]
    assert states.state_usage["free"] == 8  # capacity is not current free-space admission


def test_capture_rejects_capacity_below_one_cold_request():
    states = RecurrentStatePool(
        (0,),
        4,
        1,
        128,
        4,
        torch.bfloat16,
        "cpu",
        state_slot_capacity=2,
        tracking_buffers=2,
    )
    with pytest.raises(MemoryError, match="cannot admit one cold request requiring 3 slots"):
        _graph_batch_capacity(4, 128, SimpleNamespace(recurrent=states))
    assert states.state_usage["free"] == 2


def test_checkpoint_disabled_capture_uses_declared_factor_one():
    states = RecurrentStatePool((0,), 4, 1, 128, 4, torch.bfloat16, "cpu", tracking_buffers=2)
    assert not states.checkpoint_enabled and states.tracking_buffers == 0
    assert _graph_batch_capacity(4, 128, SimpleNamespace(recurrent=states)) == 4


@pytest.mark.parametrize("warmup_failure", [False, True])
def test_real_capture_cleanup_returns_working_and_tracking_before_real_uid(
    monkeypatch, warmup_failure
):
    config = HybridCacheConfig(2, (1,), (0,), 8, 0, 1, 128, 4)
    cache = HybridCache(
        config,
        8,
        2,
        4,
        torch.bfloat16,
        torch.device("cpu"),
        state_slot_capacity=8,
        enable_cache_overlap=True,
    )
    context = core.Context(page_size=2)
    context.kv_cache = cache
    monkeypatch.setattr(core, "_GLOBAL_CTX", context)
    seen = []

    def indexed(layer, x, pool, layer_id, indices, metadata):
        seen.append(indices.clone())
        return KDAResult(x)

    kernel = KDAKernelSpec(
        "cpu-capture-protocol", get_kda_kernel("torch").recurrence, indexed_decode=indexed
    )
    kda = KDAAttentionBackend(SimpleNamespace(cache_config=config), kernel, cache=cache)

    class MLA:
        supports_graph_padding = False

        def init_capture_graph(self, *args, **kwargs):
            pass

        def prepare_for_capture(self, batch):
            batch.attn_metadata = SimpleNamespace()

        def prepare_metadata(self, batch):
            batch.attn_metadata = SimpleNamespace()

        def prepare_for_replay(self, batch):
            pass

        def finish_capture(self, batch):
            pass

    backend = HybridAttnBackend(kda, MLA(), lambda layer: layer == 0)
    backend.init_capture_graph(16, [1, 2])
    layer = SimpleNamespace(finish=lambda x, output, **kwargs: output)
    dummy = Req(torch.tensor([1], dtype=torch.int32), 4, 0, 1, -1, SamplingParams(), None)
    previous = None
    for size in (2, 1, 2):
        batch = Batch([dummy] * size, "decode")
        try:
            backend.prepare_for_capture(batch)
            assert [req.table_idx for req in batch.reqs] == list(range(size))
            assert [req.uid for req in batch.reqs] == [-i - 1 for i in range(size)]
            try:
                with context.forward_batch(batch):
                    for _ in range(2):  # actual GraphRunner warm/capture call sequence
                        backend.forward_layer(layer, 0, torch.zeros(size, 4), batch, cache, None)
                    if warmup_failure:
                        raise RuntimeError("CPU simulated warmup failure")
            except RuntimeError as error:
                assert warmup_failure and "warmup failure" in str(error)
        finally:
            backend.finish_capture(batch)
        assert cache.recurrent.is_idle and cache.recurrent.state_usage["free"] == 8
        assert cache.recurrent.state_usage["working"] == 0
        assert cache.recurrent.state_usage["tracking"] == 0
        assert not kda._capture_reqs
        previous = batch
    # Captured metadata is storage, not ownership. A real UID may occupy a new
    # physical slot; prepare_for_replay must update the captured indices tensor.
    real = Req(torch.tensor([2], dtype=torch.int32), 3, 0, 1, 777, SamplingParams(), None)
    replay = Batch([real], "decode")
    replay.padded_reqs = replay.reqs
    backend.prepare_metadata(replay)
    with context.forward_batch(replay):
        backend.prepare_for_replay(replay)
        expected = cache.recurrent.batch_indices([3])
        assert torch.equal(kda._capture_metadata[1].state_indices, expected)
        backend.finish_capture(previous)  # repeated cleanup cannot release real UID
        assert cache.recurrent._owners[3] == 777
    cache.release_request(3, 777)
    assert cache.recurrent.state_usage["free"] == 8 and seen
