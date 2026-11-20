"""CPU lifetime/count checks; CUDA graph execution remains a separate GPU gate."""

import gc
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import Batch, Context
from minisgl.engine.engine import Engine
from minisgl.engine.graph import GraphRunner
from minisgl.kvcache.base import BaseKVCachePool
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.kvcache.mla_pool import MLACache, MLACacheWrite
from minisgl.kvcache.unified import UnifiedMemory


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("CPU write-scope check initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", fail)
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def make_cache(layers=24):
    config = HybridCacheConfig(layers, tuple(range(layers)), (), 4, 2, 1, 2, 2)
    return HybridCache(config, 4, 4, 2, torch.float32, "cpu")


def write(cache, layer, locations, *, latent=None):
    return MLACacheWrite(
        cache,
        layer,
        torch.ones(len(locations), 4) if latent is None else latent,
        torch.ones(len(locations), 2),
        locations,
        "torch",
    )


def count_validations(monkeypatch, cache):
    calls = []
    original = cache._validate_locations

    def checked(locations):
        calls.append(locations.clone())
        return original(locations)

    monkeypatch.setattr(cache, "_validate_locations", checked)
    return calls


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_24_layer_fresh_prefix_views_share_one_check(monkeypatch, strided, inference):
    cache = make_cache()
    calls = count_validations(monkeypatch, cache.mla)
    with torch.inference_mode(inference):
        backing = torch.arange(1, 7, dtype=torch.int32)
        locations = backing[::2] if strided else backing[:3]
        batch = SimpleNamespace(out_loc=locations, input_ids=torch.zeros(3))
        with cache.model_forward(batch):
            for layer in range(24):
                alias = batch.out_loc[:3]  # The real backend creates this view per layer.
                assert alias is not locations
                write(cache, layer, alias).store()
        assert cache.mla._write_locations is None
        assert len(calls) == 1
        for layer in range(24):
            latent, key = cache.gather_mla(layer, locations)
            assert latent.eq(1).all() and key.eq(1).all()
    assert len(calls) == 25  # Public reads do not borrow the write proof.


@pytest.mark.parametrize("case", ["clone", "offset", "stride", "partial", "dtype", "shape"])
def test_only_the_exact_covered_prefix_can_skip_validation(monkeypatch, case):
    cache = make_cache(1)
    calls = count_validations(monkeypatch, cache.mla)
    backing = torch.arange(1, 9, dtype=torch.int32)
    canonical = backing[:3]
    alternative = {
        "clone": canonical.clone(),
        "offset": backing[1:4],
        "stride": backing[:6:2],
        "partial": canonical[:2],
        "dtype": canonical.long(),
        "shape": canonical.view(1, 3),
    }[case]
    with cache.mla._write_scope(canonical):
        if case == "shape":
            with pytest.raises(ValueError, match="one-dimensional"):
                write(cache, 0, alternative).store()
        else:
            write(cache, 0, alternative).store()
    assert len(calls) == 2


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("case", ["write_neg", "canonical_neg", "both_neg", "double_neg"])
def test_logical_index_views_cannot_borrow_a_different_prefix(monkeypatch, dtype, case):
    from minisgl.kernel import mla as kernel

    mla = MLACache((0,), 16, 512, 64, torch.float8_e4m3fn, "cpu")
    cache = SimpleNamespace(mla=mla)
    calls = count_validations(monkeypatch, mla)
    backing = torch.tensor([1, 2], dtype=dtype)
    canonical = (-backing)._neg_view() if case in ("canonical_neg", "both_neg") else backing
    locations = {
        "write_neg": canonical._neg_view(),
        "canonical_neg": canonical._neg_view(),
        "both_neg": canonical[:],
        "double_neg": canonical._neg_view()._neg_view(),
    }[case]
    assert canonical.untyped_storage() is locations.untyped_storage()
    assert canonical.shape == locations.shape and canonical.stride() == locations.stride()
    assert canonical.storage_offset() == locations.storage_offset()
    native_calls, result = [], object()
    monkeypatch.setattr(
        kernel,
        "store_mla_and_concat_query",
        lambda *args, **kwargs: (native_calls.append(args[1]), result)[1],
    )
    with mla._write_scope(canonical):
        item = MLACacheWrite(cache, 0, torch.ones(2, 512), torch.ones(2, 64), locations, "native")
        if case in ("write_neg", "canonical_neg"):
            with pytest.raises(IndexError, match="out of range"):
                item.fused_query(None, None)
            assert not native_calls
        else:
            assert item.fused_query(None, None) is result
            assert len(native_calls) == 1
    # Logical views go through the old checked path. Resolved double-negation is ordinary.
    assert len(calls) == (1 if case == "double_neg" else 2)
    assert mla._write_locations is None


def test_scope_keeps_storage_alive_and_never_trusts_a_later_batch(monkeypatch):
    cache = make_cache(1)
    calls = count_validations(monkeypatch, cache.mla)
    locations = torch.tensor([1, 2], dtype=torch.int32)
    ref = weakref.ref(locations)
    with cache.mla._write_scope(locations):
        del locations
        gc.collect()
        assert ref() is not None
        write(cache, 0, ref()[:]).store()
    gc.collect()
    assert ref() is None
    reused = torch.tensor([1, 2], dtype=torch.int32)
    with cache.mla._write_scope(reused):
        late_write = write(cache, 0, reused[:])
    reused.fill_(16)  # Same storage, new invocation: must validate the new values.
    with pytest.raises(IndexError, match="out of range"):
        late_write.store()
    with pytest.raises(IndexError, match="out of range"):
        with cache.mla._write_scope(reused):
            pytest.fail("invalid locations entered model scope")
    assert cache.mla._write_locations is None
    assert len(calls) == 4


def test_exception_nested_scope_and_shape_failures_clear_only_their_own_scope(monkeypatch):
    cache = make_cache(1)
    calls = count_validations(monkeypatch, cache.mla)
    locations = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="model failed"):
        with cache.mla._write_scope(locations):
            with pytest.raises(RuntimeError, match="Nested"):
                with cache.mla._write_scope(locations):
                    pytest.fail("nested scope accepted")
            write(cache, 0, locations[:]).store()
            with pytest.raises(ValueError, match="shapes differ"):
                write(cache, 0, locations[:], latent=torch.ones(1, 4)).store()
            raise RuntimeError("model failed")
    assert cache.mla._write_locations is None
    write(cache, 0, locations[:]).store()
    with cache.mla._write_scope(locations):
        write(cache, 0, locations[:]).store()
    assert len(calls) == 3


def test_public_store_and_gather_always_validate_even_during_write_scope(monkeypatch):
    cache = make_cache(1)
    calls = count_validations(monkeypatch, cache.mla)
    locations = torch.tensor([1, 2], dtype=torch.int32)
    with cache.mla._write_scope(locations):
        cache.store_mla(0, torch.ones(2, 4), torch.ones(2, 2), locations[:])
        cache.gather_mla(0, locations[:])
        locations.fill_(16)  # Public APIs must reject even a violated read-only contract.
        with pytest.raises(IndexError, match="out of range"):
            cache.store_mla(0, torch.ones(2, 4), torch.ones(2, 2), locations[:])
        with pytest.raises(IndexError, match="out of range"):
            cache.gather_mla(0, locations[:])
    assert len(calls) == 5


def test_allocated_page_check_survives_the_range_proof():
    arena = UnifiedMemory(4, 4 * 6 * 4, 64, "cpu")
    mla = MLACache((0,), 16, 4, 2, torch.float32, "cpu", arena=arena, page_size=4)
    cache = SimpleNamespace(mla=mla)
    with pytest.raises(RuntimeError, match="allocated before"):
        with mla._write_scope(torch.tensor([1, 2], dtype=torch.int32)):
            write(cache, 0, mla._write_locations[:]).store()
    assert mla._write_locations is None


def test_fused_write_uses_scope_but_public_fused_api_still_checks(monkeypatch):
    from minisgl.kernel import mla as kernel

    mla = MLACache((0,), 16, 512, 64, torch.float8_e4m3fn, "cpu")
    cache = SimpleNamespace(mla=mla)
    calls = count_validations(monkeypatch, mla)
    native_calls, result = [], object()
    monkeypatch.setattr(
        kernel,
        "store_mla_and_concat_query",
        lambda *args, **kwargs: (native_calls.append(args[1]), result)[1],
    )
    locations = torch.tensor([1, 2], dtype=torch.int32)
    latent, key = torch.ones(2, 512), torch.ones(2, 64)
    with mla._write_scope(locations):
        for _ in range(3):
            item = MLACacheWrite(cache, 0, latent, key, locations[:], "native")
            assert item.fused_query(None, None) is result
        assert len(calls) == 1
        assert mla.store_and_concat_query(0, latent, key, locations[:], None, None) is result
        locations.fill_(16)
        with pytest.raises(IndexError, match="out of range"):
            mla.store_and_concat_query(0, latent, key, locations[:], None, None)
    assert len(calls) == 3 and len(native_calls) == 4


def test_non_mla_and_base_hook_do_not_require_location_metadata():
    with BaseKVCachePool.model_forward(object(), object()):
        pass
    cache = object.__new__(HybridCache)
    cache._full_layers = {}
    with cache.model_forward(object()):
        pass


def test_real_engine_eager_scope_clears_on_model_error(monkeypatch):
    cache = make_cache(1)
    ctx = Context(page_size=4)
    ctx.kv_cache = cache
    # Request state ownership is already independently tested; isolate write lifetime.
    monkeypatch.setattr(cache, "forward_batch", lambda batch: nullcontext())
    calls = count_validations(monkeypatch, cache.mla)
    batch = SimpleNamespace(out_loc=torch.tensor([1], dtype=torch.int32), input_ids=torch.zeros(1))
    engine = object.__new__(Engine)
    engine.ctx, engine.stream = ctx, object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: engine.stream)
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)

    def forward():
        write(cache, 0, ctx.batch.out_loc[:]).store()
        raise RuntimeError("real eager failure")

    engine.model = SimpleNamespace(forward=forward)
    with pytest.raises(RuntimeError, match="real eager failure"):
        engine.forward_batch(batch, None)
    assert ctx._batch is None and cache.mla._write_locations is None
    assert len(calls) == 1


def test_real_graph_capture_scopes_warm_and_capture_separately_and_replay_copies_first(monkeypatch):
    import minisgl.distributed.info as info
    import minisgl.engine.graph as module
    from minisgl.distributed.info import DistributedInfo

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    cache = make_cache(1)
    ctx = Context(page_size=4)
    ctx.kv_cache = cache
    monkeypatch.setattr(cache, "forward_batch", lambda batch: nullcontext())
    monkeypatch.setattr(module, "get_global_ctx", lambda: ctx)
    monkeypatch.setattr(module, "get_free_memory", lambda device: 1 << 30)
    monkeypatch.setattr(module, "get_tp_info", lambda: SimpleNamespace(is_primary=lambda: False))
    monkeypatch.setattr(
        module, "DistributedCommunicator", lambda: SimpleNamespace(capture=nullcontext)
    )
    for name in ["synchronize", "empty_cache", "reset_peak_memory_stats"]:
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    trace, state = [], SimpleNamespace(phase="warm")
    original = cache.mla._validate_locations

    def checked(locations):
        trace.append(("validate", state.phase, locations.clone()))
        original(locations)

    monkeypatch.setattr(cache.mla, "_validate_locations", checked)

    class Graph:
        def pool(self):
            return None

        def replay(self):
            assert cache.mla._write_locations is None
            trace.append(("replay", runner.buffer.out_loc.clone()))

    @contextmanager
    def capturing(graph, **kwargs):
        state.phase = "capture"
        try:
            yield
        finally:
            state.phase = "idle"

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capturing)

    def prepare(batch):
        batch.out_loc = torch.tensor([1, 2], dtype=torch.int32)
        trace.append(("prepare",))

    def forward():
        assert cache.mla._write_locations is not None
        trace.append(("model", state.phase))
        write(cache, 0, ctx.batch.out_loc[:]).store()
        return torch.ones(2, 4)

    model = SimpleNamespace(apply=lambda fn: None, forward=forward)
    runner = object.__new__(GraphRunner)
    runner.device, runner.stream, runner.dummy_req = "cpu", None, object()
    runner.max_graph_bs, runner.graph_bs_list = 2, [2]
    runner.attn_backend = SimpleNamespace(
        supports_graph_padding=True,
        init_capture_graph=lambda **kwargs: None,
        prepare_for_capture=prepare,
        finish_capture=lambda batch: trace.append(("finish",)),
        prepare_for_replay=lambda batch: trace.append(("prepare-replay",)),
    )
    runner._capture_graphs(16, 4, model)
    assert [item[:2] for item in trace if item[0] in ("validate", "model")] == [
        ("validate", "warm"),
        ("model", "warm"),
        ("validate", "capture"),
        ("model", "capture"),
    ]
    assert ctx._batch is None and cache.mla._write_locations is None
    validation_count = sum(item[0] == "validate" for item in trace)
    for locations in ([3, 4], [7, 5]):
        batch = Batch([object(), object()], "decode")
        batch.padded_reqs = batch.reqs
        batch.input_ids, batch.positions = (
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )
        batch.out_loc = torch.tensor(locations, dtype=torch.int32)
        runner.replay(batch)
        assert trace[-2][0] == "prepare-replay" and trace[-1][0] == "replay"
        assert trace[-1][1].tolist() == locations
    # This checks real Python wiring, not CUDA execution of the captured assertion.
    assert sum(item[0] == "validate" for item in trace) == validation_count == 2
