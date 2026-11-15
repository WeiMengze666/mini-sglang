"""CPU-only state-write/FP8 query handoff contracts; native math is a GPU gate."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mla import MLAOutput
from minisgl.attention.mla.backend import MLAAttentionBackend
from minisgl.attention.mla.flashinfer import FlashInferMLAAdapter
from minisgl.kernel import mla as kernel
from minisgl.kvcache.mla_pool import MLACache, MLACacheWrite


class CudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def cuda(*shape, dtype=torch.bfloat16):
    return torch.zeros(shape, dtype=dtype).as_subclass(CudaTensor)


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("CPU MLA protocol check initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", fail)
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


@pytest.mark.parametrize("outcome", ["accepted", "declined", "failed_after_write"])
def test_backend_write_is_consumed_exactly_once_and_uses_current_forward_locations(
    monkeypatch, outcome
):
    import minisgl.attention.mla.backend as module

    monkeypatch.setattr(
        module,
        "ContextParallel",
        lambda: SimpleNamespace(rank=0, size=1, gather_interleaved=lambda x, total: x),
    )
    writes, offered = [], []
    x = torch.ones(1, 4)
    cache = SimpleNamespace(
        store_mla=lambda *args, **kwargs: writes.append(args[3]),
        gather_mla=lambda *args: (x, x[:, :1]),
        mla=SimpleNamespace(paged_storage=lambda layer: object()),
    )

    class Layer:
        def project(self, x, positions, **kwargs):
            return x, x, x[:, :1]

        def attend_paged_partial(self, q, pages, metadata, *, attention, kv_write):
            offered.append(kv_write.locations)
            if outcome == "declined":
                return None
            kv_write.store()
            if outcome == "failed_after_write":
                raise RuntimeError("native launch failed")
            return MLAOutput(q)

        def attend_partial(self, q, *args, **kwargs):
            return MLAOutput(q)

        def finish(self, x, partial, **kwargs):
            return partial.values

    attention = SimpleNamespace(supports_kv_write=True)
    backend = MLAAttentionBackend(None, attention=attention, cache=cache)
    batch = SimpleNamespace(
        is_decode=True,
        positions=torch.tensor([0]),
        out_loc=torch.tensor([0]),
        attn_metadata=SimpleNamespace(decode=object()),
        reqs=[SimpleNamespace(extend_len=1, device_len=1, cached_len=0, table_idx=0)],
    )
    for loc in [torch.tensor([0]), torch.tensor([63])]:
        batch.out_loc = loc  # Capture's set_batch may replace the original tensor.
        before = len(writes)
        if outcome == "failed_after_write":
            with pytest.raises(RuntimeError, match="native launch failed"):
                backend.forward_layer(Layer(), 2, x, batch, cache, torch.tensor([[0]]))
        else:
            result = backend.forward_layer(Layer(), 2, x, batch, cache, torch.tensor([[0]]))
            torch.testing.assert_close(result, x, rtol=0, atol=0)
        assert len(writes) == before + 1
        assert offered[-1].data_ptr() == writes[-1].data_ptr() == loc.data_ptr()


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("failure", [False, True])
def test_selected_decoder_writes_before_attention_without_fallback_or_retry(
    monkeypatch, fused, failure
):
    calls = []
    q = cuda(2, 1, 192)
    pages = cuda(2, 64, 576, dtype=torch.float8_e4m3fn)
    ready = cuda(2, 1, 576, dtype=torch.float8_e4m3fn)

    class Write:
        def fused_query(self, query, suffix):
            calls.append("offer_fused")
            assert query.shape == (2, 1, 512) and suffix.shape == (2, 1, 64)
            return ready if fused else None

        def store(self):
            calls.append("store")

    class Plan:
        def prepare(self, *args):
            calls.append("prepare")

        def run(self, query, cache, **kwargs):
            calls.append("attend")
            assert (query is ready) if fused else isinstance(query, tuple)
            if failure:
                raise RuntimeError("attention launch failed")
            return q, torch.zeros(2, 1)

    adapter = FlashInferMLAAdapter()
    monkeypatch.setattr(adapter, "_plan", lambda *args: Plan())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 3))
    meta = SimpleNamespace(block_table=object(), page_size=64)
    args = (q, pages, torch.zeros(1, 256, 512).bfloat16(), meta)
    options = {"nope_dim": 128, "value_dim": 128, "kv_write": Write()}
    if failure:
        with pytest.raises(RuntimeError, match="attention launch failed"):
            adapter.decode(*args, **options)
    else:
        adapter.decode(*args, **options)
    assert calls == ["prepare", "offer_fused"] + ([] if fused else ["store"]) + ["attend"]


def test_decoder_declines_before_any_write(monkeypatch):
    adapter = FlashInferMLAAdapter()
    monkeypatch.setattr(adapter, "_plan", lambda *args: pytest.fail("unsupported plan"))
    write = SimpleNamespace(
        store=lambda: pytest.fail("unsupported decoder wrote cache"),
        fused_query=lambda *args: pytest.fail("unsupported decoder fused cache"),
    )
    assert (
        adapter.decode(
            torch.zeros(2, 1, 192),
            torch.zeros(2, 64, 576),
            None,
            None,
            nope_dim=128,
            value_dim=128,
            kv_write=write,
        )
        is None
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("covered", [False, True])
def test_sdk_receives_original_virtual_locations_and_instance_inputs(monkeypatch, dtype, covered):
    storage, locations = cuda(128, 576, dtype=torch.float8_e4m3fn), cuda(3, dtype=dtype)
    latent, key, query, suffix = cuda(3, 512), cuda(3, 64), cuda(3, 12, 512), cuda(3, 12, 64)
    result, calls = cuda(3, 12, 576, dtype=torch.float8_e4m3fn), []

    def cover(*args):
        assert all(a is b for a, b in zip(args, (storage, locations, latent, key, query, suffix)))
        calls.append("covered")
        return covered

    def run(*args, **options):
        calls.append("run")
        assert all(a is b for a, b in zip(args, (storage, locations, latent, key, query, suffix)))
        assert options == {"dcp_world_size": 8, "dcp_rank": 7}
        return result

    sdk = SimpleNamespace(covered_fp8=cover, set_mla_kv_concat_q_fp8=run)
    monkeypatch.setattr(kernel, "_concat_query_kernels", lambda: (sdk, nullcontext))
    actual = kernel.store_mla_and_concat_query(
        storage, locations, latent, key, query, suffix, cp_size=8, cp_rank=7
    )
    assert actual is (result if covered else None)
    assert calls == (["covered", "run"] if covered else ["covered"])


def test_native_writer_error_propagates_once(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("native writer failed")

    sdk = SimpleNamespace(covered_fp8=lambda *args: True, set_mla_kv_concat_q_fp8=fail)
    monkeypatch.setattr(kernel, "_concat_query_kernels", lambda: (sdk, nullcontext))
    with pytest.raises(RuntimeError, match="native writer failed"):
        kernel.store_mla_and_concat_query(
            cuda(128, 576, dtype=torch.float8_e4m3fn),
            cuda(2, dtype=torch.int32),
            cuda(2, 512),
            cuda(2, 64),
            cuda(2, 12, 512),
            cuda(2, 12, 64),
            cp_size=8,
            cp_rank=7,
        )
    assert calls == [True]


@pytest.mark.parametrize("arena", [None, object()])
def test_pool_dense_view_keeps_dtype_locations_and_avoids_arena_reinterpretation(
    monkeypatch, arena
):
    cache = object.__new__(MLACache)
    cache.arena, cache.latent_dim, cache.rope_dim = arena, 512, 64
    cache._device, cache.num_tokens = torch.device("cpu"), 256
    cache._cp = SimpleNamespace(size=8, rank=7)
    cache._full_layers = {3: 1}
    cache._mla = torch.zeros(2, 32, 576).to(torch.float8_e4m3fn)
    loc = torch.tensor([7, 63, 255], dtype=torch.int32)
    latent, key, query, suffix = torch.zeros(3, 512), torch.zeros(3, 64), object(), object()
    result, calls = object(), []

    def run(storage, locations, *args, **kwargs):
        calls.append(storage)
        assert storage.data_ptr() == cache._mla[1].data_ptr()
        assert locations is loc and locations.dtype == torch.int32
        assert kwargs == {"cp_size": 8, "cp_rank": 7}
        return result

    monkeypatch.setattr(kernel, "store_mla_and_concat_query", run)
    got = cache.store_and_concat_query(3, latent, key, loc, query, suffix)
    assert got is (result if arena is None else None)
    assert len(calls) == int(arena is None)


def test_write_policy_and_invalid_locations_cannot_launch(monkeypatch):
    called = []
    write = MLACacheWrite(SimpleNamespace(mla=object()), 3, None, None, None, "torch")
    assert write.fused_query(None, None) is None
    cache = MLACache((3,), 128, 512, 64, torch.float8_e4m3fn, "cpu", page_size=64)
    monkeypatch.setattr(
        kernel, "store_mla_and_concat_query", lambda *args, **kwargs: called.append(True)
    )
    with pytest.raises(IndexError, match="out of range"):
        cache.store_and_concat_query(
            3, torch.zeros(1, 512), torch.zeros(1, 64), torch.tensor([128]), None, None
        )
    assert called == []
