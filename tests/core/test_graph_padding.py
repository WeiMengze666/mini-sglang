"""Raw-request ownership stays independent of physical CUDA graph rows."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.kda.backend import KDAAttentionBackend, KDAMetadata
from minisgl.attention.kda.registry import KDAKernelSpec, get_kda_kernel
from minisgl.attention.mla.backend import MLAAttentionBackend
from minisgl.attention.mla.cutedsl import CuTeMLADecodePlan
from minisgl.attention.mla.flashinfer import FlashInferMLAAdapter
from minisgl.core import Batch
from minisgl.engine.graph import GraphRunner
from minisgl.kvcache.mla_pool import MLACache
from minisgl.kvcache.recurrent_pool import RecurrentStatePool


def cache_metadata(*, device="cuda", conv=torch.bfloat16, state=torch.bfloat16, arena=None):
    return SimpleNamespace(
        device=torch.device(device),
        dtype=torch.float8_e4m3fn,
        arena=arena,
        recurrent=SimpleNamespace(conv_dtype=conv, state_dtype=state),
    )


@pytest.mark.parametrize(
    "provider,device,conv,state,dim,expected",
    [
        ("sgl", "cuda", torch.bfloat16, torch.bfloat16, 128, True),
        ("sgl", "cpu", torch.bfloat16, torch.bfloat16, 128, False),
        ("sgl", "cuda", torch.float32, torch.bfloat16, 128, False),
        ("sgl", "cuda", torch.bfloat16, torch.float32, 128, False),
        ("sgl", "cuda", torch.bfloat16, torch.bfloat16, 64, False),
        ("torch", "cuda", torch.bfloat16, torch.bfloat16, 128, False),
        ("flashinfer", "cuda", torch.bfloat16, torch.bfloat16, 128, False),
    ],
)
def test_padding_requires_declared_native_state_contract(
    provider, device, conv, state, dim, expected
):
    backend = KDAAttentionBackend(
        SimpleNamespace(cache_config=SimpleNamespace(head_dim=dim)),
        get_kda_kernel(provider),
        cache=cache_metadata(device=device, conv=conv, state=state),
    )
    assert backend.supports_graph_padding is expected
    backend.kernel = replace(get_kda_kernel("sgl"), indexed_decode=None)
    assert not backend.supports_graph_padding


def test_actual_auto_extra_buffer_factory_keeps_native_decode_capability(monkeypatch):
    import minisgl.core as core
    import minisgl.distributed.info as info
    import minisgl.engine.engine as engine
    from minisgl.attention import create_attention_backend
    from minisgl.attention.base import PrefillDecodeAttnBackend
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig

    layout = SimpleNamespace(head_dim=128, state_bytes_per_request=lambda *a, **k: 1)
    model = SimpleNamespace(
        cache_config=layout,
        has_recurrent_state=True,
        is_moe=False,
        num_qo_heads=96,
        is_kda_layer=lambda layer: layer % 2 == 0,
    )
    config = EngineConfig(
        "unused",
        DistributedInfo(0, 8),
        torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        dcp_size=8,
        recurrent_state_dtype=torch.bfloat16,
        cuda_graph_max_bs=48,
        mxfp4_backend="torch",
        mxfp4_activation_precision="bf16",
        distributed_backend="torch",
        state_memory_ratio=0.31,
    )
    monkeypatch.setattr(info, "_TP_INFO", config.tp_info)
    object.__setattr__(config, "model_config", model)
    object.__setattr__(config, "state_cache_strategy", "extra_buffer")
    object.__setattr__(config, "cache_type", "hybrid")
    engine._adjust_config(config)
    assert config.attention_backend == "fi,sgl"
    ctx = core.Context(64, bf16_precision="default")
    ctx.kv_cache = cache_metadata()
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    backend = create_attention_backend(config.attention_backend, model)
    assert isinstance(backend, PrefillDecodeAttnBackend)
    assert backend.prefill_backend.kda_backend.kernel.name == "flashinfer"
    assert not backend.prefill_backend.supports_graph_padding
    assert backend.decode_backend.kda_backend.kernel.name == "sgl"
    assert backend.supports_graph_padding
    backend.decode_backend.mla_backend.kvcache.arena = object()
    assert not backend.supports_graph_padding


def test_kda_replay_clears_old_tail_without_touching_pool_ownership():
    pool = RecurrentStatePool((0,), 32, 1, 2, 2, torch.bfloat16, "cpu", state_dtype=torch.bfloat16)
    pool._conv.fill_(-901)
    pool._recurrent.fill_(-902)
    # Nonidentity ownership makes request-table IDs an invalid state-index oracle.
    tables = [31] + [(i * 7) % 23 for i in range(23)]
    for table in tables:
        pool.begin_request(table, 100 + table, 0)
        state = pool.get_kda_state(0, table)
        state.conv.fill_(table + 1)
        state.recurrent.fill_(table + 2)
        pool.finish_request(table, 100 + table, 1)
    backend = KDAAttentionBackend(None, cache=SimpleNamespace(recurrent=pool))
    captured = KDAMetadata(torch.arange(25, dtype=torch.int32), state_indices=torch.arange(24))
    backend._graph_sizes = {24}
    backend._capture_metadata[24] = captured
    addresses = captured.state_indices.data_ptr(), captured.cu_seqlens_q.data_ptr()
    conv, state = pool._conv.clone(), pool._recurrent.clone()
    owners, slots = list(pool._owners), list(pool._working_slots)
    for step, real in enumerate((17, 9, 21, 1, 17)):
        selected = tables[1:][step:] + tables[1:][:step]
        reqs = [SimpleNamespace(table_idx=t, extend_len=1) for t in selected[:real]]
        batch = Batch(reqs, "decode")
        batch.padded_reqs = reqs + [object()] * (24 - real)
        batch.attn_metadata = KDAMetadata(torch.arange(real + 1, dtype=torch.int32))
        for req in reqs:
            pool.begin_request(
                req.table_idx, 100 + req.table_idx, pool._processed_lens[req.table_idx]
            )
        backend.prepare_for_replay(batch)
        assert captured.state_indices[:real].tolist() == [slots[r.table_idx] for r in reqs]
        assert captured.state_indices[real:].eq(-1).all()
        assert captured.cu_seqlens_q.tolist() == list(range(real + 1)) + [real] * (24 - real)
        assert (captured.state_indices.data_ptr(), captured.cu_seqlens_q.data_ptr()) == addresses
        for req in reqs:
            pool.finish_request(
                req.table_idx, 100 + req.table_idx, pool._processed_lens[req.table_idx] + 1
            )
    torch.testing.assert_close(pool._conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(pool._recurrent, state, rtol=0, atol=0)
    assert pool._owners == owners and pool._working_slots == slots
    assert not any(pool._active)


@pytest.mark.parametrize("native_required", [False, True])
def test_capture_native_decline_cannot_enter_portable_state_gather(monkeypatch, native_required):
    calls = []

    def indexed(*args):
        calls.append("native")
        return None

    def gather(*args):
        calls.append("gather")
        raise LookupError("portable fallback")

    cache = SimpleNamespace(
        device=torch.device("cpu"), recurrent=SimpleNamespace(get_kda_batch=gather, is_idle=True)
    )
    backend = KDAAttentionBackend(
        None, KDAKernelSpec("test", None, indexed_decode=indexed), cache=cache
    )
    # Capture has R=P; its requirement cannot be inferred from x.shape alone.
    batch = Batch([SimpleNamespace(extend_len=1)], "decode")
    if native_required:
        monkeypatch.setattr(
            KDAAttentionBackend, "supports_graph_padding", property(lambda self: True)
        )
        backend._graph_sizes = {1}
        backend.prepare_for_capture(batch)
        assert batch.attn_metadata.native_decode_required
    else:
        backend.prepare_metadata(batch)
    batch.attn_metadata.state_indices = torch.tensor([4])
    error, message = (
        (RuntimeError, "indexed decode declined")
        if native_required
        else (LookupError, "portable fallback")
    )
    with pytest.raises(error, match=message):
        backend.forward_layer(None, 0, torch.zeros(1, 3), batch, cache, None)
    assert calls == (["native"] if native_required else ["native", "gather"])


def test_selected_native_error_is_not_retried_and_missing_physical_indices_fail_first():
    calls = []

    def indexed(*args):
        calls.append("native")
        raise ArithmeticError("native failed")

    cache = SimpleNamespace(
        recurrent=SimpleNamespace(
            get_kda_batch=lambda *a: pytest.fail("native error entered fallback")
        )
    )
    backend = KDAAttentionBackend(
        None, KDAKernelSpec("test", None, indexed_decode=indexed), cache=cache
    )
    batch = Batch([SimpleNamespace(extend_len=1)], "decode")
    batch.attn_metadata = KDAMetadata(torch.tensor([0, 1]), state_indices=torch.tensor([4]))
    with pytest.raises(ArithmeticError, match="native failed"):
        backend.forward_layer(None, 0, torch.zeros(1, 3), batch, cache, None)
    with pytest.raises(RuntimeError, match="physical state-index rows"):
        backend.forward_layer(None, 0, torch.zeros(2, 3), batch, cache, None)
    assert calls == ["native"]


@pytest.mark.parametrize("cp,rank", [(1, 0), (8, 0), (8, 7)])
def test_mla_replay_uses_fresh_physical_metadata_and_reserved_dummy_page(monkeypatch, cp, rank):
    import minisgl.attention.mla.backend as module
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout
    from minisgl.distributed import DistributedInfo

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, cp))
    monkeypatch.setattr(layout, "_DCP_SIZE", cp)
    cache = MLACache((0,), 25 * 64, 512, 64, torch.float8_e4m3fn, "cpu", page_size=64)
    hybrid = SimpleNamespace(mla=cache, device=torch.device("cpu"), arena=None)
    table = torch.arange(25 * 64, dtype=torch.int32).reshape(25, 64)
    # Last row is the existing cache-owned scratch page, outside all real pages.
    table[24].fill_(24 * 64)
    monkeypatch.setattr(module, "get_global_ctx", lambda: SimpleNamespace(page_table=table))
    adapter = FlashInferMLAAdapter()
    plan = CuTeMLADecodePlan(
        torch.empty(1, dtype=torch.uint8), batch_size=24, max_pages=1, page_size=64 // cp
    )
    plan.options = {}
    adapter._graph_plans[24] = plan
    backend = MLAAttentionBackend(None, attention=adapter, cache=hybrid)
    backend._graph_sizes = {24}
    capture = Batch(
        [SimpleNamespace(extend_len=1, table_idx=i, device_len=1) for i in range(24)], "decode"
    )
    capture.padded_reqs = capture.reqs
    backend.prepare_for_capture(capture)
    captured = capture.attn_metadata.decode
    plan.prepare(captured, {})
    addresses = tuple(
        t.data_ptr() for t in (plan.blocks, plan.local, plan.global_, captured.nonempty)
    )
    previous = captured
    runner = object.__new__(GraphRunner)
    runner.attn_backend, runner.max_graph_bs, runner.graph_bs_list = backend, 24, [24]
    events = []
    runner.graph_map = {24: SimpleNamespace(replay=lambda: events.append("replay"))}
    runner.buffer = SimpleNamespace(
        copy_from=lambda batch: events.append("copy"), logits=torch.arange(24 * 3).reshape(24, 3)
    )
    runner.dummy_req = SimpleNamespace(extend_len=1, table_idx=24, device_len=1)
    for step, real in enumerate((17, 9, 21, 1, 17)):
        reqs = [
            SimpleNamespace(extend_len=1, table_idx=i, device_len=(i + step) % 32 + 1)
            for i in range(real)
        ]
        batch = Batch(reqs, "decode")
        runner.pad_batch(batch)
        backend.prepare_metadata(batch)
        live = batch.attn_metadata.decode
        assert live is not previous and len(live.local_lengths) == 24
        output = runner.replay(batch)
        assert output.shape == (real, 3) and output.data_ptr() == runner.buffer.logits.data_ptr()
        assert events[-2:] == ["copy", "replay"]
        expected_global = [r.device_len for r in reqs] + [1] * (24 - real)
        expected_local = [max(0, (n + cp - 1 - rank) // cp) for n in expected_global]
        assert plan.global_.tolist() == expected_global and plan.local.tolist() == expected_local
        assert captured.nonempty.tolist() == [n > 0 for n in expected_local]
        assert plan.blocks[:real, 0].tolist() == [
            i if expected_local[i] else 0 for i in range(real)
        ]
        assert plan.blocks[real:, 0].tolist() == [24 if rank == 0 else 0] * (24 - real)
        assert (
            tuple(t.data_ptr() for t in (plan.blocks, plan.local, plan.global_, captured.nonempty))
            == addresses
        )
        assert batch.attn_metadata.cu_seqlens_q.numel() == real + 1
        assert plan.cu_seqlens_q.numel() == 25
        assert table[:24].max() < table[24, 0]
        previous = live


@pytest.mark.parametrize("capture_size", [None, 1])
def test_mla_capture_native_decline_precedes_store_and_portable_fallback(capture_size):
    from minisgl.attention.mla.backend import MLAMetadata

    calls = []

    def store(*args, **kwargs):
        calls.append("store")
        raise LookupError("ordinary eager store")

    cache = SimpleNamespace(
        device=torch.device("cpu"),
        arena=None,
        mla=SimpleNamespace(
            paged_storage=lambda layer: None,
            _validate_write_locations=lambda loc: None,
            _store_mla=store,
        ),
    )
    attention = SimpleNamespace(supports_graph_padding=True, supports_kv_write=True)
    backend = MLAAttentionBackend(None, attention=attention, cache=cache)

    class Layer:
        def project(self, x, positions, *, is_decode):
            return x, x, x[:, :1]

        def attend_paged_partial(self, *args, **kwargs):
            calls.append("native")
            return None

    batch = Batch([SimpleNamespace(extend_len=1)], "decode")
    batch.positions, batch.out_loc = torch.tensor([0]), torch.tensor([0])
    batch.attn_metadata = MLAMetadata(
        torch.tensor([0, 1]), decode=SimpleNamespace(capture_size=capture_size)
    )
    error, message = (
        (LookupError, "ordinary eager store")
        if capture_size is None
        else (RuntimeError, "MLA decode declined")
    )
    with pytest.raises(error, match=message):
        backend.forward_layer(Layer(), 0, torch.zeros(1, 3), batch, cache, None)
    assert calls == (["native", "store"] if capture_size is None else ["native"])
