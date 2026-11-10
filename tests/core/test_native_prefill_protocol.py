"""CPU protocol coverage only; fake kernels do not qualify native arithmetic."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.kda import flashinfer, prefill, registry, sgl
from minisgl.attention.kda.backend import KDAAttentionBackend, KDAMetadata
from minisgl.kvcache.recurrent_pool import RecurrentStatePool


def test_cpu_declines_without_loading_native_or_initializing_cuda(monkeypatch):
    def unexpected(*args):
        pytest.fail("CPU input attempted native initialization")

    monkeypatch.setattr(flashinfer, "_prefill_kernel", unexpected)
    monkeypatch.setattr(prefill, "_conv_kernel", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    for adapter in (flashinfer, sgl):
        assert adapter.indexed_prefill(None, torch.zeros(2, 128), None, 0, None, None) is None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("lengths", [(1,), (1, 1), (1, 3)])
def test_prefill_hook_precedes_single_step_gather_and_finish_is_layer_owned(lengths):
    calls = []
    cache = SimpleNamespace(device="cpu")
    cache.recurrent = SimpleNamespace(batch_indices=lambda tables: torch.tensor(tables))

    def hook(layer, x, pool, layer_id, indices, metadata):
        calls.append((indices.tolist(), metadata.lengths, metadata.has_initial_states))
        return x + 1

    spec = registry.KDAKernelSpec("test", None, indexed_prefill=hook)
    backend = KDAAttentionBackend(None, spec, cache=cache)
    batch = SimpleNamespace(
        is_decode=False,
        reqs=[
            SimpleNamespace(extend_len=n, cached_len=17 if i else 0, table_idx=i + 1)
            for i, n in enumerate(lengths)
        ],
    )
    backend.prepare_metadata(batch)
    x = torch.zeros(sum(lengths), 128)
    layer = SimpleNamespace(finish=lambda x, y: y + 2)
    torch.testing.assert_close(backend.forward_layer(layer, 4, x, batch, cache, None), x + 3)
    assert calls == [
        (list(range(1, len(lengths) + 1)), lengths, tuple(i > 0 for i in range(len(lengths))))
    ]


def test_metadata_cached_frontier_is_current_processed_length_and_batch_local():
    backend = KDAAttentionBackend(
        None, registry.get_kda_kernel("flashinfer"), cache=SimpleNamespace(device="cpu")
    )
    req = SimpleNamespace(extend_len=1, cached_len=0, table_idx=2, prefix_len=100)
    batch = SimpleNamespace(reqs=[req], is_decode=False)
    backend.prepare_metadata(batch)
    first = batch.attn_metadata
    assert first.has_initial_states == (False,)
    first.kernel_data["stale"] = object()
    first.state_indices = torch.tensor([999])
    req.cached_len = 23
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.has_initial_states == (True,)
    assert batch.attn_metadata.kernel_data == {}
    assert batch.attn_metadata.state_indices is None


def test_decode_does_not_enter_prefill_or_require_cached_frontier():
    calls = []

    def unexpected(*args):
        pytest.fail("Decode called prefill")

    cache = SimpleNamespace(
        device="cpu",
        recurrent=SimpleNamespace(
            batch_indices=lambda tables: torch.tensor(tables),
            get_kda_batch=lambda *args: "initial",
            set_kda_batch=lambda *args: calls.append("stored"),
        ),
    )
    spec = registry.KDAKernelSpec("test", None, indexed_prefill=unexpected)
    backend = KDAAttentionBackend(None, spec, cache=cache)
    batch = SimpleNamespace(is_decode=True, reqs=[SimpleNamespace(extend_len=1, table_idx=0)])
    backend.prepare_metadata(batch)
    layer = SimpleNamespace(
        project=lambda x: (x,), update=lambda x, state, **kw: (x, state), finish=lambda x, y: y
    )
    backend.forward_layer(layer, 0, torch.zeros(1, 128), batch, cache, None)
    assert calls == ["stored"]


def _fixture(monkeypatch, lengths=(1, 3), dtype=torch.bfloat16):
    events = []
    pool = RecurrentStatePool(
        (2, 7, 11), 5, 1, 128, 4, torch.bfloat16, "cpu", max_cached_states=1, state_dtype=dtype
    )
    # Committed checkpoint precedes active rows and must remain immutable.
    pool.begin_request(0, 50, 0)
    for layer_id in (2, 7, 11):
        for part in pool.get_kda_state(layer_id, 0):
            part.fill_(0.25)
    pool.finish_request(0, 50, 13)
    checkpoint = pool.snapshot(0, 50)
    pool.release_request(0, 50)
    for table, uid in ((1, 101), (4, 104)):
        pool.begin_request(table, uid, 0)
    indices = pool.batch_indices([1, 4])
    metadata = KDAMetadata(
        torch.tensor([0, lengths[0], sum(lengths)]),
        state_indices=indices,
        lengths=lengths,
        has_initial_states=(False, False),
    )
    layer = SimpleNamespace(
        _heads=1,
        _dim=128,
        _config=SimpleNamespace(conv_kernel_size=4, gate_lower_bound=-5.0),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(128),
    )
    layer.q_conv1d = layer.k_conv1d = layer.v_conv1d = SimpleNamespace(
        weight=torch.ones(128, 1, 4).bfloat16()
    )
    layer.prepare_kernel = lambda name, build: build()
    x = torch.arange(sum(lengths) * 128).reshape(sum(lengths), 128).bfloat16()
    projected = (x, x + 1, x + 2, x.view(-1, 1, 128), x[:, :1].contiguous())
    monkeypatch.setattr(prefill, "supports", lambda *args: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(sgl, "project", lambda *args: projected)
    monkeypatch.setattr(sgl, "_kernels", lambda: (None, None, nullcontext))

    def conv(mixed, weights, bias, state, **options):
        events.append(("conv", options))
        assert state.data_ptr() == pool.indexed_kda_state(7)[0].data_ptr()
        for index in options["cache_indices"].tolist():
            state[index].add_(1)
        return mixed.clone()

    monkeypatch.setattr(prefill, "_conv_kernel", lambda: conv)

    def native(*args, **kw):
        events.append(("native", args, kw))
        for index in kw["cache_indices"].tolist():
            kw["ssm_states"][index].add_(2)
        return args[2].clone()

    monkeypatch.setattr(flashinfer, "_prefill_kernel", lambda: SimpleNamespace(extend=native))
    return layer, x, pool, indices, metadata, events, checkpoint


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_forwards_original_storage_raw_gate_options_and_int32_plan(monkeypatch, dtype):
    layer, x, pool, indices, metadata, events, checkpoint = _fixture(monkeypatch, dtype=dtype)
    native = pool.indexed_kda_state(7)[1]
    pointer, stride, offset = native.data_ptr(), native.stride(), native.storage_offset()
    flashinfer.indexed_prefill(layer, x, pool, 7, indices, metadata)
    assert [event[0] for event in events] == ["conv", "native"]
    args, kw = events[1][1:]
    assert kw["ssm_states"].data_ptr() == pointer
    assert kw["ssm_states"].stride() == stride and kw["ssm_states"].storage_offset() == offset
    assert kw["ssm_states"].dtype == dtype
    assert kw["cache_indices"].dtype == torch.int32 and kw["cache_indices"].tolist() == [1, 4]
    assert kw["query_start_loc"].dtype == torch.int32 and kw["query_start_loc"].tolist() == [
        0,
        1,
        4,
    ]
    assert kw["A_log"] is layer.A_log and kw["dt_bias"] is layer.dt_bias
    assert kw["lower_bound"] == -5.0 and kw["return_intermediate_states"] is False
    torch.testing.assert_close(args[3][0], x.view(-1, 1, 128), rtol=0, atol=0)
    torch.testing.assert_close(args[4][0], x[:, :1], rtol=0, atol=0)
    assert args[4].data_ptr() % 16 == 0
    assert not torch.cuda.is_initialized()


def test_multi_layer_inactive_rows_checkpoint_and_uid_reuse_remain_owned(monkeypatch):
    layer, x, pool, indices, metadata, events, checkpoint = _fixture(monkeypatch)
    before = {i: tuple(v.clone() for v in pool.indexed_kda_state(i)) for i in (2, 7, 11)}
    saved_checkpoint = tuple(v.clone() for v in pool._checkpoints[checkpoint.slot])
    flashinfer.indexed_prefill(layer, x, pool, 7, indices, metadata)
    for layer_id, reference in before.items():
        for value, previous in zip(pool.indexed_kda_state(layer_id), reference):
            rows = [0, 2, 3] if layer_id == 7 else list(range(5))
            torch.testing.assert_close(value[rows], previous[rows], rtol=0, atol=0)
    for value, previous in zip(pool._checkpoints[checkpoint.slot], saved_checkpoint):
        torch.testing.assert_close(value, previous, rtol=0, atol=0)
    for table, uid in ((1, 101), (4, 104)):
        pool.finish_request(table, uid, 4)
        pool.release_request(table, uid)
        pool.begin_request(table, uid + 1000, 0)
        pool.release_request(table, uid)  # delayed old owner cannot release new row
    assert pool.batch_indices([4, 1]).tolist() == [4, 1]
    for table, uid in ((1, 1101), (4, 1104)):
        pool.finish_request(table, uid, 1)
        pool.release_request(table, uid)
    pool.restore(0, 900, checkpoint)
    pool.begin_request(0, 900, 13)
    for value, previous in zip(pool._working(0), saved_checkpoint):
        torch.testing.assert_close(value, previous, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["one", "unsafe", "hardware"])
def test_capability_fallback_happens_before_convolution_or_native_mutation(monkeypatch, reason):
    layer, x, pool, indices, metadata, events, checkpoint = _fixture(monkeypatch)
    if reason == "one":
        x = x[:1]
    elif reason == "unsafe":
        layer._config.gate_lower_bound = None
    else:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))

    def fallback(*args):
        assert not events
        return "fallback"

    monkeypatch.setattr(sgl, "indexed_prefill", fallback)
    assert flashinfer.indexed_prefill(layer, x, pool, 7, indices, metadata) == "fallback"
    assert not events


@pytest.mark.parametrize("failure", ["load", "conv", "native", "graph"])
def test_native_error_never_retries_after_partial_state_write(monkeypatch, failure):
    layer, x, pool, indices, metadata, events, checkpoint = _fixture(monkeypatch)

    def fail(*args, **kw):
        raise RuntimeError("sentinel")

    if failure == "load":
        monkeypatch.setattr(flashinfer, "_prefill_kernel", fail)
    elif failure == "conv":
        monkeypatch.setattr(prefill, "_conv_kernel", lambda: fail)
    elif failure == "native":
        monkeypatch.setattr(flashinfer, "_prefill_kernel", lambda: SimpleNamespace(extend=fail))
    else:
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(sgl, "indexed_prefill", lambda *args: pytest.fail("Native failure retried"))
    with pytest.raises(RuntimeError):
        flashinfer.indexed_prefill(layer, x, pool, 7, indices, metadata)
    assert len(events) == (1 if failure == "native" else 0)


@pytest.mark.parametrize("fault", ["lengths", "initial", "rows", "state"])
def test_contract_failure_precedes_any_state_mutation(monkeypatch, fault):
    layer, x, pool, indices, metadata, events, checkpoint = _fixture(monkeypatch)
    if fault == "lengths":
        metadata.lengths = (2, 3)
    elif fault == "initial":
        metadata.has_initial_states = ()
    elif fault == "rows":
        indices = indices[:1]
    else:
        original = pool.indexed_kda_state
        monkeypatch.setattr(
            pool,
            "indexed_kda_state",
            lambda layer_id: (original(layer_id)[0], original(layer_id)[1].transpose(-1, -2)),
        )
    with pytest.raises(ValueError):
        flashinfer.indexed_prefill(layer, x, pool, 7, indices, metadata)
    assert not events


@pytest.mark.parametrize("lengths", [(1, 1), (2, 5)])
def test_all_t1_and_ragged_batches_use_native_with_original_padded_pool(monkeypatch, lengths):
    layer, x, pool, indices, metadata, events, checkpoint = _fixture(monkeypatch, lengths=lengths)
    conv = pool.indexed_kda_state(7)[0]
    backing = torch.zeros(5 * 16400 + 64, dtype=torch.bfloat16)
    state = backing.as_strided((5, 1, 128, 128), (16400, 16384, 128, 1), storage_offset=32)
    assert state.storage_offset() > 0 and state.stride(0) > 16384
    monkeypatch.setattr(pool, "indexed_kda_state", lambda layer_id: (conv, state))
    before = backing.clone()
    flashinfer.indexed_prefill(layer, x, pool, 7, indices, metadata)
    assert [event[0] for event in events] == ["conv", "native"]
    forwarded = events[1][2]["ssm_states"]
    assert forwarded.data_ptr() == state.data_ptr()
    assert forwarded.stride() == state.stride()
    for row in (0, 2, 3):
        assert torch.equal(state[row], torch.zeros_like(state[row]))
    mask = torch.ones_like(backing, dtype=torch.bool)
    for row in (1, 4):
        start = 32 + row * 16400
        mask[start : start + 16384] = False
    assert torch.equal(backing[mask], before[mask])
    assert events[1][1][0].shape[1] == sum(lengths)


@pytest.mark.parametrize("configured", ["auto", "sgl"])
def test_default_sgl_keeps_functional_prefill_and_existing_decode_hooks(monkeypatch, configured):
    import minisgl.distributed.info as distributed_info
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig
    from minisgl.engine.engine import _adjust_config

    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    config = EngineConfig(
        "unused",
        DistributedInfo(0, 1),
        torch.bfloat16,
        attention_backend=configured,
        cuda_graph_max_bs=0,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            cache_config=SimpleNamespace(state_bytes_per_request=lambda *args, **kw: 1),
            has_recurrent_state=True,
            is_moe=False,
            num_qo_heads=1,
        ),
    )
    _adjust_config(config)
    assert config.attention_backend == "sgl"
    spec = registry.get_kda_kernel(config.attention_backend)
    assert spec.indexed_prefill is None
    assert spec.indexed_decode is not None and spec.decode is not None
    cache = SimpleNamespace(
        device="cpu", get_kda_state=lambda *args: "initial", set_kda_state=lambda *args: None
    )
    calls = []

    def forbidden(*args):
        pytest.fail("Default SGL entered the FI-only packed prefill helper")

    monkeypatch.setattr(sgl, "indexed_prefill", forbidden)
    monkeypatch.setattr(sgl, "project", lambda layer, x: (x,))
    backend = KDAAttentionBackend(None, spec, cache=cache)
    layer = SimpleNamespace(
        update=lambda x, state, **kw: (calls.append(kw["is_decode"]) or x, state),
        finish=lambda x, y: y,
    )
    batch = SimpleNamespace(is_decode=False, reqs=[SimpleNamespace(extend_len=3, table_idx=0)])
    backend.prepare_metadata(batch)  # no new cached_len dependency for default SGL
    x = torch.zeros(3, 128)
    torch.testing.assert_close(backend.forward_layer(layer, 0, x, batch, cache, None), x)
    assert calls == [False]


def test_fi_native_call_owns_ffi_stream_context_after_shared_conv(monkeypatch):
    layer, x, pool, indices, metadata, events, checkpoint = _fixture(monkeypatch)
    depth = 0
    entries = []

    class StreamContext:
        def __enter__(self):
            nonlocal depth
            depth += 1
            entries.append("enter")

        def __exit__(self, *args):
            nonlocal depth
            depth -= 1
            entries.append("exit")

    old_conv = prefill._conv_kernel()

    def conv(*args, **kw):
        assert depth == 1
        return old_conv(*args, **kw)

    def native(*args, **kw):
        assert depth == 1
        return args[2]

    monkeypatch.setattr(sgl, "_kernels", lambda: (None, None, StreamContext))
    monkeypatch.setattr(prefill, "_conv_kernel", lambda: conv)
    monkeypatch.setattr(flashinfer, "_prefill_kernel", lambda: SimpleNamespace(extend=native))
    flashinfer.indexed_prefill(layer, x, pool, 7, indices, metadata)
    assert entries == ["enter", "exit", "enter", "exit"]
    assert depth == 0
