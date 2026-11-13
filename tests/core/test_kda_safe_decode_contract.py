"""CPU contracts for the public safe-gate facade and shared indexed metadata."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.kda import sgl
from minisgl.attention.kda.backend import KDAAttentionBackend, KDAMetadata
from minisgl.attention.kda.registry import KDADecodeResult, KDAKernelSpec, get_kda_kernel


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 8, 48])
def test_safe_operator_receives_raw_inputs_views_and_envelope_state(
    monkeypatch, state_dtype, batch
):
    heads, keys, values = 2, 3, 5
    mixed = torch.randn(batch, heads * (2 * keys + values)).bfloat16()
    forget = torch.randn(batch, heads * keys).bfloat16()
    beta = torch.randn(batch, 144).bfloat16()[:, 128 : 128 + heads]
    a, dt = torch.randn(heads), torch.randn(heads * keys)
    state = torch.empty_strided(
        (batch + 2, heads, values, keys), (128, values * keys, keys, 1), dtype=state_dtype
    )
    state.zero_()
    indices = torch.arange(batch, dtype=torch.long)
    qsl = torch.arange(batch + 1, dtype=torch.long)
    seen = []

    def native(**kw):
        seen.append(kw)
        assert kw["initial_state_source"] is state and kw["initial_state_indices"] is indices
        assert kw["cu_seqlens"] is qsl and kw["a"] is forget
        assert kw["b"].data_ptr() == beta.data_ptr()
        assert kw["b"].stride()[1:] == beta.stride() == (144, 1)
        assert kw["A_log"] is a and kw["dt_bias"] is dt
        assert kw["q"].shape == kw["k"].shape == (1, batch, heads, keys)
        assert kw["v"].shape == (1, batch, heads, values)
        for offset, name in zip((0, heads * keys, 2 * heads * keys), ("q", "k", "v")):
            assert kw[name].untyped_storage()._cdata == mixed.untyped_storage()._cdata
            assert kw[name].storage_offset() == offset
        assert kw["is_kda"] and kw["use_qk_l2norm_in_kernel"] and kw["lower_bound"] == -5.0
        assert kw["softplus_beta"] == 1.0 and kw["softplus_threshold"] == 20.0
        state[0].fill_(3)
        return kw["v"].clone()

    monkeypatch.setattr(sgl, "_safe_decode_kernel", lambda: native)
    output = sgl._safe_decode(mixed, forget, beta, a, dt, state, indices, qsl, -5.0)
    assert len(seen) == 1 and output.shape == (1, batch, heads, values)
    assert state.dtype == state_dtype and state.stride(0) == 128
    assert state[0].eq(3).all() and state[1:].eq(0).all()


def test_indexed_callback_receives_shared_metadata_without_rebuilding_offsets():
    indices = torch.tensor([0, 2])
    metadata = KDAMetadata(torch.tensor([0, 1, 2]), state_indices=indices)
    calls = []
    cache = SimpleNamespace(device=torch.device("cpu"), recurrent=object())

    def indexed(layer, x, pool, layer_id, slots, batch_metadata):
        assert pool is cache.recurrent and slots is indices and batch_metadata is metadata
        calls.append(batch_metadata.cu_seqlens_q)
        return KDADecodeResult(x + 1)

    spec = KDAKernelSpec(
        "metadata-test", get_kda_kernel("torch").recurrence, indexed_decode=indexed
    )
    backend = KDAAttentionBackend(SimpleNamespace(), spec, cache=cache)
    batch = SimpleNamespace(
        is_decode=True,
        reqs=[SimpleNamespace(extend_len=1) for _ in range(2)],
        attn_metadata=metadata,
    )
    layer = SimpleNamespace(finish=lambda x, output, normed, gate=None: output)
    x = torch.zeros(2, 3)
    torch.testing.assert_close(backend.forward_layer(layer, 7, x, batch, cache, None), x + 1)
    assert calls == [metadata.cu_seqlens_q]


@pytest.mark.parametrize("bound", [None, -5.0])
def test_indexed_safe_dispatch_keeps_metadata_and_pool_identity(monkeypatch, bound):
    # Mark only the guard property; all storage/ops remain CPU and no native module loads.
    class GuardTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    x = torch.zeros(2, 7, dtype=torch.bfloat16).as_subclass(GuardTensor)
    conv = torch.zeros(4, 6 * 128, 3, dtype=torch.bfloat16)
    state = torch.zeros(4, 2, 128, 128, dtype=torch.bfloat16)
    indices = torch.tensor([0, 2])
    metadata = KDAMetadata(torch.tensor([0, 1, 2]), state_indices=indices)
    pool = SimpleNamespace(
        state_dtype=torch.bfloat16,
        conv_dtype=torch.bfloat16,
        indexed_kda_state=lambda layer: (conv, state),
    )
    layer = SimpleNamespace(
        _dim=128,
        _heads=2,
        _config=SimpleNamespace(gate_lower_bound=bound),
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(256),
        prepare_kernel=lambda name, create: object(),
    )
    projected = [torch.zeros(2, 256, dtype=torch.bfloat16) for _ in range(4)] + [
        torch.zeros(2, 2, dtype=torch.bfloat16)
    ]
    calls = []

    def convolution(mixed, cache, weights, **kw):
        assert cache is conv and kw["conv_state_indices"] is indices
        return mixed

    def safe(mixed, forget, beta, a, dt, cache, slots, qsl, lower):
        assert cache is state and slots is indices and qsl is metadata.cu_seqlens_q
        calls.append("safe")
        return torch.ones(1, 2, 2, 128, dtype=torch.bfloat16)

    def packed(**kw):
        assert kw["initial_state"] is state and kw["ssm_state_indices"] is indices
        assert "lower_bound" not in kw
        calls.append("packed")
        kw["out"].fill_(1)

    monkeypatch.setattr(sgl, "project", lambda op, value: projected)
    monkeypatch.setattr(sgl, "_safe_decode", safe)
    monkeypatch.setattr(sgl, "_safe_decode_kernel", lambda: calls.append("ready"))
    monkeypatch.setattr(sgl, "_kernels", lambda: (None, packed, nullcontext))
    from minisgl.attention.kda import causal_conv

    monkeypatch.setattr(causal_conv, "_kernels", lambda: (None, convolution))
    result = sgl.indexed_decode(layer, x, pool, 7, indices, metadata)
    assert calls == (["ready", "safe"] if bound is not None else ["packed"])
    assert result.output.shape == (2, 2, 128) and result.state is None and not result.normed


def test_registry_forwards_indexed_metadata_lazily(monkeypatch):
    calls = []
    monkeypatch.setattr(sgl, "indexed_decode", lambda *args: calls.append(args) or "result")
    args = tuple(object() for _ in range(6))
    assert get_kda_kernel("sgl").indexed_decode(*args) == "result"
    assert calls == [args]


def test_cpu_indexed_fallback_does_not_load_native_operator(monkeypatch):
    monkeypatch.setattr(
        sgl, "_safe_decode_kernel", lambda: pytest.fail("CPU loaded native operator")
    )
    assert sgl.indexed_decode(None, torch.zeros(1, 3), None, 0, None, None) is None


def test_indexed_safe_import_failure_precedes_any_pool_write(monkeypatch):
    class GuardTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    def unavailable():
        raise ImportError("public operator unavailable")

    monkeypatch.setattr(sgl, "_safe_decode_kernel", unavailable)
    pool = SimpleNamespace(
        state_dtype=torch.bfloat16,
        conv_dtype=torch.bfloat16,
        indexed_kda_state=lambda _: pytest.fail("operator import must precede state access"),
    )
    layer = SimpleNamespace(_dim=128, _config=SimpleNamespace(gate_lower_bound=-5.0))
    with pytest.raises(ImportError, match="public operator unavailable"):
        sgl.indexed_decode(
            layer,
            torch.zeros(1, 7, dtype=torch.bfloat16).as_subclass(GuardTensor),
            pool,
            0,
            None,
            None,
        )


def test_indexed_fp32_state_keeps_existing_fallback_without_loading_operator(monkeypatch):
    class GuardTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    monkeypatch.setattr(
        sgl, "_safe_decode_kernel", lambda: pytest.fail("FP32 indexed path loaded operator")
    )
    pool = SimpleNamespace(state_dtype=torch.float32, conv_dtype=torch.bfloat16)
    assert (
        sgl.indexed_decode(
            None,
            torch.zeros(1, 7, dtype=torch.bfloat16).as_subclass(GuardTensor),
            pool,
            0,
            None,
            None,
        )
        is None
    )
