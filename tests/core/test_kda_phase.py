"""Explicit KDA phase survives backend, layer, registry and native adaptation."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import KDAState, flashinfer, registry, sgl
from minisgl.attention.kda.backend import KDAAttentionBackend
from minisgl.layers.kda import KDALayer


@pytest.mark.parametrize(
    "lengths,is_decode", [([1], False), ([1, 1], False), ([1, 3], False), ([1, 1], True)]
)
def test_batch_phase_reaches_native_recurrence_with_bf16_state(monkeypatch, lengths, is_decode):
    layer = KDALayer(KDAConfig(2, 1, 2, 4))
    for op in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        op.weight.fill_(0.25)
    layer.A_log.zero_()
    layer.dt_bias.zero_()
    states = [
        KDAState(torch.zeros(3, 2, 3).bfloat16(), torch.zeros(1, 2, 2).bfloat16()) for _ in lengths
    ]
    events = []

    def get_batch(layer_id, indices):
        return KDAState(*(torch.stack([state[i] for state in states]) for i in range(2)))

    def set_batch(layer_id, indices, value):
        for slot in indices.tolist():
            states[slot] = KDAState(*(part[slot] for part in value))

    recurrent = SimpleNamespace(
        batch_indices=lambda slots: torch.tensor(slots),
        get_kda_batch=get_batch,
        set_kda_batch=set_batch,
    )
    cache = SimpleNamespace(
        device=torch.device("cpu"),
        recurrent=recurrent,
        get_kda_state=lambda layer_id, slot: states[slot],
        set_kda_state=lambda layer_id, slot, state: states.__setitem__(slot, state),
    )
    monkeypatch.setattr(layer, "project", lambda x: (x, x, x, x.view(-1, 1, 2), x[:, :1]))
    monkeypatch.setattr(layer, "finish", lambda x, output: output.flatten(-2))
    monkeypatch.setattr(registry, "_supports_cuda_recurrence", lambda q: True)
    # Disable optional complete-decode hooks: this specifically exercises the
    # portable recurrence dispatch used when full fusion is not covered.
    spec = registry.KDAKernelSpec("native-phase", registry.get_kda_kernel("sgl").recurrence)
    backend = KDAAttentionBackend(SimpleNamespace(), spec, cache=cache)

    def chunk(**kwargs):
        assert kwargs["q"].dtype == kwargs["k"].dtype == torch.bfloat16
        assert kwargs["initial_state"].dtype == torch.bfloat16
        assert kwargs["use_qk_l2norm_in_kernel"] and kwargs["beta_is_raw"]
        tokens = kwargs["q"].shape[1]
        assert kwargs["cu_seqlens"].tolist() == [0, tokens]
        assert kwargs["chunk_indices"].tolist() == [[0, 0]]
        kwargs["initial_state"].add_(tokens * 0.125)
        events.append(("chunk", tokens))
        return torch.full_like(kwargs["v"], 2)

    def decode(**kwargs):
        assert kwargs["initial_state"].dtype == torch.bfloat16
        assert kwargs["use_qk_l2norm_in_kernel"]
        kwargs["initial_state"].add_(0.125)
        kwargs["out"].fill_(3)
        events.append(("decode", 1))

    monkeypatch.setattr(sgl, "_kernels", lambda: (chunk, decode, nullcontext))
    batch = SimpleNamespace(
        is_decode=is_decode,
        reqs=[SimpleNamespace(extend_len=length, table_idx=i) for i, length in enumerate(lengths)],
        attn_metadata=SimpleNamespace(state_indices=None),
    )
    output = backend.forward_layer(
        layer, 0, torch.ones(sum(lengths), 2).bfloat16(), batch, cache, None
    )
    expected_events = (
        [("decode" if is_decode else "chunk", 1)]
        if all(length == 1 for length in lengths)
        else [("chunk", length) for length in lengths]
    )
    assert events == expected_events
    torch.testing.assert_close(
        output, torch.full_like(output, 3 if is_decode else 2), rtol=0, atol=0
    )
    for state, length in zip(states, lengths):
        assert state.recurrent.dtype == torch.bfloat16
        torch.testing.assert_close(
            state.recurrent, torch.full_like(state.recurrent, length * 0.125), rtol=0, atol=0
        )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("name", ["torch", "sgl", "flashinfer"])
@pytest.mark.parametrize("tokens", [0, 2])
def test_decode_requires_one_token_before_native_loading(name, tokens):
    q = torch.zeros(tokens, 1, 2)
    args = (q, q, q, q, torch.zeros(tokens, 1), torch.zeros(1), torch.zeros(2), None, -5.0)
    with pytest.raises(ValueError, match="one token per request"):
        registry.get_kda_kernel(name).recurrence(*args, is_decode=True)
    for adapter in (sgl.recurrent_kda, flashinfer.recurrent_kda):
        with pytest.raises(ValueError, match="one token per request"):
            adapter(*args, is_decode=True)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("is_decode", [False, True])
@pytest.mark.parametrize("name", ["sgl", "flashinfer"])
def test_registry_and_flashinfer_fallback_forward_phase(monkeypatch, name, is_decode):
    events = []
    monkeypatch.setattr(registry, "_supports_cuda_recurrence", lambda q: True)

    def native(*args, is_decode):
        events.append(is_decode)
        return args[2], args[0].new_zeros(1, 2, 2)

    monkeypatch.setattr(sgl, "recurrent_kda", native)
    if name == "flashinfer":
        monkeypatch.setattr(flashinfer, "recurrent_kda", native)
    q = torch.zeros(1, 1, 2)
    args = (q, q, q, q, torch.zeros(1, 1), torch.zeros(1), torch.zeros(2), None, -5.0)
    registry.get_kda_kernel(name).recurrence(*args, is_decode=is_decode)
    assert events == [is_decode]


@pytest.mark.parametrize("is_decode", [False, True])
def test_flashinfer_single_token_fallback_keeps_explicit_phase(monkeypatch, is_decode):
    events = []
    monkeypatch.setattr(registry, "_supports_cuda_recurrence", lambda q: True)

    def native(*args, is_decode):
        events.append(is_decode)
        return args[2], args[0].new_zeros(1, 2, 2)

    monkeypatch.setattr(sgl, "recurrent_kda", native)
    q = torch.zeros(1, 1, 2)
    flashinfer.recurrent_kda(
        q,
        q,
        q,
        q,
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.zeros(2),
        None,
        -5.0,
        is_decode=is_decode,
    )
    assert events == [is_decode]


@pytest.mark.parametrize("is_decode", [False, True])
def test_external_provider_receives_phase_without_signature_fallback(monkeypatch, is_decode):
    seen = []
    module = ModuleType("test_explicit_phase_provider")
    portable = registry.get_kda_kernel("torch").recurrence

    def recurrence(*args, is_decode):
        seen.append(is_decode)
        return portable(*args, is_decode=is_decode)

    monkeypatch.setattr(registry.KDA_KERNELS, "_registry", dict(registry.KDA_KERNELS._registry))
    module.register_kda_kernels = lambda: registry.register_kda_kernel("phase-provider", recurrence)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry.load_kda_provider(module.__name__)
    layer = KDALayer(KDAConfig(2, 1, 2, 4))
    for op in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        op.weight.fill_(0.25)
    layer.A_log.zero_()
    layer.dt_bias.zero_()
    x = torch.ones(1, 2)
    args = (x, x, x, x.view(1, 1, 2), x[:, :1])
    actual = layer.update(
        *args, kernel=registry.get_kda_kernel("phase-provider"), is_decode=is_decode
    )
    expected = layer.update(*args, is_decode=is_decode)
    assert seen == [is_decode]
    for result, reference in zip((actual[0], *actual[1]), (expected[0], *expected[1])):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    legacy = registry.KDAKernelSpec("legacy", lambda *args: portable(*args))
    with pytest.raises(TypeError, match="is_decode"):
        layer.update(*args, kernel=legacy, is_decode=is_decode)
