from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.kda.registry import (
    KDA_KERNELS,
    KDADecodeResult,
    KDAKernelSpec,
    get_kda_kernel,
    load_kda_provider,
)


def test_builtin_kda_backends_are_registered():
    assert {"torch", "flashinfer", "sgl"}.issubset(set(KDA_KERNELS.supported_names()))
    assert callable(get_kda_kernel("sgl").decode)
    assert callable(get_kda_kernel("sgl").project)


def test_decode_result_keeps_norm_stage_explicit():
    result = KDADecodeResult(output="core", state="state", normed=True)
    assert result.output == "core"
    assert result.state == "state"
    assert result.normed


def test_kda_provider_loader_uses_registration_factory(monkeypatch):
    import sys
    import types

    module = types.ModuleType("test_kda_provider")
    module.register_kda_kernels = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    load_kda_provider(module.__name__)


def test_cuda_kernel_adapters_fall_back_for_portable_inputs():
    import torch

    torch.manual_seed(21)
    q, k, v, forget = [torch.randn(3, 2, 8) for _ in range(4)]
    args = (q, k, v, forget, torch.randn(3, 2), torch.randn(2), torch.randn(16), None, -5.0)
    expected = get_kda_kernel("torch").recurrence(*args)
    for name in ("flashinfer", "sgl"):
        actual = get_kda_kernel(name).recurrence(*args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("is_decode", [False, True])
@pytest.mark.parametrize("fused_supported", [False, True])
@pytest.mark.parametrize("projection_hook", [False, True])
def test_one_token_prefill_keeps_prefill_kernel_contract(
    is_decode, fused_supported, projection_hook
):
    from minisgl.attention.kda.backend import KDAAttentionBackend

    events, states = [], []
    recurrent = SimpleNamespace(
        batch_indices=lambda slots: torch.tensor(slots),
        get_kda_batch=lambda layer_id, indices: "initial",
        set_kda_batch=lambda layer_id, indices, state: states.append(state),
    )
    cache = SimpleNamespace(device=torch.device("cpu"), recurrent=recurrent)
    spec = KDAKernelSpec(
        "phase-test",
        get_kda_kernel("torch").recurrence,
        lambda layer, x, state: (x + 2, "decoded") if fused_supported else None,
        project=(lambda layer, x: events.append("backend-project") or layer.project(x))
        if projection_hook
        else None,
    )
    backend = KDAAttentionBackend(SimpleNamespace(), spec, cache=cache)

    class Layer:
        def decode(self, x, state, *, kernel):
            events.append("decode")
            assert state == "initial"
            return kernel.decode(self, x, state)

        def project(self, x):
            events.append("project")
            return (x,)

        def update(self, projected, state, *, kernel, is_decode):
            events.append("update")
            assert is_decode == batch.is_decode
            assert projected.shape == (2, 1, 3) and state == "initial"
            return projected + 1, "updated"

        def finish(self, x, output):
            events.append("finish")
            return output

    batch = SimpleNamespace(
        is_decode=is_decode,
        reqs=[SimpleNamespace(extend_len=1, table_idx=i) for i in range(2)],
        attn_metadata=SimpleNamespace(state_indices=None),
    )
    x = torch.zeros(2, 3)
    result = backend.forward_layer(Layer(), 0, x, batch, cache, None)
    uses_fused = is_decode and fused_supported
    torch.testing.assert_close(result, x + (2 if uses_fused else 1), rtol=0, atol=0)
    assert ("backend-project" in events) == (projection_hook and not uses_fused)
    assert states == ["decoded" if uses_fused else "updated"]
    assert ("decode" in events) == is_decode
    if not uses_fused:
        assert events[-3:] == ["project", "update", "finish"]


def test_projection_backend_and_weight_reload_own_their_metadata():
    from minisgl.attention.config import KDAConfig
    from minisgl.layers.kda import KDALayer

    layer = KDALayer(KDAConfig(7, 2, 4, 4, full_rank_gate=True))
    generator = torch.Generator().manual_seed(189)
    weights = layer.state_dict()
    for value in weights.values():
        value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
    x = torch.randn(3, 7, generator=generator)
    calls = []

    def project(op, value):
        calls.append(op)
        return op.project(value * 0.5)

    spec = KDAKernelSpec("projection", get_kda_kernel("torch").recurrence, project=project)
    output, expected_state = layer.update(*layer.project(x * 0.5))
    expected = layer.finish(x, output)
    actual, state = layer.forward(x, kernel=spec)
    assert calls == [layer]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(state, expected_state):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    marker = object()
    assert layer.prepare_kernel("test", lambda: marker) is marker
    assert layer.prepare_kernel("test", lambda: None) is marker
    assert layer.prepare_kernel("none", lambda: None) is None
    assert layer.prepare_kernel("none", lambda: marker) is None
    assert layer.state_dict().keys() == weights.keys()
    layer.load_state_dict({name: value.clone() for name, value in weights.items()})
    assert layer._kernel_plans == {}


def test_native_projection_and_decode_are_lazy_for_cpu(monkeypatch):
    from minisgl.attention.config import KDAConfig
    from minisgl.attention.kda import sgl
    from minisgl.layers.kda import KDALayer

    def unexpected():
        raise AssertionError("CPU execution must not import CUDA-only kernel dependencies")

    monkeypatch.setattr(sgl, "_projection_kernels", unexpected)
    monkeypatch.setattr(sgl, "_decode_kernels", unexpected)
    layer = KDALayer(KDAConfig(7, 2, 4, 4, full_rank_gate=True))
    for weight in layer.state_dict().values():
        weight.fill_(0.1)
    x = torch.ones(3, 7)
    for got, want in zip(get_kda_kernel("sgl").project(layer, x), layer.project(x)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert get_kda_kernel("sgl").decode(layer, x, None) is None
    assert layer._kernel_plans == {}
