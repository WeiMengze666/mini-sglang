from minisgl.attention.kda.registry import (
    KDA_KERNELS,
    KDADecodeResult,
    get_kda_kernel,
    load_kda_provider,
)


def test_builtin_kda_backends_are_registered():
    assert {"torch", "triton", "flashinfer"}.issubset(set(KDA_KERNELS.supported_names()))
    assert get_kda_kernel("triton").decode is None


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
    for name in ("triton", "flashinfer"):
        actual = get_kda_kernel(name).recurrence(*args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
