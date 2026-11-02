from minisgl.attention.kda_backend import (
    KDA_KERNELS,
    KDADecodeResult,
    get_kda_kernel,
    load_kda_provider,
)


def test_builtin_kda_backends_are_registered():
    assert {"torch", "triton", "flashinfer"}.issubset(
        set(KDA_KERNELS.supported_names())
    )
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
