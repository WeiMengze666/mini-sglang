from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
from minisgl.layers import activation


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 8])
def test_cpu_preserves_dtype_broadcast_and_input_ownership(monkeypatch, dtype, strided, rows):
    def unexpected():
        raise AssertionError("CPU fallback must not import native code")

    monkeypatch.setattr(activation, "_sigmoid_mul_kernel", unexpected)
    gate = torch.linspace(-5, 5, 24, dtype=dtype).reshape(3, 8)
    gate = gate[:, ::2] if strided else gate
    up = torch.ones((rows, 1, gate.shape[-1]), dtype=dtype)
    original_gate, original_up = gate.clone(), up.clone()
    expected = up * gate.sigmoid()
    actual = activation.sigmoid_and_mul(gate, up)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(gate, original_gate) and torch.equal(up, original_up)
    assert not torch.cuda.is_initialized()


def test_bf16_sigmoid_rounding_precedes_multiply():
    gate = torch.linspace(-5, 5, 1001, dtype=torch.bfloat16)
    up = torch.full_like(gate, 1.1015625)
    expected = up * gate.sigmoid()
    single_rounding = (up.float() * gate.float().sigmoid()).to(torch.bfloat16)
    assert torch.any(expected != single_rounding), "Fixture must distinguish rounding contracts"
    assert torch.equal(activation.sigmoid_and_mul(gate, up), expected)


@dataclass
class DeviceView:
    value: torch.Tensor
    device: str = "cuda:0"
    address: int = 256
    requires_grad: bool = False
    is_cuda: bool = True

    @property
    def dtype(self):
        return self.value.dtype

    def numel(self):
        return self.value.numel()

    def data_ptr(self):
        return self.address

    def sigmoid(self):
        return self.value.sigmoid()

    def __mul__(self, other):
        return self.value * other


@pytest.mark.parametrize(
    "case", ["native", "layout", "device", "gradient", "alignment", "SDK_absent"]
)
def test_native_eligibility_and_stream_scope(monkeypatch, case):
    gate = DeviceView(torch.linspace(-2, 2, 8, dtype=torch.bfloat16))
    up = DeviceView(torch.full((8,), 1.1015625, dtype=torch.bfloat16))
    events = []

    def covered(actual_up, actual_gate):
        assert actual_up is up and actual_gate is gate
        events.append("covered")
        return case != "layout"

    @contextmanager
    def stream():
        events.append("enter")
        yield
        events.append("exit")

    def run(actual_up, actual_gate):
        assert events[-1] == "enter"
        events.append("run")
        return actual_up.value * actual_gate.value.sigmoid()

    monkeypatch.setattr(
        activation,
        "_sigmoid_mul_kernel",
        lambda: None if case == "SDK_absent" else (covered, run, stream),
    )
    if case == "device":
        up.device = "cuda:1"
    elif case == "gradient":
        gate.requires_grad = True
    elif case == "alignment":
        gate.address = 258
    actual = activation.sigmoid_and_mul(gate, up)
    assert torch.equal(actual, up.value * gate.value.sigmoid())
    assert events == (
        ["covered", "enter", "run", "exit"]
        if case == "native"
        else ["covered"]
        if case == "layout"
        else []
    )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "missing",
    ["sglang", "sglang.kernels.ops.attention.mla_output_gate", "broken_internal_dependency"],
)
def test_optional_SDK_resolution_does_not_hide_internal_import_failure(monkeypatch, missing):
    import importlib

    def missing_module(name):
        raise ModuleNotFoundError(name=missing)

    activation._sigmoid_mul_kernel.cache_clear()
    monkeypatch.setattr(importlib, "import_module", missing_module)
    try:
        if missing == "broken_internal_dependency":
            with pytest.raises(ModuleNotFoundError):
                activation._sigmoid_mul_kernel()
        else:
            assert activation._sigmoid_mul_kernel() is None
    finally:
        activation._sigmoid_mul_kernel.cache_clear()


@pytest.mark.parametrize(
    "numel,expected_native",
    [(2**32 - 8, True), (2**32, False), (2**32 + 8, False), (2**33 + 8, False)],
)
def test_native_uint32_numel_bound_before_resolving_kernel(monkeypatch, numel, expected_native):
    class NumelView(DeviceView):
        def numel(self):
            return numel

    gate = NumelView(torch.linspace(-2, 2, 8, dtype=torch.bfloat16))
    up = NumelView(torch.full((8,), 1.1015625, dtype=torch.bfloat16))
    events = []

    def covered(actual_up, actual_gate):
        assert actual_up is up and actual_gate is gate
        events.append("covered")
        return True

    @contextmanager
    def stream():
        events.append("enter")
        yield
        events.append("exit")

    def native(actual_up, actual_gate):
        events.append("native")
        return actual_up.value * actual_gate.value.sigmoid()

    def resolve():
        events.append("resolve")
        return covered, native, stream

    monkeypatch.setattr(activation, "_sigmoid_mul_kernel", resolve)
    actual = activation.sigmoid_and_mul(gate, up)
    assert torch.equal(actual, up.value * gate.value.sigmoid())
    assert events == (["resolve", "covered", "enter", "native", "exit"] if expected_native else [])
    assert not torch.cuda.is_initialized()
