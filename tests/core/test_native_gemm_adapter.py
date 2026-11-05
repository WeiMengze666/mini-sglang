"""Host ABI of optional native GEMM dispatch; no GPU arithmetic acceptance."""

from types import SimpleNamespace

import minisgl.layers.gemm as gemm
import pytest
import torch


class TensorSpec:
    def __init__(self, shape, *, dtype=torch.bfloat16, device="cuda:0", offset=0, contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, torch.device(device)
        self.ndim = len(shape)
        self.is_cuda = self.device.type == "cuda"
        self.offset, self.contiguous = offset, contiguous

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return 4096 + self.offset


@pytest.fixture
def native(monkeypatch):
    calls = []
    result = object()
    monkeypatch.setattr(gemm, "use_strict_bf16", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))

    def ordinary(*args):
        calls.append(("gemm", args))
        return result

    def buffered(*args):
        calls.append(("out", args))
        return args[2]

    def covered(*shape):
        calls.append(("covered", shape))
        return True

    monkeypatch.setattr(gemm, "_native_bf16_kernels", lambda: (ordinary, buffered, covered))
    return SimpleNamespace(calls=calls, result=result)


@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
def test_native_inputs_and_caller_owned_accumulator_are_preserved(native, dtype, bias):
    x, weight = TensorSpec((8, 7168)), TensorSpec((1536, 7168))
    b = TensorSpec((1536,)) if bias else None
    out = TensorSpec((8, 1536), dtype=dtype) if dtype is not None else None
    result = gemm.try_native_bf16_gemm(x, weight, b, out=out)
    assert native.calls[0] == ("covered", (8, 1536, 7168))
    assert result is (native.result if out is None else out)
    assert native.calls[-1][1] == ((x, weight, b) if out is None else (x, weight, out, b))


@pytest.mark.parametrize(
    "invalid", ["cpu", "fp16", "strided", "offset", "weight_offset", "inner", "bias", "out"]
)
def test_unsupported_layout_or_dtype_uses_existing_fallback(native, invalid):
    x = TensorSpec(
        (8, 7168),
        device="cpu" if invalid == "cpu" else "cuda:0",
        dtype=torch.float16 if invalid == "fp16" else torch.bfloat16,
        contiguous=invalid != "strided",
        offset=2 if invalid == "offset" else 0,
    )
    weight = TensorSpec(
        (1536, 7160 if invalid == "inner" else 7168), offset=2 if invalid == "weight_offset" else 0
    )
    bias = TensorSpec((1536,), dtype=torch.float32) if invalid == "bias" else None
    out = TensorSpec((8, 1536), dtype=torch.float16) if invalid == "out" else None
    assert gemm.try_native_bf16_gemm(x, weight, bias, out=out) is None
    assert not native.calls


def test_strict_policy_and_unsupported_hardware_do_not_load_native_sdk(native, monkeypatch):
    x, weight = TensorSpec((8, 7168)), TensorSpec((1536, 7168))
    monkeypatch.setattr(gemm, "use_strict_bf16", lambda: True)
    assert gemm.try_native_bf16_gemm(x, weight) is None
    monkeypatch.setattr(gemm, "use_strict_bf16", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert gemm.try_native_bf16_gemm(x, weight) is None
    assert not native.calls


def test_missing_optional_sdk_uses_existing_fallback(native, monkeypatch):
    monkeypatch.setattr(gemm, "_native_bf16_kernels", lambda: None)
    assert gemm.try_native_bf16_gemm(TensorSpec((8, 7168)), TensorSpec((1536, 7168))) is None
    assert not native.calls
