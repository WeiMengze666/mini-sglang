"""Host ABI fixtures: preserve caller routing precision without touching CUDA."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.moe.flashinfer import FlashInferMXFP4Experts


class DeviceInput(torch.Tensor):
    # Only bypass the executor's device guard; storage and all operations stay CPU.
    @property
    def is_cuda(self):
        return True


@pytest.fixture
def executor(monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("This host ABI test must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "init", no_cuda)
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        kwargs["output"].zero_()

    modules = {
        name: ModuleType(name)
        for name in (
            "flashinfer",
            "flashinfer.fused_moe",
            "flashinfer.fused_moe.core",
            "flashinfer.tllm_enums",
        )
    }
    modules["flashinfer"].trtllm_fp4_block_scale_routed_moe = run
    modules["flashinfer.fused_moe.core"].ActivationType = SimpleNamespace(
        Swiglu=SimpleNamespace(value=1), Situ=SimpleNamespace(value=2)
    )
    modules["flashinfer.tllm_enums"].RoutingMethodType = SimpleNamespace(
        TopK=SimpleNamespace(value=0)
    )
    modules["flashinfer.tllm_enums"].SfLayout = SimpleNamespace(layout_linear=0)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    obj = object.__new__(FlashInferMXFP4Experts)
    obj._hidden, obj._intermediate = 256, 128
    obj._precision, obj._activation = "bf16", "silu"
    obj._global_experts, obj._experts, obj._start = 8, 2, 0
    for name in ("_w13", "_w13_scale", "_w2", "_w2_scale", "_alpha", "_linear"):
        setattr(obj, name, None)
    return obj, calls


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_route_dtype_reaches_sdk_unchanged(executor, dtype, strided):
    obj, calls = executor
    x = torch.zeros(3, 256, dtype=torch.bfloat16).as_subclass(DeviceInput)
    ids = torch.arange(6, dtype=torch.int64).reshape(3, 2)
    columns = 4 if strided else 2
    weights = torch.linspace(0.1, 0.9, 3 * columns, dtype=dtype).reshape(3, columns)[:, :2]
    result = obj.forward(x, weights, ids)
    routed_ids, routed_weights = calls[0]["topk_ids"]
    assert routed_weights.dtype == dtype
    assert routed_weights.is_contiguous()
    torch.testing.assert_close(routed_weights, weights, rtol=0, atol=0)
    if not strided:
        assert routed_weights.data_ptr() == weights.data_ptr()
    assert routed_ids.dtype == torch.int32
    torch.testing.assert_close(routed_ids, ids.to(torch.int32), rtol=0, atol=0)
    assert result is calls[0]["output"]
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_unsupported_route_dtype_is_rejected_before_sdk(executor, dtype):
    obj, calls = executor
    x = torch.zeros(3, 256, dtype=torch.bfloat16).as_subclass(DeviceInput)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="route weights must be bfloat16 or float32"):
        obj.forward(x, torch.ones(3, 2, dtype=dtype), ids)
    assert calls == []
