"""GPUq checks for SiTU arithmetic, storage precision, and layer dispatch."""

import os

import pytest
import torch
from minisgl.core import Context
from minisgl.layers.activation import situ_and_mul

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1",
    reason="CUDA work must be explicitly queued",
)


def precision(monkeypatch, strict):
    monkeypatch.setattr(
        "minisgl.core._GLOBAL_CTX", Context(1, bf16_precision="strict" if strict else "default")
    )


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("linear_beta", [None, 25.0])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("tokens,channels", [(1, 7), (2, 32), (17, 127)])
def test_situ_dtype_strides_and_chunk_independence(
    monkeypatch, input_dtype, output_dtype, linear_beta, strided, tokens, channels
):
    precision(monkeypatch, True)
    torch.manual_seed(1151)
    storage = torch.randn(tokens * 2, channels * 4, device="cuda", dtype=input_dtype) * 4
    gate = storage[::2, : channels * 2 : 2]
    up = storage[::2, channels * 2 :: 2]
    if not strided:
        gate, up = gate.contiguous(), up.contiguous()
    original = storage.clone()
    actual = situ_and_mul(gate, up, 4.0, linear_beta, dtype=output_dtype)
    assert actual.shape == gate.shape and actual.dtype == output_dtype
    expected_gate = 4.0 * torch.tanh(gate.double() / 4.0) * gate.double().sigmoid()
    expected_up = up.double()
    if linear_beta is not None:
        expected_up = linear_beta * torch.tanh(expected_up / linear_beta)
    precise = expected_gate * expected_up
    # Account for FP32 operators and the final output-storage rounding.
    roundoff = 8 * torch.finfo(torch.float32).eps + torch.finfo(output_dtype).eps
    bound = roundoff * (precise.abs() + 1)
    assert ((actual.double() - precise).abs() <= bound).all()
    chunks = [
        situ_and_mul(gate[i : i + 1], up[i : i + 1], 4.0, linear_beta, dtype=output_dtype)
        for i in range(tokens)
    ]
    torch.testing.assert_close(torch.cat(chunks), actual, rtol=0, atol=0)
    torch.testing.assert_close(storage, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_situ_exact_zero_and_softcap_limits(monkeypatch, dtype):
    precision(monkeypatch, True)
    gate = torch.tensor([[-10000, 0, 10000, 10000]], device="cuda", dtype=dtype)
    up = torch.tensor([[10000, 10000, 10000, -10000]], device="cuda", dtype=dtype)
    expected = torch.tensor([[0, 0, 100, -100]], device="cuda", dtype=dtype)
    torch.testing.assert_close(situ_and_mul(gate, up, 4.0, 25.0), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_situ_graph_reads_current_inputs(monkeypatch, dtype, request):
    precision(monkeypatch, True)
    gate = torch.randn(3, 128, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    for _ in range(3):
        situ_and_mul(gate, up, 4.0, 25.0, dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = situ_and_mul(gate, up, 4.0, 25.0, dtype=torch.bfloat16)
    old = actual.clone()
    gate.add_(3)
    up.mul_(2)
    graph.replay()
    expected = situ_and_mul(gate, up, 4.0, 25.0, dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, old)
    graph.reset()
    request.node.user_properties.append(("graph_replays", 1))


@pytest.mark.parametrize("strict", [False, True])
def test_situ_fp32_output_preserves_eager_arithmetic(monkeypatch, strict):
    precision(monkeypatch, strict)
    gate = torch.randn(7, 128, device="cuda", dtype=torch.float32)
    up = torch.randn_like(gate)
    expected = (4.0 * torch.tanh(gate / 4.0) * gate.sigmoid()) * (25.0 * torch.tanh(up / 25.0))
    torch.testing.assert_close(situ_and_mul(gate, up, 4.0, 25.0), expected, rtol=0, atol=0)


def test_situ_default_policy_uses_the_native_operator(monkeypatch):
    from sglang.kernels.ops.activation import situ_and_mul as native
    from tvm_ffi import use_torch_stream

    precision(monkeypatch, False)
    gate = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    with use_torch_stream():
        expected = native(torch.cat((gate, up), -1), torch.empty_like(gate), 4.0, 25.0)
    torch.testing.assert_close(situ_and_mul(gate, up, 4.0, 25.0), expected, rtol=0, atol=0)
