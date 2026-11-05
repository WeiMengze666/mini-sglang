"""Native dense adapter agrees with its SDK and an independent FP64 error bound."""

import os

import pytest
import torch
from minisgl.core import Context
from minisgl.layers.gemm import try_native_bf16_gemm

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="explicit GPU gate"
)


@pytest.mark.parametrize("batch", [1, 8, 48])
@pytest.mark.parametrize(
    "output,input_size",
    [
        (1536, 7168),
        (7168, 1536),
        (3584, 7168),
        (7168, 3584),
        (128, 7168),
        (7168, 128),
        (128, 1536),
        (1536, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias", [False, True])
def test_native_gemm_precision_stream_and_graph(
    monkeypatch, batch, output, input_size, dtype, has_bias, request
):
    import minisgl.core as core
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
        cutedsl_bf16_gemm,
        cutedsl_bf16_gemm_out,
        use_cutedsl_bf16_gemm,
    )

    ctx = Context(1)
    ctx.bf16_precision = "default"
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    torch.manual_seed(batch + input_size + output)
    x = torch.randn(batch, input_size, device="cuda", dtype=torch.bfloat16) * 0.03
    weight = torch.randn(output, input_size, device="cuda", dtype=torch.bfloat16) * 0.03
    bias = torch.randn(output, device="cuda", dtype=torch.bfloat16) * 0.03 if has_bias else None
    buffer = (
        torch.empty(batch, output, device="cuda", dtype=dtype) if dtype == torch.float32 else None
    )
    saved = x.clone(), weight.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        actual = try_native_bf16_gemm(x, weight, bias, out=buffer)
        covered = use_cutedsl_bf16_gemm(batch, output, input_size)
        request.node.user_properties.append(("native_covered", covered))
        assert (actual is not None) == covered
        if not covered:
            return
        expected = (
            cutedsl_bf16_gemm(x, weight, bias)
            if buffer is None
            else cutedsl_bf16_gemm_out(x, weight, torch.empty_like(buffer), bias)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reference = x.double() @ weight.double().T
        if bias is not None:
            reference += bias.double()
        magnitude = x.double().abs() @ weight.double().abs().T
        u = torch.finfo(torch.float32).eps / 2
        gamma = input_size * u / (1 - input_size * u)
        bound = (
            2 * gamma * magnitude
            + torch.finfo(dtype).eps * reference.abs()
            + torch.finfo(dtype).tiny
        )
        assert ((actual.double() - reference).abs() <= bound).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = try_native_bf16_gemm(x, weight, bias, out=buffer)
        pointer = captured.data_ptr()
        for _ in range(3):
            graph.replay()
            assert captured.data_ptr() == pointer
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        graph.reset()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, saved[0], rtol=0, atol=0)
    torch.testing.assert_close(weight, saved[1], rtol=0, atol=0)
