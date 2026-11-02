"""Queued checks for exact causal-convolution decode and request state ownership."""

import os

import pytest
import torch
from minisgl.attention.kda import causal_conv1d

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1",
    reason="CUDA work must be explicitly queued",
)


def reference(x, weight, state):
    history = torch.cat((state, x.transpose(-2, -1)), dim=-1)
    products = history.float().unfold(-1, 4, 1) * weight.float()[:, None, :]
    output = torch.nn.functional.silu(products.sum(-1).transpose(-2, -1)).to(x.dtype)
    return output, history[..., -3:].clone()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels,batch", [(1, 1), (7, 2), (128, 1), (384, 2), (1536, 8)])
@pytest.mark.parametrize("strided", [False, True])
def test_decode_matches_existing_arithmetic_through_continuation(dtype, channels, batch, strided):
    torch.manual_seed(1049)
    device = "cuda"
    weight = torch.randn(channels, 4, device=device, dtype=torch.float32)
    storage = torch.randn(batch, 3, channels, 3, device=device, dtype=dtype)
    actual = storage[:, 1] if strided else storage[:, 1].contiguous()
    expected = actual.clone()
    original = actual.clone()
    for step in range(16):
        backing = torch.randn(batch, 2, channels, device=device, dtype=dtype)
        x = backing[:, :1] if strided else backing[:, :1].contiguous()
        y, final = causal_conv1d(x, weight, actual)
        ref, expected = reference(x, weight, expected)
        torch.testing.assert_close(y, ref, rtol=0, atol=0)
        torch.testing.assert_close(final, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        actual = final
        original = final.clone()
    assert actual.data_ptr() != storage.data_ptr()


def test_decode_cuda_graph_reads_current_input_without_mutating_history():
    torch.manual_seed(1051)
    x = torch.randn(2, 1, 127, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(127, 4, device="cuda", dtype=torch.float32)
    state = torch.randn(2, 127, 3, device="cuda", dtype=torch.bfloat16)
    saved = state.clone()
    for _ in range(3):
        causal_conv1d(x, weight, state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, history = causal_conv1d(x, weight, state)
    x.normal_()
    graph.replay()
    ref, expected = reference(x, weight, state)
    torch.testing.assert_close(output, ref, rtol=0, atol=0)
    torch.testing.assert_close(history, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, saved, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batched", [False, True])
def test_decode_starts_from_zero_history(dtype, batched):
    torch.manual_seed(1061)
    shape = (2, 1, 63) if batched else (1, 63)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(63, 1, 4, device="cuda", dtype=torch.float32)
    zeros = x.new_zeros((*x.shape[:-2], 63, 3))
    output, history = causal_conv1d(x, weight, None)
    expected, tail = reference(x, weight[:, 0], zeros)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(history, tail, rtol=0, atol=0)
