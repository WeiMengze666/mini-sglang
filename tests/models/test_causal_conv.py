"""Queued convolution numerics, state ownership and native stage dispatch."""

import os

import pytest
import torch
from minisgl.attention.kda import causal_conv1d
from minisgl.core import Context

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


def reference(x, weight, state):
    history = torch.cat((state, x.transpose(-2, -1)), dim=-1)
    products = history.double().unfold(-1, 4, 1) * weight.double()[:, None, :]
    precise = torch.nn.functional.silu(products.sum(-1).transpose(-2, -1))
    return precise, history[..., -3:].clone(), products.abs().sum(-1).transpose(-2, -1)


def check_numeric(actual, x, weight, state):
    precise, tail, magnitude = reference(x, weight, state)
    assert actual.dtype == x.dtype and actual.shape == x.shape
    # Native FMA/SFU operations need not share Torch's balanced reduction tree.
    # Bound FP32 evaluation and one final storage rounding independently in FP64.
    storage = torch.finfo(actual.dtype)
    unit = storage.eps / 2
    bound = 16 * torch.finfo(torch.float32).eps * (magnitude + precise.abs() + 1)
    bound += unit / (1 - unit) * precise.abs() + storage.tiny * storage.eps
    assert ((actual.double() - precise).abs() <= bound).all()
    return tail


def precision(monkeypatch, strict):
    monkeypatch.setattr(
        "minisgl.core._GLOBAL_CTX", Context(1, bf16_precision="strict" if strict else "default")
    )


@pytest.fixture(autouse=True)
def default_precision(monkeypatch):
    precision(monkeypatch, False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels,batch", [(1, 1), (7, 2), (128, 1), (384, 2), (1536, 8)])
@pytest.mark.parametrize("strided", [False, True])
def test_decode_matches_fp64_envelope_through_continuation(dtype, channels, batch, strided):
    torch.manual_seed(1049)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.float32)
    storage = torch.randn(batch, 3, channels, 3, device="cuda", dtype=dtype)
    actual = storage[:, 1] if strided else storage[:, 1].contiguous()
    expected, original = actual.clone(), actual.clone()
    for _ in range(16):
        backing = torch.randn(batch, 2, channels, device="cuda", dtype=dtype)
        x = backing[:, :1] if strided else backing[:, :1].contiguous()
        y, final = causal_conv1d(x, weight, actual, is_decode=True)
        expected = check_numeric(y, x, weight, expected)
        torch.testing.assert_close(final, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        actual, original = final, final.clone()
    assert actual.data_ptr() != storage.data_ptr()


def test_decode_cuda_graph_reads_current_input_without_mutating_history(request):
    torch.manual_seed(1051)
    x = torch.randn(2, 1, 127, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(127, 4, device="cuda", dtype=torch.float32)
    state = torch.randn(2, 127, 3, device="cuda", dtype=x.dtype)
    saved = state.clone()
    for _ in range(3):
        causal_conv1d(x, weight, state, is_decode=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, history = causal_conv1d(x, weight, state, is_decode=True)
    x.normal_()
    graph.replay()
    eager, final = causal_conv1d(x, weight, state, is_decode=True)
    torch.testing.assert_close(output, eager, rtol=0, atol=0)
    expected = check_numeric(output, x, weight, state)
    torch.testing.assert_close(history, expected, rtol=0, atol=0)
    torch.testing.assert_close(final, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, saved, rtol=0, atol=0)
    graph.reset()
    request.node.user_properties.append(("graph_replays", 1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batched", [False, True])
def test_decode_starts_from_zero_history(dtype, batched):
    torch.manual_seed(1061)
    shape = (2, 1, 63) if batched else (1, 63)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(63, 1, 4, device="cuda", dtype=torch.float32)
    zeros = x.new_zeros((*x.shape[:-2], 63, 3))
    output, history = causal_conv1d(x, weight, None, is_decode=True)
    tail = check_numeric(output, x, weight[:, 0], zeros)
    torch.testing.assert_close(history, tail, rtol=0, atol=0)


@pytest.mark.parametrize("strict", [False, True])
def test_fp32_accumulation_order_uses_the_native_operator(monkeypatch, strict):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    precision(monkeypatch, strict)
    x = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    state = torch.ones(32, 3, device="cuda", dtype=x.dtype)
    weight = torch.tensor([1.0, 2**-24, -1.0, 2**-24], device="cuda").expand(32, -1).contiguous()
    actual, tail = causal_conv1d(x, weight, state, is_decode=True)
    pool = state.T.contiguous().T.unsqueeze(0)
    expected = causal_conv1d_update(x, pool, weight, activation="silu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(tail, state, rtol=0, atol=0)
    assert state.eq(1).all()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("tokens", [1, 2, 17])
def test_chunk_append_preserves_borrowed_strided_history(
    monkeypatch, strict, dtype, batched, tokens
):
    precision(monkeypatch, strict)
    torch.manual_seed(1129)
    prefix = (3,) if batched else ()
    source = torch.randn(*prefix, tokens * 2, 32, device="cuda", dtype=dtype)
    x = source[..., ::2, :]
    backing = torch.randn(*prefix, 32, 5, device="cuda", dtype=dtype)
    state = backing[..., ::2]
    original = backing.clone()
    weight = torch.randn(32, 4, device="cuda", dtype=torch.float32)
    whole, final = causal_conv1d(x, weight, state)
    expected_tail = check_numeric(whole, x, weight, state)
    torch.testing.assert_close(final, expected_tail, rtol=0, atol=0)
    for chunk in (1, 3, 7):
        current, pieces = state, []
        for start in range(0, tokens, chunk):
            output, current = causal_conv1d(x[..., start : start + chunk, :], weight, current)
            pieces.append(output)
        torch.testing.assert_close(torch.cat(pieces, -2), whole, rtol=0, atol=0)
        torch.testing.assert_close(current, final, rtol=0, atol=0)
    torch.testing.assert_close(backing, original, rtol=0, atol=0)


def test_strict_decode_graph_uses_current_input(monkeypatch, request):
    precision(monkeypatch, True)
    x = torch.randn(3, 1, 32, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(3, 32, 3, device="cuda", dtype=x.dtype)
    saved = state.clone()
    weight = torch.randn(32, 4, device="cuda", dtype=torch.float32)
    for _ in range(3):
        causal_conv1d(x, weight, state, is_decode=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, history = causal_conv1d(x, weight, state, is_decode=True)
    x.normal_()
    graph.replay()
    expected, final = causal_conv1d(x, weight, state, is_decode=True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(history, final, rtol=0, atol=0)
    torch.testing.assert_close(state, saved, rtol=0, atol=0)
    graph.reset()
    request.node.user_properties.append(("graph_replays", 1))
