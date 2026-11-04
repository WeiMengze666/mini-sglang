"""Native combine versus unpartitioned attention and dynamic graph inputs."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="opt-in CUDA gate"
)


@pytest.mark.parametrize("shards", [2, 8])
@pytest.mark.parametrize("batch", [1, 48])
@pytest.mark.parametrize("dimension", [64, 512])
def test_native_combine_matches_unpartitioned_attention(shards, batch, dimension):
    from minisgl.distributed.context import merge_attention

    torch.manual_seed(2500 + shards + batch + dimension)
    q = torch.randn(batch, 3, 16, device="cuda", dtype=torch.float64) * 0.1
    k = torch.randn(17, 3, 16, device="cuda", dtype=torch.float64) * 0.1
    v = torch.randn(17, 3, dimension, device="cuda", dtype=torch.float64) * 0.1
    scores = torch.einsum("bhd,thd->bht", q, k) / 4
    expected = torch.einsum("bht,thv->bhv", scores.softmax(-1), v)
    partials, statistics = [], []
    for rank in range(shards):
        local = scores[..., rank::shards]
        partials.append(torch.einsum("bht,thv->bhv", local.softmax(-1), v[rank::shards]))
        statistics.append(local.logsumexp(-1))
    output, lse = torch.stack(partials).float(), torch.stack(statistics).float()
    actual = merge_attention(output, lse)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("all_empty", [False, True])
def test_native_combine_zeros_undefined_empty_shards(all_empty):
    from minisgl.distributed.context import merge_attention

    output = torch.full((8, 1, 3, 512), torch.nan, device="cuda")
    lse = torch.full((8, 1, 3), -torch.inf, device="cuda")
    expected = torch.zeros_like(output[0])
    if not all_empty:
        output[0].normal_()
        lse[0] = 2
        expected.copy_(output[0])
    actual = merge_attention(output, lse)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert output[1:].isnan().all()


@pytest.mark.parametrize("corrupt", ["output", "nan_lse", "positive_infinite_lse"])
def test_native_combine_does_not_hide_active_corruption(corrupt):
    from minisgl.distributed.context import merge_attention

    output = torch.ones(8, 1, 1, 512, device="cuda")
    lse = torch.zeros(8, 1, 1, device="cuda")
    if corrupt == "output":
        output[0] = torch.nan
    else:
        lse[0] = torch.nan if corrupt == "nan_lse" else torch.inf
    assert merge_attention(output, lse).isnan().all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_combine_graph_tracks_strided_inputs(dtype):
    from minisgl.distributed.context import merge_attention

    output = torch.randn(8, 48, 12, 513, device="cuda", dtype=dtype)[..., :512]
    lse = torch.randn(8, 48, 12, 2, device="cuda")[..., 0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            merge_attention(output, lse)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = merge_attention(output, lse)
    for _ in range(4):
        output.normal_()
        lse.normal_()
        output[0, 0] = torch.nan
        lse[0, 0] = -torch.inf
        weights = lse.double().softmax(0)
        clean = output.double().masked_fill(torch.isneginf(lse)[..., None], 0)
        expected = (clean * weights[..., None]).sum(0)
        graph.replay()
        torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.cuda.synchronize()
    graph.reset()
