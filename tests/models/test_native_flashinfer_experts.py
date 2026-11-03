"""Exercise the adapter against the pinned native runner, not another FP4 implementation."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


@pytest.fixture(scope="module")
def experts():
    from minisgl.moe.flashinfer import FlashInferMXFP4Experts
    from test_mxfp4_flashinfer import _expert

    torch.manual_seed(271)
    return FlashInferMXFP4Experts(
        [(32 + i, _expert(hidden=3584, intermediate=384)) for i in range(4)],
        activation="situ",
        beta=4.0,
        linear_beta=25.0,
        num_global_experts=896,
        activation_precision="mxfp8",
    )


@pytest.fixture
def native_runner(experts, monkeypatch):
    # Load the native quantization registry before its runner, as the runtime does.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as native
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    # A component test has no server process group. Only allocation ownership is
    # stubbed; native quantization, routing preparation and GEMM launch stay intact.
    monkeypatch.setattr(native, "get_parallel", lambda: SimpleNamespace(tp_group=None))
    monkeypatch.setattr(native, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(native, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    quant = native.FlashInferTrtllmGenMxfp4MoeQuantInfo(
        w13_weight=experts._w13,
        w2_weight=experts._w2,
        w13_weight_scale=experts._w13_scale,
        w2_weight_scale=experts._w2_scale,
        w13_weight_bias=None,
        w2_weight_bias=None,
        gemm1_alpha=experts._alpha,
        gemm1_beta=None,
        gemm1_clamp_limit=experts._linear,
        global_num_experts=896,
        local_expert_offset=32,
        local_num_experts=4,
        intermediate_size_per_partition=384,
        hidden_size=3584,
        flashinfer_mxfp4_moe_precision="default",
    )
    config = MoeRunnerConfig(activation="situ", top_k=16)

    def run(x, weights, ids):
        dispatch = StandardDispatchOutput(x, None, StandardTopKOutput(weights, ids, None))
        return native._fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch, quant, config
        ).hidden_states

    return run


def inputs(tokens, dtype, strided):
    columns = 3584 + (128 if strided else 0)
    x = (torch.randn(tokens, columns, device="cuda", dtype=dtype) * 0.05)[:, :3584]
    ids = torch.arange(16, device="cuda", dtype=torch.int32).expand(tokens, -1) + 32
    weights = torch.randn(tokens, 16, device="cuda").softmax(-1)
    return x, weights, ids


@pytest.mark.parametrize("tokens", [1, 7, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_native_quantization_and_bf16_output(
    experts, native_runner, monkeypatch, tokens, dtype, strided
):
    import flashinfer

    torch.manual_seed(272 + tokens)
    args = inputs(tokens, dtype, strided)
    expected = native_runner(*args)
    destinations = []
    kernel = flashinfer.trtllm_fp4_block_scale_routed_moe

    def observe(*a, **kw):
        destinations.append(kw["output"])
        return kernel(*a, **kw)

    monkeypatch.setattr(flashinfer, "trtllm_fp4_block_scale_routed_moe", observe)
    actual = experts.forward(*args)
    assert actual is destinations[-1]
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 7, 48])
def test_native_graph_replay_reads_current_routes(experts, native_runner, tokens):
    torch.manual_seed(320 + tokens)
    x, weights, ids = inputs(tokens, torch.float32, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            experts.forward(x, weights, ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = experts.forward(x, weights, ids)
    for step in range(3):
        x.copy_(inputs(tokens, torch.float32, False)[0])
        weights.copy_(torch.randn_like(weights).softmax(-1))
        # Move local selections to the nonlocal range and back. This also checks
        # that zero contribution is written rather than reusing stale output.
        ids.copy_(torch.arange(16, device="cuda", dtype=torch.int32) + (32 if step % 2 else 64))
        expected = native_runner(x, weights, ids)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if step % 2 == 0:
            assert actual.eq(0).all()
