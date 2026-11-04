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
@pytest.mark.parametrize("route_dtype", [torch.float32, torch.bfloat16])
def test_native_quantization_and_bf16_output(
    experts, native_runner, monkeypatch, tokens, dtype, strided, route_dtype
):
    import flashinfer

    torch.manual_seed(272 + tokens)
    x, weights, ids = inputs(tokens, dtype, strided)
    args = x, weights.to(route_dtype), ids
    expected = native_runner(*args)
    destinations = []
    kernel = flashinfer.trtllm_fp4_block_scale_routed_moe

    def observe(*a, **kw):
        destinations.append(kw["output"])
        assert kw["topk_ids"][1].dtype == route_dtype
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


def fused_inputs(tokens, dtype, strided):
    from minisgl.moe.routing import GroupedRouting

    x, _, _ = inputs(tokens, dtype, strided)
    logits = torch.randn(tokens, 896, device="cuda", dtype=torch.float32) * 0.1 - 10
    # Cover local/remote experts with nontrivial BF16 routing-weight rounding.
    logits[:, 32:48] = torch.linspace(-1.0, 2.5, 16, device="cuda")
    bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.01
    return x, GroupedRouting(logits, bias, 16, 1, 1, "sigmoid", True, 2.5)


def fused_native_reference(x, routing, native_runner):
    from sglang.srt.layers.moe import route_quant_handoff as handoff
    from tvm_ffi import use_torch_stream

    # Use the pinned runtime's actual producer/consumer handoff as a reference.
    # Mini passes prepared inputs explicitly and has no global handoff.
    handoff.stage(x)
    try:
        with use_torch_stream():
            selected = handoff.try_route_quant_fused(
                routing.logits,
                routing.bias,
                routing.topk,
                num_fused_shared_experts=0,
                renormalize=routing.renormalize,
                routed_scaling_factor=routing.scale,
                apply_routed_scaling_factor_on_output=routing.scale != 1.0,
            )
        assert selected is not None, "Native fused kernel must cover this qualification"
        return native_runner(x, *selected)
    finally:
        handoff.clear()


@pytest.mark.parametrize("tokens", [1, 7, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_native_fused_preparation_and_experts(
    experts, native_runner, monkeypatch, tokens, dtype, strided
):
    import flashinfer
    from minisgl.moe.fused import FusedMoe

    torch.manual_seed(740 + tokens)
    x, routing = fused_inputs(tokens, dtype, strided)
    expected = fused_native_reference(x, routing, native_runner)
    calls = []
    kernel = flashinfer.trtllm_fp4_block_scale_routed_moe

    def observe(*a, **kw):
        calls.append(kw)
        assert isinstance(kw["topk_ids"], torch.Tensor)
        assert kw["topk_ids"].dtype == torch.int32
        assert kw["hidden_states"].dtype == torch.float8_e4m3fn
        return kernel(*a, **kw)

    monkeypatch.setattr(flashinfer, "trtllm_fp4_block_scale_routed_moe", observe)
    actual = FusedMoe().forward_experts(experts, x, routing)
    assert len(calls) == 1 and actual is calls[0]["output"]
    assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 7, 48])
def test_fused_graph_replay_reads_current_scores_and_activations(experts, native_runner, tokens):
    from minisgl.moe.fused import FusedMoe

    torch.manual_seed(840 + tokens)
    x, routing = fused_inputs(tokens, torch.float32, True)
    backend = FusedMoe()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            backend.forward_experts(experts, x, routing)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backend.forward_experts(experts, x, routing)
    for step in range(3):
        x.copy_(inputs(tokens, torch.float32, False)[0])
        routing.logits.normal_(mean=-10, std=0.1)
        routing.logits[:, (32 if step % 2 else 64) : (48 if step % 2 else 80)] = torch.linspace(
            -1.0, 2.5, 16, device="cuda"
        )
        expected = fused_native_reference(x, routing, native_runner)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if step % 2 == 0:
            assert actual.eq(0).all()
    torch.cuda.synchronize()
    graph.reset()


@pytest.mark.parametrize("tokens", [1, 48])
def test_fused_preparation_matches_native_unfused_operators(tokens):
    from minisgl.moe.fused import FusedMoe
    from minisgl.moe.sgl import fused_route_quant
    from sglang.kernels.ops.quantization.per_token_group_quant import per_token_group_quant
    from tvm_ffi import use_torch_stream

    x, routing = fused_inputs(tokens, torch.float32, True)
    prepared = fused_route_quant(x, routing)
    assert prepared is not None
    weights, ids, packed, quantized, scales = prepared
    expected_weights, expected_ids = routing.select(FusedMoe())
    with use_torch_stream():
        expected_quantized, expected_scales = per_token_group_quant(
            x, group_size=32, scale_ue8m0=True
        )
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    assert torch.equal(packed >> 16, ids)
    rounded = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    assert torch.equal(rounded, weights.bfloat16())
    assert not torch.equal(rounded.float(), weights), "Fixture must exercise route rounding"
