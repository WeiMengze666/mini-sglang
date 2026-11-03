"""Queued SDK controls for shared primitives, ownership and graph replay."""

import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.activation import situ_and_mul
from minisgl.layers.attn_residual import attention_residual
from minisgl.layers.norm_kernel import rms_norm

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


def check_graph(run, control, change, request):
    for _ in range(2):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        change()
        expected = control()
        graph.replay()
        torch.cuda.synchronize()
        for result, ref in zip(
            captured if isinstance(captured, tuple) else (captured,),
            expected if isinstance(expected, tuple) else (expected,),
        ):
            torch.testing.assert_close(result, ref, rtol=0, atol=0)
    graph.reset()
    request.node.user_properties.append(("graph_replays", 3))


@pytest.mark.parametrize("kind,dim", [("rms", 7168), ("hf", 512), ("gated", 128)])
@pytest.mark.parametrize("tokens", [1, 27, 65])
def test_norm_sdk_current_stream_and_borrowed_input(kind, dim, tokens, request):
    from sglang.kernels.ops.attention.fla.kda import rms_norm_gated
    from sglang.kernels.ops.layernorm import rmsnorm
    from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf
    from tvm_ffi import use_torch_stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        shape = (tokens, 12, dim) if kind == "gated" else (tokens, dim)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        weight = torch.randn(dim, device="cuda", dtype=x.dtype)
        saved = x.clone()

        def run():
            return rms_norm(
                x,
                weight,
                1e-6,
                gate=gate if kind == "gated" else None,
                round_before_weight=kind == "hf",
            )

        def control():
            with use_torch_stream():
                flat = x.reshape(-1, dim)
                result = (
                    rms_norm_gated(
                        flat.clone(),
                        gate.reshape_as(flat),
                        weight,
                        None,
                        eps=1e-6,
                        activation="sigmoid",
                    )
                    if kind == "gated"
                    else rmsnorm_hf(flat, weight, 1e-6)
                    if kind == "hf"
                    else rmsnorm(flat, weight, 1e-6)
                )
                return result.reshape_as(x)

        torch.testing.assert_close(run(), control(), rtol=0, atol=0)
        torch.testing.assert_close(x, saved, rtol=0, atol=0)
        check_graph(run, control, x.neg_, request)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 27, 65])
def test_situ_adjacent_projection_sdk(dtype, tokens, request):
    from sglang.kernels.ops.activation import situ_and_mul as native
    from tvm_ffi import use_torch_stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packed = torch.randn(tokens, 6144, device="cuda", dtype=dtype)
        gate, up = packed.chunk(2, -1)
        saved = packed.clone()

        def run():
            return situ_and_mul(gate, up, 4.0, 25.0, dtype=torch.bfloat16)

        def control():
            with use_torch_stream():
                output = torch.empty_like(gate, dtype=torch.bfloat16)
                return native(packed, output, 4.0, 25.0)

        torch.testing.assert_close(run(), control(), rtol=0, atol=0)
        torch.testing.assert_close(packed, saved, rtol=0, atol=0)
        check_graph(run, control, packed.neg_, request)
    torch.cuda.current_stream().wait_stream(stream)


class Norm:
    def __init__(self, dim):
        self.weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        self.eps = self.variance_epsilon = 1e-6

    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)

    forward = __call__


@pytest.mark.parametrize("tokens", [1, 27, 65])
@pytest.mark.parametrize("blocks", [1, 3, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_residual_native_portfolio_graphs_and_borrowed_bank(tokens, blocks, normalize, request):
    from sglang.srt.layers.attn_residual import _aggregate, aggregate_stream
    from tvm_ffi import use_torch_stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        head = torch.randn(tokens, 7168, device="cuda", dtype=torch.bfloat16)
        bank = [torch.randn_like(head) for _ in range(blocks)]
        saved = [head.clone(), *[b.clone() for b in bank]]
        norm, output_norm = Norm(7168), Norm(7168) if normalize else None
        proj = SimpleNamespace(weight=torch.randn(1, 7168, device="cuda", dtype=head.dtype))
        score = SimpleNamespace(weight=proj.weight)

        def run():
            return attention_residual(head, bank, norm, proj, output_norm)

        def control():
            with use_torch_stream():
                rows = torch.stack([*bank, head], dim=1)
                return (
                    _aggregate(head, rows, blocks, score, norm, output_norm)
                    if normalize
                    else aggregate_stream(head, rows, blocks, score, norm)
                )

        torch.testing.assert_close(run(), control(), rtol=0, atol=0)
        for value, before in zip([head, *bank], saved):
            torch.testing.assert_close(value, before, rtol=0, atol=0)
        check_graph(run, control, head.neg_, request)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("channels", [3072, 12288])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,is_decode", [(1, False), (1, True), (17, False), (129, False)])
@pytest.mark.parametrize("strided", [False, True])
def test_causal_convolution_native_stages_and_graphs(
    tokens, is_decode, strided, weight_dtype, channels, request
):
    from minisgl.attention.kda.causal_conv import causal_conv1d_update
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update as native_update

    batch = 3
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.randn(batch, tokens * 2, channels, device="cuda", dtype=torch.bfloat16)
        x = source[:, ::2] if strided else source[:, ::2].contiguous()
        backing = torch.randn(batch, channels, 6, device="cuda", dtype=x.dtype)
        history = backing[..., ::2] if strided else backing[..., ::2].contiguous()
        original = backing.clone(), history.clone()
        weight = torch.randn(channels, 4, device="cuda", dtype=weight_dtype)
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * tokens
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        has_state = torch.ones(batch, device="cuda", dtype=torch.bool)

        def run():
            return causal_conv1d_update(x, weight, history, is_decode=is_decode)

        def control():
            pool = (
                history.transpose(1, 2).clone(memory_format=torch.contiguous_format).transpose(1, 2)
            )
            if is_decode:
                result = native_update(
                    x.reshape(batch, channels), pool, weight, activation="silu"
                ).reshape(batch, 1, channels)
            else:
                result = causal_conv1d_fn(
                    x.reshape(batch * tokens, channels).T,
                    weight,
                    None,
                    conv_states=pool,
                    query_start_loc=starts,
                    seq_lens_cpu=[tokens] * batch,
                    cache_indices=indices,
                    has_initial_state=has_state,
                    activation="silu",
                ).T.reshape(batch, tokens, channels)
            return result, pool

        result, final = run()
        expected, state = control()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(final, state, rtol=0, atol=0)
        torch.testing.assert_close(
            final, torch.cat((history, x.transpose(-1, -2)), -1)[..., -3:], rtol=0, atol=0
        )
        torch.testing.assert_close(backing, original[0], rtol=0, atol=0)
        torch.testing.assert_close(history, original[1], rtol=0, atol=0)
        check_graph(run, control, source.neg_, request)
        torch.testing.assert_close(history, original[1], rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("normalize", [False, True])
def test_residual_strided_head_sdk_graphs(normalize, request):
    from sglang.srt.layers.attn_residual import _aggregate, aggregate_stream
    from tvm_ffi import use_torch_stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        storage = torch.randn(27, 2, 7168, device="cuda", dtype=torch.bfloat16)
        head, bank = storage[:, 1], [storage[:, 0]]
        saved_bank = bank[0].clone()
        norm, output_norm = Norm(7168), Norm(7168) if normalize else None
        proj = SimpleNamespace(weight=torch.randn(1, 7168, device="cuda", dtype=head.dtype))

        def run():
            return attention_residual(head, bank, norm, proj, output_norm)

        def control():
            with use_torch_stream():
                rows = torch.stack([*bank, head], dim=1)
                return (
                    _aggregate(head.contiguous(), rows, 1, proj, norm, output_norm)
                    if normalize
                    else aggregate_stream(head.contiguous(), rows, 1, proj, norm)
                )

        torch.testing.assert_close(run(), control(), rtol=0, atol=0)
        check_graph(run, control, head.neg_, request)
        torch.testing.assert_close(bank[0], saved_bank, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)
