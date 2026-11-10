"""Persistent BF16 state must follow the same update/rounding as the native KDA pool."""

import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


def values(tokens, heads):
    return [
        torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)
    ] + [torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)]


def native_chunk(args, pool, indices, a, dt):
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from tvm_ffi import use_torch_stream

    q, k, v, forget, beta = args
    tokens = len(q)
    count = (tokens + 63) // 64
    chunks = torch.stack(
        (
            torch.zeros(count, device="cuda", dtype=torch.int32),
            torch.arange(count, device="cuda", dtype=torch.int32),
        ),
        dim=1,
    )
    with use_torch_stream():
        return chunk_kda(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.clone().unsqueeze(0),
            g=forget.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=pool,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
            A_log=a,
            dt_bias=dt,
            lower_bound=-5.0,
            beta_is_raw=True,
            chunk_indices=chunks,
        )[0]


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [7, 64, 143])
def test_chunk_continuation_keeps_native_state_dtype(state_dtype, tokens):
    from minisgl.attention.kda.sgl import recurrent_kda

    torch.manual_seed(410 + tokens)
    heads = 12
    pool = torch.randn(5, heads, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    indices = torch.tensor([3], device="cuda", dtype=torch.int32)
    state = pool[3].transpose(-1, -2).contiguous()
    original = pool.clone()
    a, dt = torch.randn(heads, device="cuda"), torch.randn(heads * 128, device="cuda")
    for _ in range(2):
        args = values(tokens, heads)
        saved = state.clone()
        expected = native_chunk(args, pool, indices, a, dt)
        actual, final = recurrent_kda(*args, a, dt, state, -5.0)
        assert final.dtype == state_dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(final, pool[3].transpose(-1, -2), rtol=0, atol=0)
        torch.testing.assert_close(state, saved, rtol=0, atol=0)
        state = final
    for slot in (0, 1, 2, 4):
        torch.testing.assert_close(pool[slot], original[slot], rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 8, 48])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_decode_graph_matches_indexed_native_pool(batch, state_dtype, request):
    from minisgl.attention.kda.sgl import recurrent_kda
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from tvm_ffi import use_torch_stream

    torch.manual_seed(500 + batch)
    heads = 12
    pool = torch.randn(2 * batch + 1, heads, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    indices = torch.arange(batch, device="cuda", dtype=torch.int32) * 2 + 1
    state = pool[indices.long()].transpose(-1, -2).contiguous().flatten(0, 1)
    args = values(1, batch * heads)
    a, dt = torch.randn(heads, device="cuda"), torch.randn(heads * 128, device="cuda")
    repeated = a.repeat(batch), dt.repeat(batch)
    for _ in range(3):
        recurrent_kda(*args, *repeated, state, -5.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, final = recurrent_kda(*args, *repeated, state, -5.0)
    for step in range(3):
        for part in args:
            part.mul_(1.015625)
        state.copy_(pool[indices.long()].transpose(-1, -2).contiguous().flatten(0, 1))
        saved = state.clone()
        untouched = pool[::2].clone()
        q, k, v, forget, beta = args
        # Compare graph replay exactly with the same native launch geometry.
        # The independently indexed B,H pool below also checks request isolation.
        canonical_pool = saved.transpose(-1, -2).contiguous().unsqueeze(0)
        with use_torch_stream():
            canonical_output = fused_sigmoid_gating_delta_rule_update(
                q=q.unsqueeze(0),
                k=k.unsqueeze(0),
                v=v.unsqueeze(0),
                a=forget.reshape(1, -1),
                b=beta.unsqueeze(0),
                A_log=repeated[0],
                dt_bias=repeated[1],
                initial_state_source=canonical_pool,
                initial_state_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                is_kda=True,
                lower_bound=-5.0,
            )
            expected = fused_sigmoid_gating_delta_rule_update(
                q=q.reshape(1, batch, heads, 128),
                k=k.reshape(1, batch, heads, 128),
                v=v.reshape(1, batch, heads, 128),
                a=forget.reshape(batch, -1),
                b=beta.reshape(1, batch, heads),
                A_log=a,
                dt_bias=dt,
                initial_state_source=pool,
                initial_state_indices=indices,
                cu_seqlens=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                is_kda=True,
                lower_bound=-5.0,
            ).transpose(0, 1)
        graph.replay()
        assert final.dtype == state_dtype
        canonical_state = canonical_pool[0].transpose(-1, -2)
        torch.testing.assert_close(
            actual.reshape_as(canonical_output), canonical_output, rtol=0, atol=0
        )
        torch.testing.assert_close(final, canonical_state, rtol=0, atol=0)
        indexed_state = pool[indices.long()].transpose(-1, -2).reshape_as(final)
        if state_dtype == torch.bfloat16:
            torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=0, atol=0)
            torch.testing.assert_close(final, indexed_state, rtol=0, atol=0)
        else:
            # Two official FP32-state launches can differ when B,H are factored
            # differently. Preserve this diagnostic; the adapter's identical-ABI
            # comparison above remains exact, including every state update.
            delta = canonical_output.reshape_as(expected).float() - expected.float()
            state_delta = canonical_state - indexed_state
            request.node.user_properties.append(
                (
                    "native_layout_rounding",
                    json.dumps(
                        {
                            "step": step,
                            "output_changed": int(delta.count_nonzero()),
                            "output_max_abs": float(delta.abs().max()),
                            "state_changed": int(state_delta.count_nonzero()),
                            "state_max_abs": float(state_delta.abs().max()),
                        }
                    ),
                )
            )
        torch.testing.assert_close(state, saved, rtol=0, atol=0)
        torch.testing.assert_close(pool[::2], untouched, rtol=0, atol=0)
    graph.reset()
