"""GPU qualification against the actual public safe-gate operator, never old packed as oracle."""

import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued component GPU gate"
)


@pytest.mark.parametrize("batch", [1, 8, 48])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("null_row", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_safe_decode_matches_public_operator_and_reports_packed_delta(
    batch, state_dtype, null_row, graph_mode, request, monkeypatch
):
    from minisgl.attention.kda import sgl
    from minisgl.attention.kda.sgl import _safe_decode
    from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_kda_packed_decode
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from tvm_ffi import use_torch_stream

    torch.manual_seed(2700 + batch)
    heads, dim, slots = 12, 128, 2 * batch + 2
    mixed = torch.empty_strided(
        (batch, 3 * heads * dim), (3 * heads * dim + 128, 1), device="cuda", dtype=torch.bfloat16
    )
    forget = torch.empty_strided(
        (batch, heads * dim), (heads * dim + 128, 1), device="cuda", dtype=torch.bfloat16
    )
    beta = torch.empty((batch, 144), device="cuda", dtype=torch.bfloat16)[:, 128 : 128 + heads]
    for value in (mixed, forget, beta):
        value.normal_(std=0.1)
    a = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1
    dt = torch.randn(heads * dim, device="cuda", dtype=torch.float32) * 0.1
    initial = torch.empty_strided(
        (slots, heads, dim, dim),
        (heads * dim * dim + 128, dim * dim, dim, 1),
        device="cuda",
        dtype=state_dtype,
    )
    initial.normal_(std=0.02)

    def owned_state():
        value = torch.empty_strided(
            initial.shape, initial.stride(), device="cuda", dtype=state_dtype
        )
        return value.copy_(initial)

    actual_state, expected_state, packed_state = [owned_state() for _ in range(3)]
    indices = torch.arange(batch, device="cuda", dtype=torch.long) * 2
    if null_row:
        indices[-1] = -1
    qsl = torch.arange(batch + 1, device="cuda", dtype=torch.long)
    live = {int(x) for x in indices.tolist() if x >= 0}
    inactive = torch.tensor(
        [i for i in range(slots) if i not in live], device="cuda", dtype=torch.long
    )
    graph = None
    received = []

    def observed_public(**kw):
        assert kw["a"] is forget and kw["b"].data_ptr() == beta.data_ptr()
        assert kw["b"].shape == (1, batch, heads) and kw["b"].stride()[1:] == (144, 1)
        assert kw["initial_state_source"] is actual_state and kw["cu_seqlens"] is qsl
        received.append(
            {
                "parameters": {
                    name: {
                        "shape": list(kw[name].shape),
                        "stride": list(kw[name].stride()),
                        "dtype": str(kw[name].dtype),
                        "storage_offset": kw[name].storage_offset(),
                    }
                    for name in (
                        "q",
                        "k",
                        "v",
                        "a",
                        "b",
                        "A_log",
                        "dt_bias",
                        "initial_state_source",
                        "initial_state_indices",
                        "cu_seqlens",
                    )
                },
                "beta_shape": list(kw["b"].shape),
                "beta_stride": list(kw["b"].stride()),
                "beta_storage_offset": kw["b"].storage_offset(),
            }
        )
        return fused_sigmoid_gating_delta_rule_update(**kw)

    monkeypatch.setattr(sgl, "_safe_decode_kernel", lambda: observed_public)

    def adapter():
        with use_torch_stream():
            return _safe_decode(mixed, forget, beta, a, dt, actual_state, indices, qsl, -5.0)

    def public_reference():
        q, k, v = mixed.split(heads * dim, dim=-1)
        with use_torch_stream():
            return fused_sigmoid_gating_delta_rule_update(
                q=q.reshape(1, batch, heads, dim),
                k=k.reshape(1, batch, heads, dim),
                v=v.reshape(1, batch, heads, dim),
                a=forget,
                b=beta.unsqueeze(0),
                A_log=a,
                dt_bias=dt,
                initial_state_source=expected_state,
                initial_state_indices=indices,
                cu_seqlens=qsl,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                is_kda=True,
                lower_bound=-5.0,
            )

    try:
        if graph_mode:
            for _ in range(2):
                adapter()
                actual_state.copy_(initial)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = adapter()
            actual_state.copy_(initial)
        for step in range(3):
            if step:
                mixed.mul_(1.015625)
                forget.add_(0.015625)
                beta.sub_(0.015625)
            expected = public_reference()
            if graph_mode:
                graph.replay()
            else:
                actual = adapter()
            # Identical public facade ABI is the oracle; this is not a quality/perf claim.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_state.index_select(0, inactive),
                initial.index_select(0, inactive),
                rtol=0,
                atol=0,
            )
            assert (
                actual_state.dtype == state_dtype
                and torch.isfinite(actual_state).all()
                and torch.isfinite(actual).all()
            )
            legacy = mixed.new_empty((batch, 1, heads, dim))
            with use_torch_stream():
                fused_recurrent_kda_packed_decode(
                    mixed_qkv=mixed,
                    a=forget,
                    b=beta,
                    A_log=a,
                    dt_bias=dt,
                    scale=dim**-0.5,
                    initial_state=packed_state,
                    out=legacy,
                    ssm_state_indices=indices,
                    use_qk_l2norm_in_kernel=True,
                    lower_bound=-5.0,
                )
            delta = actual[0].float() - legacy[:, 0].float()
            state_delta = actual_state.float() - packed_state.float()
            request.node.user_properties.append(
                (
                    "public_vs_old_packed",
                    json.dumps(
                        {
                            "step": step,
                            "state_dtype": str(state_dtype),
                            "batch": batch,
                            "null_row": null_row,
                            "graph": graph_mode,
                            "output_changed": int(delta.count_nonzero()),
                            "output_max_abs": float(delta.abs().max()),
                            "state_changed": int(state_delta.count_nonzero()),
                            "state_max_abs": float(state_delta.abs().max()),
                            "public_finite": bool(
                                torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
                            ),
                            "public_output_max_abs_error": float((actual - expected).abs().max()),
                            "public_state_max_abs_error": float(
                                (actual_state - expected_state).abs().max()
                            ),
                            "packed_finite": bool(
                                torch.isfinite(legacy).all() and torch.isfinite(packed_state).all()
                            ),
                            "state_slot_stride": actual_state.stride(0),
                            "indices": indices.tolist(),
                            "actual_public_received": received[-1],
                            "scope": "safe core/functional FP32 supported; framework indexed_decode remains BF16-only",
                        }
                    ),
                )
            )
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()


def test_indexed_fp32_state_stays_on_existing_fallback(monkeypatch):
    from types import SimpleNamespace

    from minisgl.attention.kda import sgl

    x = torch.zeros(1, 7, device="cuda", dtype=torch.bfloat16)
    pool = SimpleNamespace(state_dtype=torch.float32, conv_dtype=torch.bfloat16)
    monkeypatch.setattr(
        sgl,
        "_safe_decode_kernel",
        lambda: pytest.fail("FP32 indexed must preserve existing fallback"),
    )
    assert sgl.indexed_decode(None, x, pool, 0, None, None) is None
