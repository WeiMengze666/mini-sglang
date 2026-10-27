"""Queued native KDA checks: raw gates, FP32 state axes and chunk continuation."""

import os

import pytest
import torch
from minisgl.attention.kda import kda_log_decay, kda_recurrence
from minisgl.attention.kda_fi import recurrent_kda

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("bound", [-5.0, None])
def test_flashinfer_kda_matches_fp32_state_and_continuation(tokens, bound):
    torch.manual_seed(103)
    q, k, v, forget = [
        torch.randn(tokens, 12, 128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)
    ]
    beta = torch.randn(tokens, 12, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(12, device="cuda") * 0.1
    dt = torch.randn(12 * 128, device="cuda") * 0.1
    # Deliberately nonsymmetric so an accidental K,V reinterpretation fails.
    initial = torch.randn(12, 128, 128, device="cuda") * 0.02
    original = initial.clone()
    decay = kda_log_decay(forget, a, dt, bound)
    expected, expected_state = kda_recurrence(q, k, v, decay, beta.float().sigmoid(), initial)
    actual, state = recurrent_kda(q, k, v, forget, beta, a, dt, initial, bound)
    assert state.dtype == torch.float32
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0.035, atol=0.001)
    torch.testing.assert_close(state, expected_state, rtol=0.035, atol=0.001)
    assert (state - expected_state).norm() / expected_state.norm() < 0.01
    split = max(1, tokens // 2)
    pieces = []
    state = initial
    for start, stop in ((0, split), (split, tokens)):
        if start == stop:
            continue
        output, state = recurrent_kda(
            q[start:stop],
            k[start:stop],
            v[start:stop],
            forget[start:stop],
            beta[start:stop],
            a,
            dt,
            state,
            bound,
        )
        pieces.append(output)
    torch.testing.assert_close(torch.cat(pieces), expected, rtol=0.035, atol=0.001)
    torch.testing.assert_close(state, expected_state, rtol=0.035, atol=0.001)
