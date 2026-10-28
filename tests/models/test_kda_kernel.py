"""Queued native KDA checks: raw gates, FP32 state axes and chunk continuation."""

import os

import pytest
import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import KDAState, kda_log_decay, kda_recurrence
from minisgl.attention.kda_fi import recurrent_kda
from minisgl.layers.kda import KDALayer

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


@pytest.mark.parametrize("tokens", [1, 7, 64, 99, 143, 149, 226])
@pytest.mark.parametrize("bound", [-5.0, None])
@pytest.mark.parametrize("unaligned_beta", [False, True])
def test_flashinfer_kda_matches_fp32_state_and_continuation(tokens, bound, unaligned_beta):
    torch.manual_seed(103)
    q, k, v, forget = [
        torch.randn(tokens, 12, 128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)
    ]
    storage = torch.randn(tokens + 1, 12, device="cuda", dtype=torch.bfloat16)
    beta = storage[1:] if unaligned_beta else storage[:tokens]
    assert beta.is_contiguous()
    assert (beta.data_ptr() % 16 != 0) == unaligned_beta
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


@pytest.mark.parametrize("batch", [2, 8])
@pytest.mark.parametrize("bound", [-5.0, None])
def test_batched_native_decode_keeps_request_states_separate(batch, bound):
    torch.manual_seed(157)
    with torch.device("cuda"):
        layer = KDALayer(KDAConfig(32, 2, 128, 4, gate_lower_bound=bound))
    for tensor in layer.state_dict().values():
        tensor.normal_(std=0.1)
    projected = [torch.randn(batch, 1, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    projected += [
        torch.randn(batch, 1, 2, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch, 1, 2, device="cuda", dtype=torch.bfloat16),
    ]
    initial = KDAState(
        torch.randn(batch, 3, 256, 3, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch, 2, 128, 128, device="cuda") * 0.02,
    )
    saved = KDAState(*(part.clone() for part in initial))
    output, final = layer.update(*projected, initial, implementation="flashinfer")
    for index in range(batch):
        expected, state = layer.update(
            *(part[index] for part in projected),
            KDAState(*(part[index] for part in initial)),
        )
        torch.testing.assert_close(output[index], expected, rtol=0.035, atol=0.001)
        torch.testing.assert_close(final.conv[index], state.conv, rtol=0, atol=0)
        torch.testing.assert_close(final.recurrent[index], state.recurrent, rtol=0.035, atol=0.001)
        assert (final.recurrent[index] - state.recurrent).norm() / state.recurrent.norm() < 0.01
    for current, original in zip(initial, saved):
        torch.testing.assert_close(current, original, rtol=0, atol=0)
