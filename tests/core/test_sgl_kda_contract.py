"""State ownership and semantic-axis contract of the optional kernel adapter."""

from contextlib import nullcontext

import pytest
import torch
from minisgl.attention.kda import sgl


@pytest.mark.parametrize("tokens", [1, 7, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_state", [False, True])
def test_native_inplace_pool_does_not_mutate_framework_inputs(
    monkeypatch, tokens, dtype, has_state
):
    generator = torch.Generator().manual_seed(73)
    heads, keys, values = 2, 3, 5
    q, k, forget = [torch.randn(tokens, heads, keys, generator=generator) for _ in range(3)]
    v = torch.randn(tokens, heads, values, generator=generator)
    beta = torch.randn(tokens, heads, generator=generator)
    a, dt = torch.randn(heads, generator=generator), torch.randn(heads * keys, generator=generator)
    state = torch.arange(heads * keys * values, dtype=dtype).reshape(heads, keys, values)
    before = [part.clone() for part in (q, k, v, forget, beta, state)]
    expected = state if has_state else torch.zeros_like(state, dtype=torch.float32)
    expected = expected + 2
    seen = []

    def update(pool):
        torch.testing.assert_close(pool[0], (expected - 2).transpose(-1, -2), rtol=0, atol=0)
        assert pool.dtype == (dtype if has_state else torch.float32)
        pool.add_(2)
        seen.append(pool)

    def chunk(**kwargs):
        update(kwargs["initial_state"])
        assert kwargs["beta_is_raw"] and kwargs["use_qk_l2norm_in_kernel"]
        torch.testing.assert_close(kwargs["beta"][0], beta, rtol=0, atol=0)
        assert kwargs["cu_seqlens"].tolist() == [0, tokens]
        assert kwargs["cu_seqlens"].dtype == kwargs["chunk_indices"].dtype == torch.int32
        assert kwargs["chunk_indices"].shape == ((tokens + 63) // 64, 2)
        # The mature chunk kernel consumes V as its output allocation.
        kwargs["v"].fill_(9)
        return kwargs["v"]

    def decode(**kwargs):
        update(kwargs["initial_state"])
        assert kwargs["use_qk_l2norm_in_kernel"]
        torch.testing.assert_close(kwargs["b"], beta, rtol=0, atol=0)
        kwargs["out"].fill_(9)

    monkeypatch.setattr(sgl, "_kernels", lambda: (chunk, decode, nullcontext))
    output, final = sgl.recurrent_kda(
        q, k, v, forget, beta, a, dt, state if has_state else None, -5.0
    )
    assert len(seen) == 1 and output.shape == v.shape
    torch.testing.assert_close(output, torch.full_like(v, 9), rtol=0, atol=0)
    torch.testing.assert_close(final, expected, rtol=0, atol=0)
    for value, original in zip((q, k, v, forget, beta, state), before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_empty_sequence_preserves_state_without_loading_sdk(monkeypatch):
    monkeypatch.setattr(sgl, "_kernels", lambda: pytest.fail("empty input loaded SDK"))
    q = torch.empty(0, 2, 3)
    state = torch.arange(30.0).reshape(2, 3, 5)
    output, final = sgl.recurrent_kda(
        q,
        q,
        torch.empty(0, 2, 5),
        q,
        torch.empty(0, 2),
        torch.zeros(2),
        torch.zeros(6),
        state,
        None,
    )
    assert output.shape == (0, 2, 5)
    torch.testing.assert_close(final, state, rtol=0, atol=0)
    assert final.data_ptr() != state.data_ptr()
