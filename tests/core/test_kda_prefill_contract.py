"""The public native ABI needs alignment as well as contiguous request rows."""

import sys
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.kda import kda_log_decay, kda_recurrence
from minisgl.attention.kda_fi import recurrent_kda


@pytest.mark.parametrize("offset", [0, 1])
def test_native_prefill_preserves_fp32_state_and_aligns_contiguous_slices(monkeypatch, offset):
    torch.manual_seed(271)
    tokens, heads, dim = 7, 3, 4
    q, k, v, forget = [torch.randn(tokens, heads, dim).bfloat16() for _ in range(4)]
    beta = torch.randn(tokens + 1, heads).bfloat16()[offset : offset + tokens]
    assert beta.is_contiguous()
    assert (beta.data_ptr() % 16 != 0) == bool(offset)
    a, dt = torch.randn(heads), torch.randn(heads * dim)
    initial = torch.randn(heads, dim, dim)
    saved = initial.clone()

    def native(q, k, v, g, beta, **kwargs):
        assert kwargs["backend"] == "cute-dsl"
        assert beta.data_ptr() % 16 == 0 and beta.is_contiguous()
        assert kwargs["initial_state"].dtype == torch.float32
        output, state = kda_recurrence(
            q[0],
            k[0],
            v[0],
            kda_log_decay(g[0], a, dt, -5),
            beta[0].float().sigmoid(),
            kwargs["initial_state"][0].transpose(-1, -2),
        )
        return output[None], state.transpose(-1, -2)[None].contiguous()

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(recurrent_kda=native))
    actual, final = recurrent_kda(q, k, v, forget, beta, a, dt, initial, -5)
    expected, state = kda_recurrence(
        q,
        k,
        v,
        kda_log_decay(forget, a, dt, -5),
        beta.float().sigmoid(),
        initial,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(final, state, rtol=0, atol=0)
    torch.testing.assert_close(initial, saved, rtol=0, atol=0)
