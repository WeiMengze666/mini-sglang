"""Native convolution ABI, explicit execution stage and history ownership."""

import pytest
import torch
from minisgl.attention.kda import causal_conv as native


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("tokens,is_decode", [(1, False), (1, True), (7, False)])
@pytest.mark.parametrize("native_layout", [False, True])
def test_native_convolution_state_layout_and_stage(
    monkeypatch, batched, tokens, is_decode, native_layout
):
    batch, channels, width = (2 if batched else 1), 5, 4
    x = torch.arange(batch * tokens * channels).reshape(batch, tokens, channels).bfloat16()
    state = (
        torch.arange(batch * channels * (width - 1)).reshape(batch, channels, width - 1).bfloat16()
    )
    if native_layout:
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    before = state.clone()
    original = x.clone()
    weight = torch.ones(channels, width)
    seen = []

    def update(pool):
        assert pool.shape == (batch, channels, width - 1) and pool.stride(1) == 1
        torch.testing.assert_close(pool, before, rtol=0, atol=0)
        pool.add_(7)

    def prefill(value, w, bias, **options):
        seen.append("prefill")
        assert value.shape == (channels, batch * tokens) and value.stride(0) == 1
        assert options["seq_lens_cpu"] == [tokens] * batch
        assert options["query_start_loc"].tolist() == [i * tokens for i in range(batch + 1)]
        update(options["conv_states"])
        return torch.full_like(value, 9)

    def decode(value, pool, w, **options):
        seen.append("decode")
        assert value.shape == (batch, channels) and options["activation"] == "silu"
        update(pool)
        return torch.full_like(value, 9)

    monkeypatch.setattr(native, "_kernels", lambda: (prefill, decode))
    out, final = native.causal_conv1d_update(
        x if batched else x[0], weight, state if batched else state[0], is_decode=is_decode
    )
    assert seen == ["decode" if is_decode else "prefill"]
    torch.testing.assert_close(out, torch.full_like(x if batched else x[0], 9), rtol=0, atol=0)
    torch.testing.assert_close(final, (before + 7) if batched else (before[0] + 7), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(state, before, rtol=0, atol=0)
