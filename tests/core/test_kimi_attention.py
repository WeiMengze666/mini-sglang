"""CPU numerical regressions for eager KDA and latent MLA (no model download)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.models.kimi_attention import (
    apply_rotary,
    causal_conv1d,
    gated_rms_norm,
    kda_log_decay,
    kda_recurrence,
    mla_attention,
)
from minisgl.models.kimi_attention_ops import KimiKDA, KimiMLA


def config(**kwargs):
    values = {
        "hidden_size": 12,
        "num_attention_heads": 2,
        "qk_nope_head_dim": 3,
        "qk_rope_head_dim": 4,
        "v_head_dim": 3,
        "kv_lora_rank": 5,
        "q_lora_rank": 6,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 32,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "rope_theta": 10000.0,
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def initialize(op):
    for name, tensor in op.state_dict().items():
        tensor.normal_(std=0.2)
        if "norm.weight" in name:
            tensor.fill_(1)
    return op


@pytest.mark.parametrize("width", [1, 3, 5])
def test_conv_matches_depthwise_and_streaming(width):
    torch.manual_seed(4)
    x, w = torch.randn(9, 7), torch.randn(7, 1, width)
    actual, state = causal_conv1d(x, w)
    expected = F.silu(F.conv1d(F.pad(x.T[None], (width - 1, 0)), w, groups=7))[0].T
    torch.testing.assert_close(actual, expected)
    pieces, history = [], None
    for chunk in x.split([2, 1, 6]):
        y, history = causal_conv1d(chunk, w, history)
        pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces), expected)
    torch.testing.assert_close(history, state)


def test_safe_gate_is_sigmoid_not_clamped_softplus():
    f = torch.tensor([[[-2.0, 0.0, 2.0], [2.0, -1.0, 0.0]]])
    a, bias = torch.tensor([0.0, 0.7]), torch.arange(6.0) / 10
    expected = -5 * torch.sigmoid(a.exp()[None, :, None] * (f + bias.reshape(2, 3)))
    torch.testing.assert_close(kda_log_decay(f, a, bias, -5.0), expected)
    torch.testing.assert_close(
        kda_log_decay(f, a, bias), -a.exp()[None, :, None] * F.softplus(f + bias.reshape(2, 3))
    )


def test_kda_matches_explicit_transition_matrix_and_nonzero_state():
    torch.manual_seed(8)
    q, k, v = (torch.randn(5, 2, 4) for _ in range(3))
    decay, beta = -torch.rand_like(q), torch.rand(5, 2)
    initial = torch.randn(2, 4, 4)
    saved = initial.clone()
    actual, final = kda_recurrence(q, k, v, decay, beta, initial)
    expected, state = [], initial.clone()
    for t in range(len(q)):
        head_out = []
        for h in range(2):
            kt = k[t, h] / (k[t, h].square().sum() + 1e-6).sqrt()
            qt = q[t, h] / (q[t, h].square().sum() + 1e-6).sqrt() / 2
            transition = torch.eye(4) - beta[t, h] * torch.outer(kt, kt)
            state[h] = transition @ torch.diag(decay[t, h].exp()) @ state[h]
            state[h] += beta[t, h] * torch.outer(kt, v[t, h])
            head_out.append(qt @ state[h])
        expected.append(torch.stack(head_out))
    torch.testing.assert_close(actual, torch.stack(expected), atol=5e-7, rtol=2e-5)
    torch.testing.assert_close(final, state, atol=5e-7, rtol=2e-5)
    torch.testing.assert_close(initial, saved)
    first, middle = kda_recurrence(q[:2], k[:2], v[:2], decay[:2], beta[:2], initial)
    rest, streamed = kda_recurrence(q[2:], k[2:], v[2:], decay[2:], beta[2:], middle)
    torch.testing.assert_close(torch.cat((first, rest)), actual)
    torch.testing.assert_close(streamed, final)


def test_gated_norm_and_zero_queries_are_finite():
    x, g, w = torch.randn(3, 2, 4), torch.randn(3, 2, 4), torch.rand(4)
    torch.testing.assert_close(
        gated_rms_norm(x, g, w),
        x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt() * w * g.sigmoid(),
    )
    zero = torch.zeros_like(x)
    output, state = kda_recurrence(zero, zero, x, -torch.ones_like(x), torch.ones(3, 2))
    assert torch.isfinite(output).all() and torch.isfinite(state).all()
    assert output.count_nonzero() == 0


def test_mla_matches_expanded_sdpa_and_causal_mask():
    torch.manual_seed(2)
    q, latent, suffix = torch.randn(4, 2, 7), torch.randn(6, 5), torch.randn(6, 4)
    weight = torch.randn(12, 5)
    positions, history = torch.arange(2, 6), torch.arange(6)
    actual = mla_attention(q, latent, suffix, weight, positions, history, nope_dim=3, value_dim=3)
    expanded = F.linear(latent, weight).reshape(6, 2, 6)
    k, v = expanded.split(3, dim=-1)
    k = torch.cat((k, suffix[:, None, :].expand(-1, 2, -1)), dim=-1)
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=history[None] <= positions[:, None],
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    # A future key cannot affect an earlier query.
    changed = latent.clone()
    changed[-1] += 1000
    altered = mla_attention(q, changed, suffix, weight, positions, history, nope_dim=3, value_dim=3)
    torch.testing.assert_close(actual[:-1], altered[:-1])


@pytest.mark.parametrize("interleaved", [True, False])
def test_rotary_pair_layout(interleaved):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    angles = torch.tensor([1.0, 0.01])
    a, b = (x[0, 0::2], x[0, 1::2]) if interleaved else x[0].chunk(2)
    real, imag = a * angles.cos() - b * angles.sin(), b * angles.cos() + a * angles.sin()
    expected = torch.stack((real, imag), -1).flatten() if interleaved else torch.cat((real, imag))
    torch.testing.assert_close(
        apply_rotary(x, torch.tensor([1]), interleaved=interleaved)[0], expected
    )


@pytest.mark.parametrize("cls", [KimiKDA, KimiMLA])
def test_attention_prefill_equals_chunk_and_decode(cls):
    torch.manual_seed(10)
    op = initialize(cls(config()))
    x, positions = torch.randn(8, 12), torch.arange(8)
    expected, full_state = op.forward(x, positions)
    state, outputs = None, []
    for start, end in [(0, 3), (3, 5), (5, 6), (6, 7), (7, 8)]:
        y, state = op.forward(x[start:end], positions[start:end], state)
        outputs.append(y)
    torch.testing.assert_close(torch.cat(outputs), expected, atol=3e-6, rtol=3e-5)
    for a, b in zip(state, full_state):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def test_k3_nope_keeps_shared_suffix_and_ignores_rotary_positions():
    torch.manual_seed(5)
    op = initialize(KimiMLA(config()))
    x = torch.randn(4, 12)
    q, latent, key = op.project(x, torch.arange(4))
    q2, latent2, key2 = op.project(x, torch.arange(4) + 100)
    for a, b in ((q, q2), (latent, latent2), (key, key2)):
        torch.testing.assert_close(a, b)
    _, projected_suffix = op.kv_a_proj_with_mqa.forward(x).split((5, 4), -1)
    torch.testing.assert_close(key, projected_suffix)
    expected = op.attend(x, q, latent, key, torch.arange(4), torch.arange(4))
    zero_suffix = op.attend(x, q, latent, torch.zeros_like(key), torch.arange(4), torch.arange(4))
    assert not torch.allclose(expected, zero_suffix)


@pytest.mark.parametrize("full_gate", [True, False])
@pytest.mark.parametrize("width", [1, 3])
def test_kda_gate_variants_and_unit_kernel(full_gate, width):
    torch.manual_seed(11)
    cfg = config()
    cfg.linear_attn_config.update(use_full_rank_gate=full_gate, short_conv_kernel_size=width)
    op = initialize(KimiKDA(cfg))
    x = torch.randn(3, 12)
    full, _ = op.forward(x)
    first, state = op.forward(x[:1])
    last, _ = op.forward(x[1:], state=state)
    torch.testing.assert_close(torch.cat((first, last)), full, atol=2e-6, rtol=3e-5)


def test_standard_rotary_mla_without_q_lora_or_output_gate():
    torch.manual_seed(12)
    op = initialize(
        KimiMLA(config(mla_use_nope=False, q_lora_rank=None, mla_use_output_gate=False))
    )
    x, positions = torch.randn(4, 12), torch.arange(4)
    full, _ = op.forward(x, positions)
    first, state = op.forward(x[:2], positions[:2])
    last, _ = op.forward(x[2:], positions[2:], state)
    torch.testing.assert_close(torch.cat((first, last)), full, atol=2e-6, rtol=3e-5)


def test_empty_primitive_chunks_preserve_state():
    state = torch.randn(2, 4, 4)
    empty = torch.empty(0, 2, 4)
    out, final = kda_recurrence(empty, empty, empty, empty, torch.empty(0, 2), state)
    assert out.shape == empty.shape
    torch.testing.assert_close(final, state)
    history = torch.randn(4, 2)
    out, final = causal_conv1d(torch.empty(0, 4), torch.randn(4, 3), history)
    assert out.shape == (0, 4)
    torch.testing.assert_close(final, history)
