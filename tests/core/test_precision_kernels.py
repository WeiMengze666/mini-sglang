import os

import pytest
import torch
import torch.nn.functional as F
from minisgl.attention import SUPPORTED_ATTENTION_BACKENDS
from minisgl.layers.attn_residual import attention_residual
from minisgl.layers.linear import headwise_linear, linear_fp32
from minisgl.layers.norm import TorchRMSNorm


def test_cpu_precision_helpers_match_reference():
    torch.manual_seed(3)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.randn(5, 8, dtype=torch.bfloat16)
    actual = linear_fp32(x, weight)
    expected = F.linear(x.float(), weight.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    head_x = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    head_w = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    torch.testing.assert_close(
        headwise_linear(head_x, head_w),
        torch.einsum("thk,hnk->thn", head_x, head_w),
        rtol=0,
        atol=0,
    )


def test_cpu_rms_and_residual_helpers_match_reference():
    torch.manual_seed(4)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    norm = TorchRMSNorm(8, 1e-5)
    norm.weight.copy_(torch.randn(8, dtype=torch.bfloat16))
    actual = norm.forward(x)
    xf = x.float()
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)).to(x.dtype)
    torch.testing.assert_close(actual, expected * norm.weight, rtol=0, atol=0)

    head = torch.randn(3, 8, dtype=torch.bfloat16)
    bank = [torch.randn(3, 8, dtype=torch.bfloat16)]
    residual_norm = TorchRMSNorm(8, 1e-5)
    residual_norm.weight.copy_(torch.randn(8, dtype=torch.bfloat16))
    proj = type("Projection", (), {"weight": torch.randn(1, 8, dtype=torch.bfloat16)})()
    rows = torch.stack([*bank, head], dim=1).float()
    scores = F.linear(
        rows
        * torch.rsqrt(rows.square().mean(-1, keepdim=True) + residual_norm.eps)
        * residual_norm.weight.float(),
        proj.weight.float(),
    ).squeeze(-1)
    expected_residual = (scores.softmax(1).unsqueeze(-1) * rows).sum(1).to(head.dtype)
    torch.testing.assert_close(
        attention_residual(head, bank, residual_norm, proj),
        expected_residual,
        rtol=0,
        atol=0,
    )


def test_flashinfer_supports_hybrid_attention():
    assert SUPPORTED_ATTENTION_BACKENDS["fi"].stateful_layers


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
@pytest.mark.parametrize("row_stride", [1, 2])
def test_strict_rms_norm_rounds_after_weight_multiplication(monkeypatch, row_stride):
    import minisgl.core as core

    ctx = core.Context(page_size=1, bf16_precision="strict")
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    x = torch.tensor(
        [[0.03125, 0.5, 0.765625, 1.9921875], [0.1875, -0.203125, 1.71875, 2.5]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([2.0, 0.333984375, 1.5, -2.25], dtype=torch.bfloat16)
    normalized = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5)
    expected = (normalized * weight.double()).to(x.dtype)
    # The fixture distinguishes a single output rounding from an intermediate BF16 cast.
    early_rounding = normalized.to(x.dtype) * weight
    assert not torch.equal(early_rounding, expected)
    storage = torch.empty(2 * row_stride, 4, dtype=x.dtype, device="cuda")
    rows = storage[::row_stride]
    rows.copy_(x)
    norm = TorchRMSNorm(4, 1e-5)
    norm.weight = weight.to("cuda")
    actual = norm.forward(rows)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["strict", "default"])
def test_cpu_rms_norm_uses_selected_rounding_policy(monkeypatch, precision):
    import minisgl.core as core

    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1, bf16_precision=precision))
    x = torch.tensor(
        [[0.03125, 0.5, 0.765625, 1.9921875], [0.1875, -0.203125, 1.71875, 2.5]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([2.0, 0.333984375, 1.5, -2.25], dtype=x.dtype)
    normalized = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5)
    late = (normalized * weight.double()).to(x.dtype)
    early = normalized.to(x.dtype) * weight
    assert not torch.equal(late, early)
    norm = TorchRMSNorm(4, 1e-5)
    norm.weight = weight
    expected = late if precision == "strict" else early
    torch.testing.assert_close(norm.forward(x), expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_strict_linear_matches_fp64_and_token_partition(monkeypatch, output_dtype):
    import minisgl.core as core
    from minisgl.layers.linear import linear_bf16

    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1, bf16_precision="strict"))
    torch.manual_seed(76)
    for tokens, hidden, width in [(0, 65, 17), (27, 129, 35)]:
        storage = torch.randn(tokens * 2, hidden * 2, dtype=torch.bfloat16, device="cuda")
        x = storage[::2, ::2]
        weight = torch.randn(width, hidden, dtype=x.dtype, device=x.device)
        project = linear_bf16 if output_dtype == torch.bfloat16 else linear_fp32
        expected = F.linear(x.double(), weight.double()).to(output_dtype)
        actual = project(x, weight)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.008 if output_dtype == torch.bfloat16 else 1e-5,
            atol=1e-5,
        )
        if tokens:
            chunks = torch.cat([project(x[i : i + 7], weight) for i in range(0, tokens, 7)])
            torch.testing.assert_close(actual, chunks, rtol=0, atol=0)
    x = torch.randn(27, 3, 65, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(3, 17, 65, dtype=x.dtype, device=x.device)
    actual = headwise_linear(x, weight)
    expected = torch.einsum("thk,hnk->thn", x.double(), weight.double()).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
    chunks = torch.cat([headwise_linear(x[i : i + 7], weight) for i in range(0, len(x), 7)])
    torch.testing.assert_close(actual, chunks, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
@pytest.mark.parametrize("width", [1, 128, 513, 1024, 3584, 7168])
@pytest.mark.parametrize("gated", [False, True])
def test_streaming_rms_matches_formula_and_token_partition(width, gated):
    from minisgl.layers.norm_kernel import rms_norm

    generator = torch.Generator(device="cuda").manual_seed(157)
    storage = torch.randn(10, width * 2, dtype=torch.bfloat16, device="cuda", generator=generator)
    x = storage[::2, ::2]
    weight = torch.randn(width, dtype=x.dtype, device=x.device, generator=generator)
    gate = (
        torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) if gated else None
    )
    xd = x.double()
    expected = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-5) * weight.double()
    if gate is not None:
        expected *= gate.double().sigmoid()
    actual = rms_norm(x, weight, 1e-5, gate=gate)
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0.008, atol=1e-5)
    chunks = []
    for start in range(0, len(x), 2):
        part_gate = None if gate is None else gate[start : start + 2]
        chunks.append(rms_norm(x[start : start + 2], weight, 1e-5, gate=part_gate))
    torch.testing.assert_close(actual, torch.cat(chunks), rtol=0, atol=0)
    empty = rms_norm(x[:0], weight, 1e-5, gate=None if gate is None else gate[:0])
    assert empty.shape == x[:0].shape and empty.dtype == x.dtype
