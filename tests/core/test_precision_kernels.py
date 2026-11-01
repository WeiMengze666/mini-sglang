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
        rows * torch.rsqrt(rows.square().mean(-1, keepdim=True) + residual_norm.eps) * residual_norm.weight.float(),
        proj.weight.float(),
    ).squeeze(-1)
    expected_residual = (scores.softmax(1).unsqueeze(-1) * rows).sum(1).to(head.dtype)
    torch.testing.assert_close(
        attention_residual(head, bank, residual_norm, proj),
        expected_residual,
        rtol=0,
        atol=0,
    )


def test_recurrent_fi_triton_backend_is_registered():
    assert "recurrent-fi-triton" in SUPPORTED_ATTENTION_BACKENDS.supported_names()
