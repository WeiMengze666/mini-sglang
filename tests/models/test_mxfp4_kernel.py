"""Opt-in CUDA checks against directly generated, exactly representable weights."""

import os

import pytest
import torch
import torch.nn.functional as F
from minisgl.quantization.mxfp4 import MXFP4Linear

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 17, 32),
        (2, 67, 96),
        (7, 83, 512),
        (65, 83, 512),
        (33, 67, 96),
        (27, 128, 7168),
        (1, 129, 12288),
    ],
)
def test_fused_weight_only_linear(dtype, m, n, k):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(59)
    codes = torch.randint(16, (n, k), device="cuda", dtype=torch.uint8)
    scales = torch.randint(123, 128, (n, k // 32), device="cuda", dtype=torch.uint8)
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    expected_weight = (
        table[codes.long()] * (2.0 ** (scales.float() - 127)).repeat_interleave(32, -1)
    ).to(dtype)
    layer = MXFP4Linear(k, n, has_bias=True, device="cuda", dtype=dtype)
    layer.weight_packed.copy_(codes[:, ::2] | codes[:, 1::2] << 4)
    layer.weight_scale.copy_(scales)
    layer.bias.normal_(std=0.1)
    layer.set_backend("triton")
    # A noncontiguous leading dimension exercises flattening without changing K.
    x = (torch.randn(2 * m, k, device="cuda", dtype=dtype) * 0.08)[::2]
    if m > 1:
        assert not x.is_contiguous()
    # The kernel accumulates in FP32; default BF16 cuBLAS may truncate split-K partials.
    expected = F.linear(x.float(), expected_weight.float(), layer.bias.float()).to(dtype)
    actual = layer.forward(x)
    tolerance = (3e-5, 3e-5) if dtype == torch.float32 else (0.015, 0.015)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    torch.testing.assert_close(layer.forward(x[:0]), expected[:0])


def test_codebook_sign_and_scale_blocks_exactly():
    layer = MXFP4Linear(64, 16, device="cuda", dtype=torch.float32)
    codes = torch.arange(16, device="cuda", dtype=torch.uint8)
    layer.weight_packed.copy_((codes | codes << 4)[:, None].expand(16, 32))
    layer.weight_scale[:, 0] = 126
    layer.weight_scale[:, 1] = 128
    x = torch.zeros(2, 64, device="cuda")
    x[0, 13], x[1, 49] = 1, 1
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    layer.set_backend("triton")
    torch.testing.assert_close(
        layer.forward(x), torch.stack((table * 0.5, table * 2)), rtol=0, atol=0
    )
