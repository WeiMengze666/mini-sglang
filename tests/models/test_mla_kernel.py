"""Queued absorbed MLA/LSE checks, including padded TP head counts."""

import os

import pytest
import torch
from minisgl.attention.mla import mla_attention
from minisgl.attention.mla_fi import FlashInferMLA

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


@pytest.mark.parametrize(
    "heads,length,tokens,start,offset",
    [
        (12, 1, 1, 1, 0),
        (12, 17, 1, 33, 0),
        (24, 127, 1, 253, 0),
        (24, 0, 1, 0, 0),
        (24, 10, 7, 13, 0),
        (24, 4, 7, 0, 0),
        (24, 3, 7, 0, 1),
        (12, 17, 1, 0, 1),
    ],
)
def test_absorbed_decode_values_and_natural_lse(heads, length, tokens, start, offset):
    torch.manual_seed(107)
    q = torch.randn(tokens, heads, 256, dtype=torch.bfloat16, device="cuda") * 0.1
    latent = torch.randn(length, 512, dtype=torch.bfloat16, device="cuda") * 0.1
    key = torch.randn(length, 64, dtype=torch.bfloat16, device="cuda") * 0.1
    weight = torch.randn(heads * 320, 512, dtype=torch.bfloat16, device="cuda") * 0.05
    positions = torch.arange(length, device="cuda") * 2 + offset
    query_pos = torch.arange(start, start + tokens, device="cuda")
    expected, expected_lse = mla_attention(
        q, latent, key, weight, query_pos, positions, nope_dim=192, value_dim=128, return_lse=True
    )
    native = FlashInferMLA()
    for _ in range(2):
        actual, lse = native(
            q,
            latent,
            key,
            weight,
            query_pos,
            positions,
            nope_dim=192,
            value_dim=128,
            return_lse=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
        torch.testing.assert_close(lse, expected_lse, rtol=0.001, atol=0.001)
    assert len(native._plans) == int(bool((positions[None, :] <= query_pos[:, None]).any()))
