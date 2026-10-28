"""The same numerical comparison accepts independently saved engine formats."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from compare_suite import full_logprobs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_saved_mini_logits_and_official_logprobs_share_the_same_comparison(dtype):
    logits = torch.tensor([[1.0, -2.0, 3.25, 0.5]], dtype=dtype)
    expected = logits.float().log_softmax(-1).reshape(-1)
    for saved in ({"logprobs": expected}, {"full": {"logits": logits}}):
        torch.testing.assert_close(full_logprobs(saved), expected, rtol=0, atol=0)
