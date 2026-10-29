"""The same numerical comparison accepts independently saved engine formats."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from checkpoint_suite import assert_chunked_prefill
from compare_suite import full_logprobs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_saved_mini_logits_and_official_logprobs_share_the_same_comparison(dtype):
    logits = torch.tensor([[1.0, -2.0, 3.25, 0.5]], dtype=dtype)
    expected = logits.float().log_softmax(-1).reshape(-1)
    for saved in ({"logprobs": expected}, {"full": {"logits": logits}}):
        torch.testing.assert_close(full_logprobs(saved), expected, rtol=0, atol=0)


def test_chunk_gate_rejects_greedy_change_within_pointwise_tolerance():
    full = {"logits": torch.tensor([[0.02, 0.01]]), "token_ids": [0]}
    chunked = {"logits": torch.tensor([[0.01, 0.02]]), "token_ids": [1]}
    with pytest.raises(AssertionError, match="greedy tokens"):
        assert_chunked_prefill(full, chunked, "bfloat16")


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_chunk_gate_checks_non_greedy_vocabulary_entries(dtype):
    full = {"logits": torch.tensor([[2.0, -1.0]]), "token_ids": [0]}
    assert_chunked_prefill(full, full, dtype)
    changed = {"logits": torch.tensor([[2.0, -0.5]]), "token_ids": [0]}
    with pytest.raises(AssertionError):
        assert_chunked_prefill(full, changed, dtype)
