"""The grouped native mask must equal global-position causality, exactly."""

import torch
from minisgl.attention.mla import causal_query_groups


def test_interleaved_causal_groups_cover_every_visible_query_key_pair():
    for stride in (1, 2, 4, 8):
        for rank in range(stride):
            for start in (0, 1, 5, 16):
                for length in (1, 2, 7, 19):
                    keys = torch.arange(start + length)[rank::stride]
                    queries = torch.arange(start, start + length)
                    expected = keys[None, :] <= queries[:, None]
                    actual = torch.zeros_like(expected)
                    groups = causal_query_groups(start, length, rank, len(keys), stride)
                    for rows, visible in groups:
                        # Native bottom-right causal mask for this subsequence.
                        local = torch.arange(visible)[None, :] <= torch.arange(len(rows))[
                            :, None
                        ] + visible - len(rows)
                        actual[rows, :visible] = local
                    assert torch.equal(actual, expected), (stride, rank, start, length)
