"""Stable grouping against a separate Python bucket oracle."""

import random

import pytest
import torch
from minisgl.moe.fused import moe_align_block_size
from minisgl.moe.routing import align_routes


def oracle(ids, block, experts):
    flat = ids.reshape(-1).tolist()
    result, owners = [], []
    for expert in [*range(experts), -1]:
        bucket = [index for index, value in enumerate(flat) if value == expert]
        if bucket:
            bucket += [len(flat)] * (-len(bucket) % block)
            result.extend(bucket)
            owners.extend([expert] * (len(bucket) // block))
    return result, owners


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("block", [1, 3, 8, 16])
@pytest.mark.parametrize("rows,topk,experts", [(0, 3, 5), (1, 1, 1), (7, 3, 5), (27, 16, 896)])
def test_stable_grouping(dtype, block, rows, topk, experts):
    rng = random.Random(103)
    ids = torch.tensor(
        [rng.randrange(-1, experts) for _ in range(rows * topk)], dtype=dtype
    ).reshape(rows, topk)
    before = ids.clone()
    expected, owner_ids = oracle(ids, block, experts)
    first = align_routes(ids, block, experts)
    assert first[2].tolist() == [len(expected)]
    assert first[0][: len(expected)].tolist() == expected
    assert first[1][: len(owner_ids)].tolist() == owner_ids
    assert (first[0][len(expected) :] == rows * topk).all()
    assert (first[1][len(owner_ids) :] == -1).all()
    assert torch.equal(ids, before)
    for actual, reference in zip(moe_align_block_size(ids, block, experts, stable=True), first):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("remote", [False, True])
def test_single_bucket_and_noncontiguous_inputs(remote):
    ids = torch.full((7, 6), -1 if remote else 2, dtype=torch.int32)[:, ::2]
    assert not ids.is_contiguous()
    expected, owners = oracle(ids, 8, 5)
    actual, expert_ids, count = align_routes(ids, 8, 5)
    assert count.item() == len(expected)
    assert actual[: len(expected)].tolist() == expected
    assert expert_ids[: len(owners)].tolist() == owners


@pytest.mark.parametrize(
    "ids,block,experts",
    [
        (torch.zeros(2), 8, 3),
        (torch.zeros(2, 2), 8, 3),
        (torch.tensor([[-2]]), 8, 3),
        (torch.tensor([[3]]), 8, 3),
        (torch.tensor([[0]]), 0, 3),
        (torch.tensor([[0]]), 8, 0),
    ],
)
def test_invalid_metadata(ids, block, experts):
    with pytest.raises(ValueError):
        align_routes(ids, block, experts)
