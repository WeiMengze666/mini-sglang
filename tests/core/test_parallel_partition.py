import pytest
import torch
from minisgl.distributed.layout import ParallelLayout
from minisgl.distributed.partition import Partition


def test_logical_slices_and_global_routing_ids():
    world = torch.arange(96).reshape(8, 12)
    for rank in range(4):
        partition = Partition(rank, 4)
        torch.testing.assert_close(
            partition.shard(1, 4, block=3).slice(world), world[:, rank * 3 : (rank + 1) * 3]
        )
        ids = torch.tensor([[0, 3, 4, 7]])
        actual = partition.local_ids(ids, 8)
        owned = partition.range(8)
        expected = [[int(i) - owned.start if int(i) in owned else -1 for i in ids[0]]]
        torch.testing.assert_close(actual, torch.tensor(expected))
    with pytest.raises(ValueError):
        Partition(2, 2)
    with pytest.raises(ValueError):
        Partition(0, 3).range(8)


def test_attention_and_expert_axes_are_independent():
    layout = ParallelLayout(5, 8, dcp_size=2, ep_size=4)
    assert layout.tensor == Partition(5, 8)
    assert layout.attention == Partition(2, 4)
    assert layout.experts == Partition(1, 4)
    assert layout.expert_tensor == Partition(1, 2)
