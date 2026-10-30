"""Queued stable-routing correctness and replay across changed routing values."""

import os

import pytest
import torch
from minisgl.moe.routing import align_routes

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


def expected(ids, block, experts):
    flat = ids.reshape(-1).tolist()
    result, owner = [], []
    for expert in [*range(experts), -1]:
        group = [index for index, value in enumerate(flat) if value == expert]
        group.extend([len(flat)] * (-len(group) % block))
        result.extend(group)
        owner.extend([expert] * (len(group) // block))
    return result, owner


@pytest.mark.parametrize(
    "rows,topk,experts,block", [(1, 1, 1, 1), (7, 3, 5, 8), (27, 16, 896, 8), (149, 16, 112, 16)]
)
def test_routing_replay(rows, topk, experts, block):
    generator = torch.Generator().manual_seed(103)
    cpu = torch.randint(-1, experts, (rows, topk), dtype=torch.int32, generator=generator)
    ids = cpu.cuda()
    align_routes(ids, block, experts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = align_routes(ids, block, experts)
    for sample in [cpu, torch.full_like(cpu, -1), torch.zeros_like(cpu), cpu.flip(0)]:
        ids.copy_(sample)
        graph.replay()
        sorted_ids, owners, count = [value.cpu() for value in outputs]
        token_ids, expert_ids = expected(sample, block, experts)
        assert count.tolist() == [len(token_ids)]
        assert sorted_ids[: len(token_ids)].tolist() == token_ids
        assert owners[: len(expert_ids)].tolist() == expert_ids
        assert (sorted_ids[len(token_ids) :] == rows * topk).all()
        assert (owners[len(expert_ids) :] == -1).all()
        for _ in range(3):
            graph.replay()
            assert all(
                torch.equal(before, after.cpu())
                for before, after in zip([sorted_ids, owners, count], outputs)
            )
