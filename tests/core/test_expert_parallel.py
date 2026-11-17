"""Actual CPU Gloo row semantics/ownership, not native CUDA/NCCL qualification."""

import importlib.util
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.distributed.expert import ExpertParallel
from minisgl.distributed.impl import DistributedCommunicator, TorchDistributedImpl
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.layout import ParallelLayout


def test_single_expert_partition_has_no_group_or_device_side_effect(monkeypatch):
    monkeypatch.setattr(dist, "get_backend", lambda *a: pytest.fail("EP1 created a group"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("EP1 initialized CUDA"))
    assert ExpertParallel.prepare(SimpleNamespace(ep_size=1), None, "cuda:0") is None


@pytest.mark.parametrize("ep", [2, 4, 8])
@pytest.mark.parametrize("total", [0, 1, 3, 7, 8, 9])
def test_host_source_plan_uses_physical_token_rows(ep, total):
    mesh = object.__new__(ExpertParallel)
    mesh.info = DistributedInfo(0, ep)
    counts = mesh.source_counts(total)
    assert counts == tuple(len(range(rank, total, ep)) for rank in range(ep))
    assert sum(counts) == total


@pytest.mark.parametrize("total", [-1, True, 1.5])
def test_invalid_host_source_plan_is_rejected(total):
    mesh = object.__new__(ExpertParallel)
    mesh.info = DistributedInfo(0, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        mesh.source_counts(total)


def _standalone():
    mesh = object.__new__(ExpertParallel)
    mesh.info, mesh.tensor_info = DistributedInfo(0, 2), DistributedInfo(0, 2)
    mesh.device, mesh.closed, mesh.failed = torch.device("cpu"), False, False
    mesh._owned = []
    return mesh


@pytest.mark.parametrize("operation", ["reduce", "gather"])
def test_unknown_collective_failure_is_poisoned_without_retry(operation):
    mesh = _standalone()
    calls = []

    def fail(value):
        calls.append("native")
        raise RuntimeError("collective failed")

    mesh._reduce = SimpleNamespace(all_reduce=fail)
    mesh._gather = SimpleNamespace(all_gather=fail)
    call = (
        (lambda: mesh.reduce_source(torch.ones(1, 3)))
        if operation == "reduce"
        else (lambda: mesh.gather_source(torch.ones(1, 3), 2))
    )
    with pytest.raises(RuntimeError, match="collective failed"):
        call()
    assert mesh.failed and calls == ["native"]
    with pytest.raises(RuntimeError, match="cannot be retried"):
        call()
    with pytest.raises(RuntimeError, match="worker abort"):
        mesh.close()
    assert calls == ["native"]


def test_close_is_reverse_owned_only_and_idempotent():
    mesh = _standalone()
    calls = []
    mesh._owned = [
        SimpleNamespace(close=lambda: calls.append("EP")),
        SimpleNamespace(close=lambda: calls.append("ETP")),
    ]
    mesh._gather = SimpleNamespace(close=lambda: pytest.fail("borrowed TP closed"))
    mesh.close()
    mesh.close()
    assert calls == ["ETP", "EP"] and mesh.closed
    with pytest.raises(RuntimeError, match="closed"):
        mesh.gather_source(torch.ones(1, 3), 2)


@pytest.mark.parametrize("bad", ["shape", "stride", "rowplan"])
def test_metadata_violation_precedes_collective(bad):
    mesh = _standalone()
    mesh._gather = SimpleNamespace(
        all_gather=lambda value: pytest.fail("bad producer entered collective")
    )
    value = torch.ones(1, 3)
    if bad == "shape":
        value = value.flatten()
    elif bad == "stride":
        value = torch.ones(1, 6)[:, ::2]
    else:
        value = torch.ones(2, 3)
    with pytest.raises(ValueError):
        mesh.gather_source(value, 2)
    assert mesh.failed


def _rank_mixture(rank, total, active, dtype):
    # Independent whole-mixture oracle: every TP rank contributes its own full rows.
    return torch.tensor(
        [
            [rank + 1 + token + col if token < active else 0 for col in range(3)]
            for token in range(total)
        ],
        dtype=dtype,
    ).reshape(total, 3)


def _gloo_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(
        AssertionError("CPU test initialized CUDA")
    )
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    tp = dist.group.WORLD
    dcp = dist.new_group([0, 1, 2, 3], backend="gloo")
    borrowed = TorchDistributedImpl(tp)
    DistributedCommunicator.plugins = [borrowed]
    # Same actual mature CPU group-root adapter used by FI SymmDeviceMemory.
    path = Path(torch.__file__).parent.parent / "flashinfer/comm/comm_backend.py"
    spec = importlib.util.spec_from_file_location("fi_ep_mesh_comm_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ep in (2, 4):
        layout = ParallelLayout(DistributedInfo(rank, 4), dcp_size=4, ep_size=ep)
        mesh = ExpertParallel.prepare(layout, tp, "cpu")
        assert mesh.info == layout.experts and mesh.tensor_info == layout.expert_tensor
        assert mesh.ep_members == tuple(range(rank // ep * ep, (rank // ep + 1) * ep))
        assert mesh.etp_members == tuple(range(rank % ep, 4, ep))
        assert tuple(dist.get_process_group_ranks(mesh.cpu_group)) == mesh.ep_members
        if ep == 4:
            assert mesh.cpu_group is tp and mesh._gather is borrowed and mesh._owned == []
        backend = module.TorchDistBackend(mesh.cpu_group)
        root = dist.get_global_rank(mesh.cpu_group, 0)
        assert backend.bcast(rank if mesh.info.rank == 0 else None, root=0) == root
        for dtype in (torch.bfloat16, torch.float32):
            for total, active in ((0, 0), (1, 1), (3, 3), (7, 7), (8, 7), (9, 3)):
                owner_tokens = list(range(mesh.info.rank, total, ep))
                t = mesh.tensor_info.rank
                # Stand-in for completed EP combine: all experts at this fixed t
                # have contributed, but ETP slices have not yet been summed.
                source = sum(
                    (_rank_mixture(t * ep + e, total, active, dtype) for e in range(ep)),
                    torch.zeros(total, 3, dtype=dtype),
                )[owner_tokens].contiguous()
                pointer = source.data_ptr()
                reduced = mesh.reduce_source(source)
                expected = (
                    torch.stack([_rank_mixture(r, total, active, torch.float32) for r in range(4)])
                    .sum(0)
                    .to(dtype)
                )
                assert reduced.dtype == dtype and reduced.data_ptr() == pointer
                torch.testing.assert_close(reduced, expected[owner_tokens], rtol=0, atol=0)
                saved = reduced.clone()
                actual = mesh.gather_source(reduced, total)
                assert actual.shape == (total, 3) and actual.dtype == dtype
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(reduced, saved, rtol=0, atol=0)
        mesh.close()
        mesh.close()
        assert mesh.closed and all(not item.owns_group for item in mesh._owned)
        # Both borrowed namespaces remain live after owned EP/ETP groups close.
        dist.barrier(group=tp)
        dist.barrier(group=dcp)
        torch.testing.assert_close(
            borrowed.all_reduce(torch.tensor([rank + 1.0])), torch.tensor([10.0]), rtol=0, atol=0
        )
    dist.destroy_process_group(dcp)
    dist.destroy_process_group()
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in real multiprocess gate"
)
def test_real_four_rank_EP_ETP_rows_empty_padding_and_borrowed_ownership(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)
