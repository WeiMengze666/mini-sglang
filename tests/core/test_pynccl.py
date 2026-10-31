"""Queued real collectives: FP32 mixtures, gather and independent DCP groups."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@pytest.mark.parametrize("wheel", [False, True])
def test_nccl_linker_discovery(tmp_path, monkeypatch, wheel):
    import minisgl.kernel.pynccl as loader

    captured = {}
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path)]) if wheel else None
    library = tmp_path / "lib/libnccl.so.2"
    if wheel:
        library.parent.mkdir()
        library.touch()
    monkeypatch.setattr(loader.importlib.util, "find_spec", lambda _: spec)
    monkeypatch.setattr(loader, "load_aot", lambda *args, **kwargs: captured.update(kwargs))
    loader._load_nccl_module.cache_clear()
    try:
        loader._load_nccl_module()
        expected = [str(library), f"-Wl,-rpath,{library.parent}"] if wheel else ["-lnccl"]
        assert captured == {"cuda_files": ["pynccl.cu"], "extra_ldflags": expected}
    finally:
        loader._load_nccl_module.cache_clear()


def _worker(rank, world, init_file):
    import minisgl.distributed.info as info
    from minisgl.distributed import (
        DistributedCommunicator,
        DistributedInfo,
        destroy_distributed,
        enable_pynccl_distributed,
    )
    from minisgl.distributed.context import ContextParallel, init_context_parallel
    from minisgl.distributed.layout import configure_parallel

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    info._TP_INFO = DistributedInfo(rank, world)
    try:
        # Exercise both symmetric-buffer and oversized direct reductions.
        enable_pynccl_distributed(info._TP_INFO, dist.group.WORLD, max_bytes=65536)
        comm = DistributedCommunicator()
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for count in (17, 65536):
                x = torch.full((count,), rank + 1, device="cuda", dtype=dtype)
                expected = world * (world + 1) // 2
                assert comm.all_reduce(x).data_ptr() == x.data_ptr()
                torch.testing.assert_close(x, torch.full_like(x, expected), rtol=0, atol=0)
                # Reusing the internal buffer must not overwrite prior results.
                other = torch.ones_like(x)
                comm.all_reduce(other)
                torch.testing.assert_close(other, torch.full_like(other, world), rtol=0, atol=0)
                torch.testing.assert_close(x, torch.full_like(x, expected), rtol=0, atol=0)
            source = torch.full((3, 13), rank, device="cuda", dtype=dtype)
            gathered = comm.all_gather(source)
            expected = torch.arange(world, device="cuda", dtype=dtype).repeat_interleave(3)
            torch.testing.assert_close(gathered, expected[:, None].expand(-1, 13), rtol=0, atol=0)

        configure_parallel(dcp_size=2)
        init_context_parallel(backend="nccl")
        cp = ContextParallel()
        partial = torch.full((3, 4, 5), cp.rank + 1, device="cuda")
        lse = torch.zeros((3, 4), device="cuda")
        output = cp.exchange(partial, lse)
        torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)
        torch.cuda.synchronize()
    finally:
        destroy_distributed()
        configure_parallel()
        info._TP_INFO = None
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
def test_eight_rank_pynccl_and_nccl_context_groups(tmp_path):
    mp.spawn(_worker, args=(8, str(tmp_path / "pynccl-group")), nprocs=8, join=True)
