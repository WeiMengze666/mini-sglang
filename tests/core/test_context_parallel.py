"""DCP communication needs only partial outputs/LSE, independent of model or MLA."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.distributed.context import ContextParallel, init_context_parallel, merge_attention
from minisgl.distributed.layout import configure_parallel


@pytest.mark.parametrize("all_empty", [False, True])
def test_empty_shard_may_have_an_undefined_partial_output(all_empty):
    output = torch.tensor([[[[float("nan"), float("nan")]]], [[[5.0, 7.0]]]])
    lse = torch.tensor([[[-torch.inf]], [[-torch.inf if all_empty else 0.0]]])
    expected = torch.tensor([[[0.0, 0.0] if all_empty else [5.0, 7.0]]])
    torch.testing.assert_close(merge_attention(output, lse), expected, rtol=0, atol=0)
    assert output[0].isnan().all()  # Merging must not modify a producer's buffer.


@pytest.mark.parametrize("invalid", ["output", "nan_lse", "infinite_lse"])
def test_corrupt_active_shard_is_not_treated_as_empty(invalid):
    output = torch.tensor([[[[3.0]]], [[[5.0]]]])
    lse = torch.zeros(2, 1, 1)
    if invalid == "output":
        output[0] = torch.nan
    else:
        lse[0] = torch.nan if invalid == "nan_lse" else torch.inf
    assert merge_attention(output, lse).isnan().all()


def _worker(rank, init_file):
    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    info._TP_INFO = DistributedInfo(rank, 4)
    try:
        configure_parallel(dcp_size=4)
        init_context_parallel()
        cp = ContextParallel()
        torch.manual_seed(67)
        # Ordinary multi-head attention, with a distinct 5-D value space.
        q, k, v = torch.randn(3, 8, 7), torch.randn(9, 8, 7), torch.randn(9, 8, 5)
        q_pos, k_pos = torch.tensor([0, 3, 8]), torch.arange(9)
        score = torch.einsum("thd,shd->ths", q, k) / 7**0.5
        score.masked_fill_(k_pos[None, None, :] > q_pos[:, None, None], -torch.inf)
        expected = torch.einsum("ths,shv->thv", score.softmax(-1), v)
        owned = cp.owns(k_pos)
        local_score = score[..., owned]
        lse = local_score.logsumexp(-1)
        probabilities = torch.nan_to_num((local_score - lse[..., None]).exp(), nan=0)
        partial = torch.einsum("ths,shv->thv", probabilities, v[owned])
        output = cp.exchange(partial, lse)
        torch.testing.assert_close(
            output, expected[:, rank * 2 : (rank + 1) * 2], rtol=1e-5, atol=1e-6
        )
    finally:
        info._TP_INFO = None
        configure_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in multiprocess gate"
)
def test_four_rank_dcp_for_ordinary_multi_head_attention(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "context-group"),), nprocs=4, join=True)
