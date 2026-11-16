"""DCP communication needs only partial outputs/LSE, independent of model or MLA."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.distributed.context import ContextParallel, init_context_parallel, merge_attention
from minisgl.distributed.layout import configure_parallel


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("all_empty", [False, True])
def test_neutral_empty_shards_preserve_producer_dtype(dtype, all_empty):
    output = torch.tensor([[[[0.0, 0.0]]], [[[5.0, 7.0]]]], dtype=dtype)
    if all_empty:
        output[1].zero_()
    lse = torch.tensor([[[-torch.inf]], [[-torch.inf if all_empty else 0.0]]])
    saved = output.clone()
    expected = torch.tensor([[[0.0, 0.0] if all_empty else [5.0, 7.0]]], dtype=dtype)
    actual = merge_attention(output, lse)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(output, saved, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf])
def test_invalid_lse_has_zero_weight_when_output_is_finite(invalid):
    output = torch.tensor([[[[3.0]]], [[[5.0]]]])
    lse = torch.tensor([[[invalid]], [[0.0]]])
    torch.testing.assert_close(merge_attention(output, lse), output[1], rtol=0, atol=0)


@pytest.mark.parametrize("lse_value", [-torch.inf, 0.0, torch.nan, torch.inf])
@pytest.mark.parametrize("value", [torch.nan, torch.inf])
def test_output_exceptions_are_not_sanitized_even_at_zero_weight(lse_value, value):
    output = torch.tensor([[[[value]]], [[[5.0]]]])
    lse = torch.tensor([[[lse_value]], [[0.0]]])
    result = merge_attention(output, lse)
    if value == torch.inf and lse_value == 0:
        assert result.isposinf().all()
    else:
        assert result.isnan().all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("base", ["e", "2"])
def test_weighted_merge_matches_independent_scalar_oracle(dtype, base):
    import math

    output = torch.tensor([[[[4.0, -8.0]]], [[[8.0, 4.0]]]], dtype=dtype)
    lse = torch.tensor([[[0.0]], [[math.log(3) if base == "e" else math.log2(3)]]])
    expected = torch.tensor([[[7.0, 1.0]]], dtype=dtype)
    actual = merge_attention(output, lse, lse_base=base)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_portable_exchange_restores_producer_dtype_and_head_order(dtype):
    from types import SimpleNamespace
    from minisgl.distributed.info import DistributedInfo

    cp = object.__new__(ContextParallel)
    cp.group = DistributedInfo(0, 2)
    output = torch.tensor([[[4.0, 8.0], [12.0, 16.0]]], dtype=dtype)
    lse = torch.zeros(1, 2)
    seen = []

    def exchange(send):
        seen.append(send)
        # rank 0 destination receives one head from each KV shard.
        return torch.stack((send[0], send[0]))

    cp._comm = SimpleNamespace(_backend=lambda: object(), all_to_all=exchange)
    actual = cp.exchange(output, lse)
    assert seen[0].dtype == torch.float32
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, output[:, :1], rtol=0, atol=0)
    cp.group = DistributedInfo(0, 1)
    assert cp.exchange(output, lse) is output


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
        # Restore ragged cyclic partitions, including empty ranks and strided inputs.
        for dtype in (torch.float32, torch.bfloat16):
            for total in (0, 1, 3, 17, 65):
                full = torch.arange(total * 6).reshape(total, 3, 2).to(dtype)
                local = full[rank::4, :, 1]
                restored = cp.gather_interleaved(local, total)
                torch.testing.assert_close(restored, full[:, :, 1], rtol=0, atol=0)
    finally:
        from minisgl.distributed.impl import destroy_distributed

        destroy_distributed()
        info._TP_INFO = None
        configure_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in multiprocess gate"
)
def test_four_rank_dcp_for_ordinary_multi_head_attention(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "context-group"),), nprocs=4, join=True)


@pytest.mark.parametrize("total,local_rows", [(-1, 0), (7, 1), (0, 1)])
def test_token_gather_rejects_inconsistent_host_lengths(monkeypatch, total, local_rows):

    cp = object.__new__(ContextParallel)
    cp.group = type("Group", (), {"rank": 0, "size": 4})()
    with pytest.raises(ValueError, match="cyclic partition"):
        cp.gather_interleaved(torch.empty(local_rows, 3), total)


def test_context_bootstrap_prepares_only_member_group_before_return(monkeypatch):
    from types import SimpleNamespace

    import minisgl.distributed.context as context
    from minisgl.distributed.info import DistributedInfo

    events = []
    monkeypatch.setattr(
        context,
        "get_parallel_layout",
        lambda: SimpleNamespace(
            tensor=DistributedInfo(1, 4),
            dcp_size=2,
        ),
    )
    monkeypatch.setattr(context.DistributedCommunicator, "groups", {})

    def group(ranks, *, backend):
        events.append(("group", tuple(ranks)))
        return tuple(ranks)

    def prepare(self, device):
        assert self.group == (0, 1) and device == torch.device("cuda:1")
        events.append(("warmup", self.group))

    monkeypatch.setattr(context.dist, "new_group", group)
    monkeypatch.setattr(context.TorchDistributedImpl, "warmup", prepare)
    context.init_context_parallel(backend="nccl", device=torch.device("cuda:1"))
    assert events == [("group", (0, 1)), ("warmup", (0, 1)), ("group", (2, 3))]
    assert context.DistributedCommunicator.groups["context"].group == (0, 1)
