"""Compressed MLA readouts must be merged before BF16 value projection."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.attention.config import MLAConfig
from minisgl.attention.mla import MLALatentOutput
from minisgl.distributed.context import merge_attention
from minisgl.layers.mla import MLALayer


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gated", [False, True])
def test_latent_readout_is_merged_before_local_value_projection(monkeypatch, cp_size, dtype, gated):
    import minisgl.layers.mla as module

    rank = cp_size - 1
    torch.manual_seed(317)
    partial = torch.randn(cp_size, 3, cp_size * 2, 5).to(dtype)
    logs = torch.randn(cp_size, 3, cp_size * 2)
    partial[:, 0] = 0
    logs[:, 0] = -torch.inf
    calls = []

    def exchange(values, lse):
        calls.append(tuple(values.shape))
        merged = merge_attention(partial, logs)
        return merged[:, rank * 2 : (rank + 1) * 2]

    monkeypatch.setattr(
        module, "TensorParallel", lambda: SimpleNamespace(size=cp_size, all_reduce=lambda x: x)
    )
    monkeypatch.setattr(
        module,
        "ContextParallel",
        lambda: SimpleNamespace(size=cp_size, rank=rank, exchange=exchange),
    )
    monkeypatch.setattr(
        module, "get_parallel_layout", lambda: SimpleNamespace(mla_decode=SimpleNamespace(size=1))
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        layer = MLALayer(MLAConfig(7, cp_size * 2, 3, 2, 3, 5, None, 32, output_gate=gated))
    finally:
        torch.set_default_dtype(previous)
    for tensor in layer.state_dict().values():
        tensor.normal_(std=0.1)
    x = torch.randn(3, 7).to(dtype)
    got = layer.finish(x, MLALatentOutput(partial[rank], logs[rank]))
    probabilities = torch.nan_to_num(logs.double().softmax(0))
    merged = torch.einsum("sth,sthl->thl", probabilities, partial.double())
    if cp_size > 1:
        merged = merged[:, rank * 2 : (rank + 1) * 2]
    weights = layer.kv_b_proj.weight.view(cp_size * 2, 6, 5)[rank * 2 : (rank + 1) * 2, 3:]
    values = torch.einsum("thl,hvl->thv", merged.to(dtype), weights).flatten(-2)
    if gated:
        values = values * torch.nn.functional.linear(x, layer.g_proj.weight).sigmoid()
    expected = torch.nn.functional.linear(values, layer.o_proj.weight)
    torch.testing.assert_close(
        got,
        expected,
        rtol=0 if dtype == torch.bfloat16 else 1e-5,
        atol=0 if dtype == torch.bfloat16 else 1e-7,
    )
    assert calls == ([(3, cp_size * 2, 5)] if cp_size > 1 else [])


def test_bf16_projection_does_not_commute_with_shard_merge():
    torch.manual_seed(319)
    values = torch.randn(2, 3, 4, 512).bfloat16() * 0.01
    logs = torch.randn(2, 3, 4)
    weights = torch.randn(4, 128, 512).bfloat16() * 0.01
    correct = torch.einsum("thl,hvl->thv", merge_attention(values, logs).bfloat16(), weights)
    projected = torch.stack([torch.einsum("thl,hvl->thv", v, weights) for v in values])
    wrong = merge_attention(projected, logs).bfloat16()
    assert torch.count_nonzero(correct != wrong) > 0


def _gloo_worker(rank, init_file):
    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo
    from minisgl.distributed.context import init_context_parallel
    from minisgl.distributed.layout import configure_parallel

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
        torch.manual_seed(331)
        partial = torch.randn(4, 3, 8, 5).bfloat16()
        logs = torch.randn(4, 3, 8)
        partial[:, 0] = torch.nan
        logs[:, 0] = -torch.inf
        layer = MLALayer(MLAConfig(7, 8, 3, 2, 3, 5, None, 32))
        weights = torch.randn(8, 6, 5).bfloat16() * 0.1
        layer.kv_b_proj.weight = weights.flatten(0, 1)
        layer.o_proj.weight = torch.randn(7, 6).bfloat16() * 0.1
        # This test isolates context exchange; tensor-parallel reduction has its own gate.
        layer._tp = SimpleNamespace(size=4, all_reduce=lambda x: x)
        x = torch.zeros(3, 7, dtype=torch.bfloat16)
        got = layer.finish(x, MLALatentOutput(partial[rank], logs[rank]))
        probabilities = torch.nan_to_num(logs.double().softmax(0))
        clean = partial.double().nan_to_num()
        merged = torch.einsum("sth,sthl->thl", probabilities, clean)[:, rank * 2 : (rank + 1) * 2]
        local_weights = weights[rank * 2 : (rank + 1) * 2, 3:]
        projected = torch.einsum("thl,hvl->thv", merged.bfloat16(), local_weights).flatten(-2)
        expected = torch.nn.functional.linear(projected, layer.o_proj.weight)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    finally:
        info._TP_INFO = None
        configure_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in multiprocess gate"
)
def test_four_rank_latent_decode_merge_and_value_projection(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path / "mla-latent"),), nprocs=4, join=True)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_full_history_prefill_selects_local_heads_without_output_exchange(monkeypatch, cp_size):
    import minisgl.layers.mla as module

    rank = cp_size - 1

    def exchange(*args):
        raise AssertionError("Full-history prefill must not exchange partial outputs")

    monkeypatch.setattr(
        module, "TensorParallel", lambda: SimpleNamespace(size=cp_size, all_reduce=lambda x: x)
    )
    monkeypatch.setattr(
        module,
        "ContextParallel",
        lambda: SimpleNamespace(size=cp_size, rank=rank, exchange=exchange),
    )
    monkeypatch.setattr(
        module, "get_parallel_layout", lambda: SimpleNamespace(mla_decode=SimpleNamespace(size=1))
    )
    layer = MLALayer(MLAConfig(7, cp_size * 2, 3, 2, 3, 5, None, 32))
    for tensor in layer.state_dict().values():
        tensor.normal_(std=0.1)
    q, latent, key = torch.randn(3, cp_size * 2, 5), torch.randn(7, 5), torch.randn(7, 2)
    positions, history = torch.arange(4, 7), torch.arange(7)
    expected = torch.randn(3, 2, 3)

    def attention(query, cache, suffix, weight, query_positions, key_positions, **kwargs):
        torch.testing.assert_close(query, q[:, rank * 2 : (rank + 1) * 2], rtol=0, atol=0)
        selected = layer.kv_b_proj.weight.view(cp_size * 2, 6, 5)[
            rank * 2 : (rank + 1) * 2
        ].flatten(0, 1)
        torch.testing.assert_close(weight, selected, rtol=0, atol=0)
        assert cache is latent and suffix is key
        assert query_positions is positions and key_positions is history
        assert not kwargs["return_lse"]
        return expected

    partial = layer.attend_partial(
        q, latent, key, positions, history, attention=attention, context_sharded=False
    )
    assert partial.values is expected and partial.lse is None
    got = layer.finish(torch.zeros(3, 7), partial)
    torch.testing.assert_close(
        got, torch.nn.functional.linear(expected.flatten(-2), layer.o_proj.weight), rtol=0, atol=0
    )
