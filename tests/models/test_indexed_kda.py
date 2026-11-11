"""Indexed official KDA updates match functional execution and preserve unrelated state."""

import os

import pytest
import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import KDAState
from minisgl.attention.kda.backend import KDAMetadata
from minisgl.attention.kda.registry import get_kda_kernel
from minisgl.attention.kda.sgl import indexed_decode, project
from minisgl.distributed import DistributedInfo
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.kvcache.unified import UnifiedMemory
from minisgl.layers.kda import KDALayer

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="explicit GPU gate"
)


@pytest.mark.parametrize("batch", [1, 8, 48])
@pytest.mark.parametrize("bound", [None, -5.0])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_indexed_decode_keeps_native_precision_and_slot_ownership(
    monkeypatch, batch, bound, unified, graph_mode
):
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr(layout, "_DCP_SIZE", 1)
    torch.manual_seed(1800 + batch)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            layer = KDALayer(KDAConfig(512, 12, 128, 4, gate_lower_bound=bound))
    finally:
        torch.set_default_dtype(previous_dtype)
    for tensor in layer.state_dict().values():
        tensor.normal_(std=0.02)
    layer.o_norm.weight.fill_(1)
    capacity = 2 * batch + 1
    state_bytes = (12 * 128 * 128 + 3 * 12 * 128 * 3) * 2
    arena = UnifiedMemory(capacity + 2, state_bytes, state_bytes, "cuda") if unified else None
    if arena is not None:
        arena.buffer.zero_()
    pool = RecurrentStatePool(
        (0,),
        capacity,
        12,
        128,
        4,
        torch.bfloat16,
        "cuda",
        max_cached_states=2,
        state_dtype=torch.bfloat16,
        arena=arena,
    )
    tables = list(reversed(range(0, 2 * batch, 2)))
    for table in tables:
        pool.begin_request(table, table + 100, 0)
        state = pool.get_kda_state(0, table)
        state.conv.normal_(std=0.1)
        state.recurrent.normal_(std=0.02)
        pool.finish_request(table, table + 100, 13)
    checkpoint = pool.snapshot(tables[0], tables[0] + 100, expected_len=13)
    for table in tables:
        pool.begin_request(table, table + 100, 13)
    indices = pool.batch_indices(tables)
    metadata = KDAMetadata(
        torch.arange(batch + 1, device="cuda", dtype=torch.int32), state_indices=indices
    )
    initial = pool.get_kda_batch(0, indices)
    x = torch.randn(batch, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    kernel = get_kda_kernel("sgl")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        if graph_mode:
            for _ in range(2):
                indexed_decode(layer, x, pool, 0, indices, metadata)
                pool.set_kda_batch(0, indices, initial)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = indexed_decode(layer, x, pool, 0, indices, metadata)
            pool.set_kda_batch(0, indices, initial)
        native_conv, native_state = pool.indexed_kda_state(0)
        inactive = torch.tensor(
            [i for i in range(len(native_state)) if i not in indices.tolist()],
            device="cuda",
            dtype=torch.long,
        )
        untouched = native_conv.index_select(0, inactive), native_state.index_select(0, inactive)
        for step in range(3):
            if step:
                for table in tables:
                    pool.begin_request(table, table + 100, 13 + step)
            previous = pool.get_kda_batch(0, indices)
            projected = project(layer, x)
            expected, final = layer.update(
                *(part.unsqueeze(1) for part in projected[:5]),
                previous,
                kernel=kernel,
                is_decode=True,
            )
            if graph_mode:
                graph.replay()
            else:
                result = indexed_decode(layer, x, pool, 0, indices, metadata)
            assert result.state is None and not result.normed
            torch.testing.assert_close(result.output, expected[:, 0], rtol=0, atol=0)
            actual = pool.get_kda_batch(0, indices)
            for value, reference in zip(actual, final):
                assert value.dtype == torch.bfloat16
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            for value, reference in zip((native_conv, native_state), untouched):
                torch.testing.assert_close(
                    value.index_select(0, inactive), reference, rtol=0, atol=0
                )
            for table in tables:
                pool.finish_request(table, table + 100, 14 + step)
            x.mul_(1.015625)
        if graph is not None:
            graph.reset()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for table in tables:
        pool.release_request(table, table + 100)
    pool.restore(0, 900, checkpoint)
    pool.begin_request(0, 900, 13)
    restored = pool.get_kda_state(0, 0)
    for value, reference in zip(restored, KDAState(*(part[0] for part in initial))):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    pool.finish_request(0, 900, 14)
    pool.release_request(0, 900)
    pool.drop_checkpoint(checkpoint)
    assert pool.is_idle
    if arena is not None:
        arena.check_integrity()
