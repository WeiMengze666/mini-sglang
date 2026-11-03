"""Queued native MLA: FP8 query/KV, physical pages and cyclic DCP lengths."""

import os

import minisgl.distributed.info as info
import minisgl.distributed.layout as layout
import pytest
import torch
from minisgl.attention.mla.cutedsl import CuTeMLADecodePlan
from minisgl.attention.mla.flashinfer import FlashInferMLAAdapter
from minisgl.distributed import DistributedInfo
from minisgl.kvcache.mla_pool import MLACache
from minisgl.kvcache.unified import UnifiedMemory

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


def expected_partial(q, weight, cache, table, lengths, order, cp, rank):
    heads = q.shape[1]
    w = weight.view(heads, 256, 512)[:, :128]
    absorbed = torch.bmm(q[..., :128].transpose(0, 1), w).transpose(0, 1)
    query = torch.cat((absorbed, q[..., 128:]), -1).to(cache.dtype).double()
    values, logs = [], []
    for index, row in enumerate(order):
        latent, key = cache.gather_mla(1, table[row, rank : lengths[row] : cp])
        if not len(latent):
            values.append(torch.zeros(heads, 512, device=q.device, dtype=q.dtype))
            logs.append(torch.full((heads,), -torch.inf, device=q.device))
            continue
        keys = torch.cat((latent, key), -1).double()
        scores = query[index] @ keys.T * 192**-0.5
        values.append((scores.softmax(-1) @ latent.double()).to(q.dtype))
        logs.append(scores.logsumexp(-1).float())
    return torch.stack(values), torch.stack(logs)


@pytest.mark.parametrize("cp", [1, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("unified", [False, True])
def test_native_mla_precision_pages_graphs_and_stream(monkeypatch, cp, dtype, unified, request):
    rank = int(os.environ.get("LOCAL_RANK", "0")) % cp
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, cp))
    monkeypatch.setattr(layout, "_DCP_SIZE", cp)
    torch.manual_seed(220 + rank)
    logical_page, physical_page, num_pages = 64 * cp, 64, 32
    heads = 12 * cp
    lengths = [1, 65 * cp + 3, 129 * cp + 1]
    arena = (
        UnifiedMemory(num_pages, 2 * physical_page * 576 * dtype.itemsize, 256, "cuda")
        if unified
        else None
    )
    page_ids = [[7], [3, 11], [17, 5, 23]]
    if arena is not None:
        arena.allocate_pages([31] + [p for pages in page_ids for p in pages])
    cache = MLACache(
        (1, 3),
        num_pages * logical_page,
        512,
        64,
        dtype,
        "cuda",
        arena=arena,
        page_size=logical_page,
    )
    table = torch.zeros((3, 3 * logical_page), device="cuda", dtype=torch.int32)
    for row, pages in enumerate(page_ids):
        for index, page in enumerate(pages):
            table[row, index * logical_page : (index + 1) * logical_page] = (
                page * logical_page + torch.arange(logical_page, device="cuda")
            )
        for layer in (1, 3):
            source = (torch.randn(lengths[row], 576, device="cuda") * 0.1).bfloat16()
            other_layer = 3 if layer == 1 else 1
            before = cache.paged_storage(other_layer).view(torch.uint8).clone()
            cache.store_mla(
                layer,
                source[:, :512],
                source[:, 512:],
                table[row, : lengths[row]],
                implementation="sgl",
            )
            actual = torch.cat(cache.gather_mla(layer, table[row, rank : lengths[row] : cp]), -1)
            torch.testing.assert_close(
                actual.float(), source[rank::cp].to(dtype).float(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                cache.paged_storage(other_layer).view(torch.uint8), before, rtol=0, atol=0
            )
    q = (torch.randn(3, heads, 192, device="cuda") * 0.1).bfloat16()
    weight = (torch.randn(heads * 256, 512, device="cuda") * 0.03).bfloat16()
    native = FlashInferMLAAdapter()
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        order = [2, 0, 1]
        meta = cache.decode_metadata(order, [lengths[row] for row in order], table)
        expected, logs = expected_partial(q, weight, cache, table, lengths, order, cp, rank)
        output, lse = native.decode(
            q, cache.paged_storage(1), weight, meta, nope_dim=128, value_dim=128
        )
        assert isinstance(native._decode_plan, CuTeMLADecodePlan)
        torch.testing.assert_close(output, expected, rtol=0.03, atol=0.002)
        torch.testing.assert_close(lse, logs, rtol=0.001, atol=0.001)
        native.init_graphs([3], 3, q.device)
        captured_meta = cache.decode_metadata([0, 1, 2], [cp] * 3, table)
        captured_meta.capture_size = 3
        for _ in range(2):
            native.decode(
                q, cache.paged_storage(1), weight, captured_meta, nope_dim=128, value_dim=128
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=producer):
            captured, captured_lse = native.decode(
                q, cache.paged_storage(1), weight, captured_meta, nope_dim=128, value_dim=128
            )
        for iteration, order in enumerate(([1, 2, 0], [2, 0, 1], [0, 1, 2])):
            if iteration == 1 and arena is not None:
                arena.free_pages_by_id([31])
            q.neg_()
            meta = cache.decode_metadata(order, [lengths[row] for row in order], table)
            native.prepare_replay(meta, 3)
            graph.replay()
            expected, logs = native.decode(
                q, cache.paged_storage(1), weight, meta, nope_dim=128, value_dim=128
            )
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            torch.testing.assert_close(captured_lse, logs, rtol=0, atol=0)
            oracle, oracle_lse = expected_partial(q, weight, cache, table, lengths, order, cp, rank)
            torch.testing.assert_close(captured, oracle, rtol=0.03, atol=0.002)
            torch.testing.assert_close(captured_lse, oracle_lse, rtol=0.001, atol=0.001)
        graph.reset()
        # Every request can be empty on a rank during very short DCP decode.
        short_lengths = [1, 1, 1]
        meta = cache.decode_metadata([0, 1, 2], short_lengths, table)
        output, lse = native.decode(
            q, cache.paged_storage(1), weight, meta, nope_dim=128, value_dim=128
        )
        expected, logs = expected_partial(
            q, weight, cache, table, short_lengths, [0, 1, 2], cp, rank
        )
        torch.testing.assert_close(output, expected, rtol=0.03, atol=0.002)
        torch.testing.assert_close(lse, logs, rtol=0.001, atol=0.001)
    torch.cuda.current_stream().wait_stream(producer)
    torch.cuda.synchronize()
    request.node.user_properties.append(("graph_replays", 3))
