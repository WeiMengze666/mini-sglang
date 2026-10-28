"""Paged MLA must follow request/page indirection, including empty DCP shards."""

import os

import minisgl.distributed.info as info
import minisgl.distributed.layout as layout
import pytest
import torch
from minisgl.attention.mla import mla_attention
from minisgl.attention.mla_fi import FlashInferMLA
from minisgl.distributed import DistributedInfo
from minisgl.kvcache.mla_pool import MLACache
from minisgl.kvcache.unified import UnifiedMemory

CUDA = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


def make_cache(monkeypatch, cp, unified, device, *, native=False, long=False):
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(cp - 1, cp))
    monkeypatch.setattr(layout, "_DCP_SIZE", cp)
    page_size, layers, width = 64, (1, 3), 576
    lengths = [4099, 1025, 1] if long else [131, 70, 1]
    num_pages = 128 if long else 64
    page_lists = ([17, 3, 29], [11, 5], [27])
    if long:
        available = [page for page in reversed(range(num_pages)) if page != 31]
        counts = [(length + page_size - 1) // page_size for length in lengths]
        page_lists, start = [], 0
        for count in counts:
            page_lists.append(available[start : start + count])
            start += count
    arena = (
        UnifiedMemory(num_pages, len(layers) * page_size // cp * width * 2, 256, device)
        if unified
        else None
    )
    if arena is not None:
        arena.allocate_pages([31] + [page for pages in page_lists for page in pages])
    cache = MLACache(
        layers,
        num_pages * page_size,
        512,
        64,
        torch.bfloat16,
        device,
        arena=arena,
        page_size=page_size,
    )
    max_length = (max(lengths) + page_size - 1) // page_size * page_size
    table = torch.zeros((3, max_length), dtype=torch.int32, device=device)
    for row, pages in enumerate(page_lists):
        for index, page in enumerate(pages):
            table[row, index * page_size : (index + 1) * page_size] = (
                page * page_size + torch.arange(page_size, device=device)
            )
        for layer in layers:
            data = torch.randn(lengths[row], width, dtype=torch.bfloat16, device=device) * 0.1
            cache.store_mla(
                layer,
                data[:, :512],
                data[:, 512:],
                table[row, : lengths[row]],
                implementation="triton" if native else "torch",
            )
            if native:
                actual = cache.gather_mla(layer, table[row, cp - 1 : lengths[row] : cp])
                torch.testing.assert_close(
                    torch.cat(actual, -1), data[cp - 1 :: cp], rtol=0, atol=0
                )
    return cache, table, lengths


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
@pytest.mark.parametrize("unified", [False, True])
def test_page_metadata_matches_gather_after_compaction(monkeypatch, cp, unified):
    cache, table, lengths = make_cache(monkeypatch, cp, unified, "cpu")
    order = [2, 0, 1]
    for compact in (False, True):
        if compact and unified:
            cache.arena.free_pages_by_id([31])
        meta = cache.decode_metadata(order, [lengths[row] for row in order], table)
        for layer in (1, 3):
            latent, key = cache.paged_layer(layer)
            for index, row in enumerate(order):
                ids = meta.indices[meta.indptr[index] : meta.indptr[index + 1]].long()
                count = meta.lengths[index]
                actual = (latent[ids].flatten(0, 1)[:count], key[ids].flatten(0, 1)[:count])
                expected = cache.gather_mla(layer, table[row, cp - 1 : lengths[row] : cp])
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            # Both split tensors must be views into the original allocation.
            storage = cache.arena.buffer if unified else cache._mla
            assert latent.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
            assert key.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()


@CUDA
@pytest.mark.parametrize("heads", [12, 24])
@pytest.mark.parametrize("cp", [1, 2, 4, 8])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("long", [False, True])
def test_paged_decode_matches_independent_sequences(monkeypatch, heads, cp, unified, long):
    torch.manual_seed(173)
    cache, table, lengths = make_cache(monkeypatch, cp, unified, "cuda", native=True, long=long)
    native = FlashInferMLA()
    weight = torch.randn(heads * 320, 512, dtype=torch.bfloat16, device="cuda") * 0.05
    for iteration, order in enumerate(([2, 0, 1], [1, 2, 0])):
        if iteration and unified:
            cache.arena.free_pages_by_id([31])
        meta = cache.decode_metadata(order, [lengths[row] for row in order], table)
        q = torch.randn(3, heads, 256, dtype=torch.bfloat16, device="cuda") * 0.1
        for layer in (1, 3):
            expected = []
            for index, row in enumerate(order):
                positions = torch.arange(cp - 1, max(cp - 1, lengths[row]), cp, device="cuda")
                latent, key = cache.gather_mla(layer, table[row, cp - 1 : lengths[row] : cp])
                expected.append(
                    mla_attention(
                        q[index : index + 1],
                        latent,
                        key,
                        weight,
                        torch.tensor([lengths[row] - 1], device="cuda"),
                        positions,
                        nope_dim=192,
                        value_dim=128,
                        return_lse=True,
                    )
                )
            actual, lse = native.decode(
                q,
                cache.paged_layer(layer),
                weight,
                meta,
                nope_dim=192,
                value_dim=128,
                return_lse=True,
            )
            values, logs = (torch.cat(parts) for parts in zip(*expected))
            torch.testing.assert_close(actual, values, rtol=0.03, atol=0.002)
            torch.testing.assert_close(lse, logs, rtol=0.001, atol=0.001)


@CUDA
@pytest.mark.parametrize("cp", [1, 2, 4, 8])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("long", [False, True])
def test_paged_decode_graph_replans_after_compaction(monkeypatch, cp, unified, long):
    torch.manual_seed(179)
    cache, table, lengths = make_cache(monkeypatch, cp, unified, "cuda", native=True, long=long)
    native = FlashInferMLA()
    native.init_graphs([3], table.shape[1] // cache.page_size, torch.device("cuda"))
    q = torch.randn(3, 12, 256, dtype=torch.bfloat16, device="cuda") * 0.1
    weight = torch.randn(12 * 320, 512, dtype=torch.bfloat16, device="cuda") * 0.05
    captured = cache.decode_metadata([0, 1, 2], [cp] * 3, table)
    captured.capture_size = 3
    captured.nonempty = captured.lengths.to("cuda") > 0

    def run(meta):
        return native.decode(
            q, cache.paged_layer(3), weight, meta, nope_dim=192, value_dim=128, return_lse=True
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(captured)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output, lse = run(captured)
        for iteration, order in enumerate(([2, 0, 1], [1, 2, 0])):
            if iteration and unified:
                cache.arena.free_pages_by_id([31])
            q.copy_(torch.randn_like(q) * 0.1)
            meta = cache.decode_metadata(order, [lengths[row] for row in order], table)
            captured.nonempty.copy_(meta.lengths.to("cuda") > 0)
            native.prepare_replay(meta, 3)
            graph.replay()
            # Execute the same re-planned kernel eagerly. Planning may choose a
            # different split reduction for a non-graph wrapper; that numerical
            # comparison belongs to the independent attention oracle above.
            expected, logs = run(meta)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(lse, logs, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)
