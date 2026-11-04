"""Native MLA ownership/precision ABI without executing GPU work."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mla.cutedsl import CuTeMLADecodePlan


def metadata(lengths, pages, *, cp=1, rank=0):
    local = torch.tensor(lengths, dtype=torch.int32)
    return SimpleNamespace(
        block_table=torch.tensor(pages, dtype=torch.int32),
        local_lengths=local,
        global_lengths=local * cp,
        page_size=64,
        max_seq_len=int(local.max()) * cp,
        cp_size=cp,
        cp_rank=rank,
    )


def test_graph_metadata_buffers_keep_addresses_and_clear_old_pages():
    plan = CuTeMLADecodePlan(torch.empty(1, dtype=torch.uint8), batch_size=2, max_pages=3)
    first = metadata([129, 65], [[7, 2, 9], [4, 8, 0]], cp=4, rank=3)
    plan.prepare(first, {})
    buffers = tuple(t.data_ptr() for t in (plan.blocks, plan.local, plan.global_))
    second = metadata([1, 0], [[6], [0]], cp=4, rank=3)
    plan.prepare(second, {})
    assert tuple(t.data_ptr() for t in (plan.blocks, plan.local, plan.global_)) == buffers
    assert plan.blocks.tolist() == [[6, 0, 0, 0], [0, 0, 0, 0]]
    assert plan.local.tolist() == [1, 0] and plan.global_.tolist() == [4, 0]
    with pytest.raises(ValueError, match="capacity"):
        plan.prepare(metadata([256, 256], [[1, 2, 3, 4]] * 2), {})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("cp", [1, 8])
def test_native_call_preserves_storage_and_uses_matching_query_precision(monkeypatch, dtype, cp):
    calls, joined = [], []

    def concat(q, suffix):
        joined.append(True)
        return torch.cat((q, suffix), -1)

    def run(**options):
        calls.append(options)
        q = options["query"]
        output = torch.ones((*q.shape[:-1], 512), dtype=torch.bfloat16)
        lse = torch.ones(q.shape[:-1], dtype=torch.float32)
        output[1].fill_(float("nan"))
        lse[1].fill_(float("nan"))
        return output, lse

    monkeypatch.setitem(
        sys.modules, "flashinfer.decode", SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=run)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.attention.utils",
        SimpleNamespace(concat_mla_absorb_q_general=concat),
    )
    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(use_torch_stream=nullcontext))
    # Physical pages can have alignment gaps; the adapter must preserve strides.
    storage = torch.empty((5, 64, 580), dtype=dtype)
    pages = storage[..., :576]
    plan = CuTeMLADecodePlan(torch.empty(1, dtype=torch.uint8))
    meta = metadata([65, 0], [[3, 1], [0, 0]], cp=cp, rank=cp - 1)
    plan.prepare(meta, {})
    q = torch.linspace(-2, 2, 2 * 12 * 512).reshape(2, 12, 512).bfloat16()
    suffix = torch.linspace(-1, 1, 2 * 12 * 64).reshape(2, 12, 64).bfloat16()
    saved = q.clone(), suffix.clone()
    output, lse = plan.run((q, suffix), pages, nope_dim=128, scale=192**-0.5)
    assert len(calls) == len(joined) == 1
    call = calls[0]
    assert call["kv_cache"].data_ptr() == pages.data_ptr()
    assert call["kv_cache"].stride()[-3:] == (pages.stride(0), *pages.stride()[1:])
    assert call["query"].dtype == call["kv_cache"].dtype == dtype
    torch.testing.assert_close(
        call["query"].squeeze(1).float(),
        torch.cat((q, suffix), -1).to(dtype).float(),
        rtol=0,
        atol=0,
    )
    assert call["backend"] == "cute-dsl" and call["return_lse_base"] == "basee"
    assert call["cp_world"] == cp and call["cp_rank"] == cp - 1
    assert call["enable_dcp"] == (cp > 1)
    assert call["causal_seqlens_kv_global"] is (meta.global_lengths if cp > 1 else None)
    assert call["block_tables"] is meta.block_table
    assert call["seq_lens"] is meta.local_lengths
    assert call["bmm1_scale"] == 192**-0.5
    assert output[1].eq(0).all() and lse[1].eq(-torch.inf).all()
    torch.testing.assert_close(q, saved[0], rtol=0, atol=0)
    torch.testing.assert_close(suffix, saved[1], rtol=0, atol=0)


@pytest.mark.parametrize("unified", [False, True])
def test_dense_metadata_keeps_dummy_column_on_empty_dcp_rank(monkeypatch, unified):
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout
    from minisgl.distributed import DistributedInfo
    from minisgl.kvcache.mla_pool import MLACache
    from minisgl.kvcache.unified import UnifiedMemory

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(7, 8))
    monkeypatch.setattr(layout, "_DCP_SIZE", 8)
    arena = UnifiedMemory(4, 64 * 576 * 2, 256, "cpu") if unified else None
    cache = MLACache((0,), 4 * 512, 512, 64, torch.bfloat16, "cpu", arena=arena, page_size=512)
    # No token/page has been allocated on this rank. A zero local length makes
    # the dense placeholder safe, even when the arena's page map contains -1.
    table = torch.zeros((3, 512), dtype=torch.int32)
    meta = cache.decode_metadata([2, 0, 1], [7, 1, 3], table)
    assert meta.block_table.shape == (3, 1) and meta.block_table.eq(0).all()
    assert meta.local_lengths.eq(0).all() and not len(meta.indices)
    assert meta.indptr.tolist() == [0, 0, 0, 0]
    assert meta.global_lengths.tolist() == [7, 1, 3]
    assert meta.max_seq_len == 7 and meta.cp_rank == 7 and meta.cp_size == 8


@pytest.mark.parametrize("width", [1, 3, 4])
def test_native_dense_page_table_alignment_does_not_change_logical_lengths(width):
    plan = CuTeMLADecodePlan(torch.empty(1, dtype=torch.uint8))
    meta = metadata([1, 0], [list(range(width)), [0] * width])
    original = meta.block_table.clone()
    plan.prepare(meta, {})
    expected_width = (width + 1) // 2 * 2
    assert plan.blocks.shape == (2, expected_width)
    torch.testing.assert_close(plan.blocks[:, :width], original, rtol=0, atol=0)
    assert plan.blocks[:, width:].eq(0).all()
    assert plan.local is meta.local_lengths and plan.global_ is meta.global_lengths
    torch.testing.assert_close(meta.block_table, original, rtol=0, atol=0)


@pytest.mark.parametrize("cp,rank", [(1, 0)] + [(8, rank) for rank in range(8)])
@pytest.mark.parametrize("global_length", [1, 7, 513, 1048575])
@pytest.mark.parametrize("graph", [False, True])
def test_native_scheduling_bound_is_local_and_positive(monkeypatch, cp, rank, global_length, graph):
    calls = []

    def run(**options):
        calls.append(options)
        q = options["query"]
        return torch.zeros((*q.shape[:-1], 512)), torch.zeros(q.shape[:-1])

    monkeypatch.setitem(
        sys.modules, "flashinfer.decode", SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=run)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.attention.utils",
        SimpleNamespace(concat_mla_absorb_q_general=lambda q, suffix: torch.cat((q, suffix), -1)),
    )
    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(use_torch_stream=nullcontext))
    local_length = global_length // cp + int(rank < global_length % cp)
    local_pages = (local_length + 63) // 64
    meta = metadata([local_length], [[0] * max(local_pages, 1)], cp=cp, rank=rank)
    meta.global_lengths.fill_(global_length)
    meta.max_seq_len = global_length
    capacity = (1048576 + 64 * cp - 1) // (64 * cp)
    plan = CuTeMLADecodePlan(
        torch.empty(1, dtype=torch.uint8),
        **({"batch_size": 1, "max_pages": capacity} if graph else {}),
    )
    plan.prepare(meta, {})
    plan.run(
        (torch.zeros(1, 12, 512), torch.zeros(1, 12, 64)),
        torch.empty(1, 64, 576),
        nope_dim=128,
        scale=192**-0.5,
    )
    call = calls[0]
    assert call["max_seq_len"] == (capacity * 64 if graph else max(local_length, 1))
    assert call["seq_lens"].tolist() == [local_length]
    if cp > 1:
        assert call["causal_seqlens_kv_global"].tolist() == [global_length]
    else:
        assert call["causal_seqlens_kv_global"] is None
