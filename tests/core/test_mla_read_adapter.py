"""Caller-owned MLA reads preserve values, row order, DCP ownership and cache state."""

from contextlib import nullcontext

import minisgl.distributed.info as info
import minisgl.distributed.layout as layout
import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.kernel import mla as adapter
from minisgl.kvcache.mla_pool import MLACache
from minisgl.kvcache.unified import UnifiedMemory


@pytest.mark.parametrize("cp", [1, 2, 8])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("dims", [(4, 2), (4, 0), (3, 2)])
@pytest.mark.parametrize("empty", [False, True])
def test_caller_output_preserves_values_order_and_cache(monkeypatch, cp, unified, dims, empty):
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(cp - 1, cp))
    monkeypatch.setattr(layout, "_DCP_SIZE", cp)
    width = sum(dims)
    arena = UnifiedMemory(2, 2 * 64 // cp * width, 256, "cpu") if unified else None
    if arena is not None:
        arena.allocate_pages([1, 0])
    cache = MLACache((1, 7), 128, *dims, torch.float8_e4m3fn, "cpu", arena=arena, page_size=64)
    positions = torch.arange(128)
    values = (torch.arange(128 * width).reshape(128, width) % 31 - 15).bfloat16() / 8
    for layer in (1, 7):
        payload = values + layer
        if arena is None:
            # Torch has no FP8 index_copy_cpu; initialize the read fixture directly.
            cache._mla[cache._full_layers[layer]].copy_(payload[cp - 1 :: cp])
        else:
            cache.store_mla(layer, payload[:, : dims[0]], payload[:, dims[0] :], positions)
    backing = cache._mla if arena is None else arena.buffer
    before = backing.view(torch.uint8).clone()
    rows = [] if empty else [7, 1, 7, 0, 9]
    ids = torch.tensor([row * cp + cp - 1 for row in rows], dtype=torch.int64)
    # The logical indices are non-contiguous, non-monotone and may repeat.
    ids = torch.stack((ids, torch.zeros_like(ids)), -1).flatten()[::2]
    padded = torch.full((len(ids), width + 3), -99, dtype=torch.bfloat16)
    out = padded[:, :width]
    result = cache.gather_mla(7, ids, out=out)
    expected = (values + 7).to(torch.float8_e4m3fn).bfloat16()[ids]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert all(
        value.untyped_storage().data_ptr() == out.untyped_storage().data_ptr() for value in result
    )
    assert all(value.stride(0) == width + 3 for value in result)
    torch.testing.assert_close(padded[:, width:], torch.full_like(padded[:, width:], -99))
    assert torch.equal(backing.view(torch.uint8), before)


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("invalid", ["negative", "range", "owner", "cache_alias"])
def test_invalid_read_rejected_before_native_dispatch(monkeypatch, invalid, unified):
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(1, 2))
    monkeypatch.setattr(layout, "_DCP_SIZE", 2)
    arena = UnifiedMemory(2, 2 * 32 * 6 * 2, 256, "cpu") if unified else None
    if arena is not None:
        arena.allocate_pages([1, 0])
    cache = MLACache((1, 7), 128, 4, 2, torch.bfloat16, "cpu", arena=arena, page_size=64)
    backing = cache._mla if arena is None else arena.buffer
    ids = torch.tensor([1, 5])
    out = torch.empty(2, 6, dtype=torch.bfloat16)
    if invalid == "negative":
        ids[0] = -1
    elif invalid == "range":
        ids[0] = 129
    elif invalid == "owner":
        ids[0] = 0
    else:
        # A different layer still aliases this cache's entire backing allocation.
        out = cache._mla[0, :2] if arena is None else cache._page_storage()[0, 0, :2]
    monkeypatch.setattr(adapter, "try_gather_mla", lambda *args: pytest.fail("native dispatch"))
    before = backing.clone()
    with pytest.raises((ValueError, IndexError)):
        cache.gather_mla(7, ids, out=out)
    assert torch.equal(backing, before)


class CpuCudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("rope_dim", [0, 2])
def test_native_adapter_forwards_views_and_contiguous_indices(monkeypatch, rope_dim):
    width = 4 + rope_dim
    source = torch.arange(16 * width).reshape(16, width).to(torch.float8_e4m3fn)
    source = source.as_subclass(CpuCudaTensor)
    indices = torch.tensor([7, 0, 1, 0, 7, 0])[::2].as_subclass(CpuCudaTensor)
    combined = torch.empty(3, width, dtype=torch.bfloat16).as_subclass(CpuCudaTensor)
    latent, key = combined[:, :4], combined[:, 4:]
    calls = []

    def native(buffer, loc, nope, rope):
        calls.append((buffer, loc, nope, rope))
        assert loc.is_contiguous()
        assert nope.data_ptr() == combined.data_ptr()
        assert rope.data_ptr() == combined[:, 4:].data_ptr()
        assert nope.stride(0) == rope.stride(0) == width
        # CPU emulation is an ABI spy, not native mathematical qualification.
        selected = buffer.index_select(0, loc).bfloat16()
        nope.copy_(selected[:, :4])
        rope.copy_(selected[:, 4:])

    monkeypatch.setattr(adapter, "_gather_kernels", lambda: (native, nullcontext))
    assert adapter.try_gather_mla(source, indices, latent, key)
    assert len(calls) == 1
    torch.testing.assert_close(combined, source.index_select(0, indices).bfloat16(), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="native failure"):
        monkeypatch.setattr(
            adapter,
            "_gather_kernels",
            lambda: (
                lambda *args: (_ for _ in ()).throw(RuntimeError("native failure")),
                nullcontext,
            ),
        )
        adapter.try_gather_mla(source, indices, latent, key)


def test_cpu_read_declines_before_sdk_import(monkeypatch):
    monkeypatch.setattr(adapter, "_gather_kernels", lambda: pytest.fail("SDK import"))
    for rows in (0, 3):
        source = torch.zeros(8, 6, dtype=torch.float8_e4m3fn)
        output = torch.empty(rows, 6, dtype=torch.bfloat16)
        assert not adapter.try_gather_mla(source, torch.arange(rows), output[:, :4], output[:, 4:])
