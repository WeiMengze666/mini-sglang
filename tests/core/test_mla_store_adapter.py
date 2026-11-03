"""Mature cache kernel ABI with cyclic owners, page permutation and byte-page gaps."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.kernel import mla as adapter


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_writer_preserves_other_layers_and_page_padding(
    monkeypatch, cp_size, indirect, dtype
):
    cp_rank, page_size = cp_size - 1, 8
    local_size, width, layers = page_size // cp_size, 6, 2
    page_stride = layers * local_size * width + 5
    backing = torch.zeros(3, page_stride, dtype=dtype)
    all_layers = backing.as_strided(
        (3, layers, local_size, width), (page_stride, local_size * width, width, 1)
    )
    storage = all_layers[:, 1]
    mapping = torch.tensor([2, 0, 1]) if indirect else None
    locations = torch.tensor([0, 1, 3, 7, 8, 17], dtype=torch.int64)
    latent = (torch.arange(len(locations) * 4).reshape(len(locations), 4) / 16).bfloat16()
    key = (-torch.arange(len(locations) * 2).reshape(len(locations), 2) / 16).bfloat16()
    expected = backing.clone()
    expected_layers = expected.as_strided(all_layers.shape, all_layers.stride())
    for row, position in enumerate(locations.tolist()):
        if position % cp_size == cp_rank:
            page = position // page_size
            physical = int(mapping[page]) if mapping is not None else page
            offset = (position % page_size) // cp_size
            expected_layers[physical, 1, offset] = torch.cat((latent[row], key[row])).to(dtype)
    seen = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def invoke(buffer, nope, rope, indices, skip, stride, ns, rs, nd, rd, **options):
                assert grid[0] == len(locations) and stride == 1 and skip == -1
                assert (ns, rs, nd, rd) == (latent.stride(0), key.stride(0), 4, 2)
                flat = buffer.as_strided((backing.numel(),), (1,), storage_offset=0)
                for row, offset in enumerate(indices.tolist()):
                    if offset != skip:
                        start = buffer.storage_offset() + offset
                        flat[start : start + width] = torch.cat((nope[row], rope[row])).to(dtype)
                seen.append(self.name)

            return invoke

    sdk = SimpleNamespace(
        set_mla_kv_buffer_kernel=Kernel("plain"),
        set_mla_kv_buffer_fp8_quant_kernel=Kernel("fp8"),
    )
    monkeypatch.setattr(adapter, "_kernels", lambda: (sdk, False, nullcontext))
    adapter.store_mla(
        latent,
        key,
        locations,
        mapping,
        storage,
        page_size=page_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )
    assert seen == ["fp8" if dtype == torch.float8_e4m3fn else "plain"]
    torch.testing.assert_close(backing.float(), expected.float(), rtol=0, atol=0)
