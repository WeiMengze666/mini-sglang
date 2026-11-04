"""Native pool views preserve functional K,V state and immutable checkpoints."""

import pytest
import torch
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.kvcache.unified import UnifiedMemory


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layer_id", [2, 7])
def test_native_views_update_only_the_selected_owner_and_restore_private_state(
    unified, dtype, layer_id
):
    arena = UnifiedMemory(32, 256, 1024, "cpu") if unified else None
    pool = RecurrentStatePool(
        (2, 7),
        4,
        2,
        3,
        3,
        torch.bfloat16,
        "cpu",
        max_cached_states=4,
        state_dtype=dtype,
        arena=arena,
    )
    pool.begin_request(3, 10, 0)
    indices = pool.batch_indices([3])
    logical = pool.get_kda_state(layer_id, 3)
    conv = torch.arange(logical.conv.numel()).reshape(logical.conv.shape).to(logical.conv.dtype)
    recurrent = torch.arange(logical.recurrent.numel()).reshape(logical.recurrent.shape).to(dtype)
    logical.conv.copy_(conv)
    logical.recurrent.copy_(recurrent)
    native_conv, native_state = pool.indexed_kda_state(layer_id)
    row = int(indices[0])
    assert native_conv.stride(1) == 1
    assert native_state.stride()[-3:] == (9, 3, 1)
    assert native_state[row].data_ptr() == logical.recurrent.data_ptr()
    torch.testing.assert_close(native_conv[row], conv.flatten(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(native_state[row], recurrent.transpose(-1, -2), rtol=0, atol=0)
    pool.finish_request(3, 10, 13)
    checkpoint = pool.snapshot(3, 10, expected_len=13)
    pool.begin_request(3, 10, 13)
    native_conv[row].fill_(-5)
    native_state[row].fill_(-7)
    pool.finish_request(3, 10, 14)
    pool.release_request(3, 10)
    pool.restore(1, 11, checkpoint)
    pool.begin_request(1, 11, 13)
    restored = pool.get_kda_state(layer_id, 1)
    torch.testing.assert_close(restored.conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(restored.recurrent, recurrent, rtol=0, atol=0)
    pool.finish_request(1, 11, 14)
    pool.release_request(1, 11)
    pool.drop_checkpoint(checkpoint)
    assert pool.is_idle
    if arena is not None:
        arena.check_integrity()
