"""Mechanism layers must work without a model-family config or global Context."""

import pytest
import torch
from minisgl.attention.config import KDAConfig, MLAConfig
from minisgl.attention.kda import KDAState
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.layers.kda import KDALayer
from minisgl.layers.mla import MLALayer


@pytest.mark.parametrize(
    "layer",
    [
        lambda: KDALayer(KDAConfig(12, 2, 4, 3, full_rank_gate=True, gate_lower_bound=-5)),
        lambda: MLALayer(MLAConfig(12, 2, 3, 4, 3, 5, 6, 32, output_gate=True)),
    ],
)
def test_mechanism_config_streaming(layer):
    torch.manual_seed(67)
    op = layer()
    for name, tensor in op.state_dict().items():
        tensor.normal_(std=0.1)
        if "norm.weight" in name:
            tensor.fill_(1)
    x, positions = torch.randn(7, 12), torch.arange(7)
    expected, expected_state = op.forward(x, positions)
    outputs, state = [], None
    for start, stop in ((0, 3), (3, 6), (6, 7)):
        output, state = op.forward(x[start:stop], positions[start:stop], state)
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs), expected, atol=3e-6, rtol=3e-5)
    for actual, reference in zip(state, expected_state):
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-5)


def test_cache_layout_is_independent_of_checkpoint_schedule_conventions():
    layout = HybridCacheConfig(3, (1,), (0, 2), 5, 2, 2, 3, 4)
    cache = HybridCache(layout, 3, 4, 2, torch.float32, torch.device("cpu"))
    assert cache.allocated_bytes == (
        12 * layout.cache_bytes_per_token(torch.float32)
        + 2 * layout.state_bytes_per_request(torch.float32)
    )
    cache.begin_request(0, 101, 0)
    state = cache.get_kda_state(2, 0)
    state.recurrent.fill_(7)
    cache.finish_request(0, 101, 4)
    cache.release_request(0, 100)
    cache.begin_request(0, 101, 4)
    assert state.recurrent.eq(7).all()
    cache.release_request(0, 101)
    cache.begin_request(0, 102, 0)
    assert not cache.get_kda_state(2, 0).recurrent.any()
    with pytest.raises(ValueError, match="partition"):
        HybridCacheConfig(3, (1,), (0, 1), 5, 2, 2, 2, 4)


@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("width,bound", [(1, None), (3, -5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kda_batched_updates_preserve_independent_histories(tokens, width, bound, dtype):
    torch.manual_seed(151)
    layer = KDALayer(KDAConfig(12, 2, 4, width, gate_lower_bound=bound))
    for tensor in layer.state_dict().values():
        tensor.normal_(std=0.1)
    # Distinct, nonsymmetric histories expose request or K/V axis confusion.
    initial = KDAState(torch.randn(3, 3, 8, width - 1).to(dtype), torch.randn(3, 2, 4, 4))
    saved = KDAState(*(part.clone() for part in initial))
    projected = [torch.randn(3, tokens, 8).to(dtype) for _ in range(3)]
    projected += [torch.randn(3, tokens, 2, 4).to(dtype), torch.randn(3, tokens, 2).to(dtype)]
    expected = [
        layer.update(
            *(part[index] for part in projected),
            KDAState(*(part[index] for part in initial)),
        )
        for index in range(3)
    ]
    actual, final = layer.update(*projected, initial)
    for index, (output, state) in enumerate(expected):
        torch.testing.assert_close(actual[index], output, rtol=1e-5, atol=1e-6)
        for batched, independent in zip(final, state):
            torch.testing.assert_close(batched[index], independent, rtol=1e-5, atol=1e-6)
    # Reordering a decode batch must follow its states, not the old row order.
    order = torch.tensor([2, 0, 1])
    shuffled, _ = layer.update(
        *(part[order] for part in projected), KDAState(*(part[order] for part in initial))
    )
    torch.testing.assert_close(shuffled, actual[order], rtol=1e-5, atol=1e-6)
    for current, original in zip(initial, saved):
        torch.testing.assert_close(current, original, rtol=0, atol=0)
