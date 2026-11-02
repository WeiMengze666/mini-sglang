"""Weight-aware expert dispatch must preserve the portable mixture contract."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.linear import Linear
from minisgl.moe import TorchExpertExecutor, create_expert_executor


def test_linear_formats_preserve_checkpoint_fields_and_dispatch():
    dense = Linear(32, 4)
    packed = Linear(32, 4, weight_format="mxfp4")
    assert set(dense.state_dict()) == {"weight"}
    assert set(packed.state_dict()) == {"weight_packed", "weight_scale"}
    assert dense.weight.dtype.is_floating_point
    assert packed.weight_packed.dtype == packed.weight_scale.dtype == torch.uint8
    with pytest.raises(ValueError, match="Unsupported linear weight format"):
        Linear(32, 4, weight_format="uint8")


def test_portable_mixture_keeps_global_expert_ids_and_ignores_remote_routes():
    def expert(factor):
        return SimpleNamespace(
            w1=Linear(2, 2),
            w3=Linear(2, 2),
            w2=Linear(2, 2),
            forward=lambda x: x * factor,
        )

    executor = create_expert_executor([(2, expert(2)), (7, expert(3))], activation="silu")
    assert isinstance(executor, TorchExpertExecutor)
    assert not executor.supports_cuda_graph
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    ids = torch.tensor([[2, 7], [99, 2]])
    torch.testing.assert_close(
        executor.forward(x, weights, ids), torch.tensor([[2.75, 5.5], [3.0, 4.0]]), rtol=0, atol=0
    )


def test_empty_expert_partition_returns_a_zero_local_mixture():
    executor = create_expert_executor([], activation="silu")
    x = torch.ones(2, 3, dtype=torch.bfloat16)
    output = executor.forward(x, torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long))
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.zeros(2, 3), rtol=0, atol=0)
