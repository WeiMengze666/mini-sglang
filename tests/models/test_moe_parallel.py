"""EP contracts for the existing stacked MoE path, independent of Kimi models."""

import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.distributed.partition import Partition


@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_standard_checkpoint_expert_selection_and_tensor_slices(tmp_path, monkeypatch, ep_size):
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layouts
    import minisgl.models.weight as loader
    from minisgl.models.config import ModelConfig
    from safetensors.torch import save_file

    torch.manual_seed(19)
    state = {}
    for expert in range(4):
        for name in ("gate_proj", "up_proj", "down_proj"):
            shape = (8, 12) if name == "down_proj" else (12, 8)
            state[f"model.layers.0.mlp.experts.{expert}.{name}.weight"] = torch.randn(shape)
    state["model.layers.0.mlp.gate.weight"] = torch.randn(4, 8)
    save_file(state, tmp_path / "weights.safetensors")
    config = SimpleNamespace(is_kimi=False, is_moe=True, num_experts=4, num_kv_heads=2)
    monkeypatch.setattr(loader, "cached_load_hf_config", lambda _: None)
    monkeypatch.setattr(loader, "download_hf_weight", lambda _: str(tmp_path))
    monkeypatch.setattr(ModelConfig, "from_hf", lambda _: config)
    monkeypatch.setattr(layouts, "_EP_SIZE", ep_size)
    for rank in range(4):
        monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 4))
        weights = dict(loader.load_weight(str(tmp_path), torch.device("cpu")))
        layout = layouts.get_parallel_layout()
        gate, down = [], []
        for expert in layout.expert_range(4):
            prefix = f"model.layers.0.mlp.experts.{expert}."
            shard = layout.expert_tensor.shard(0, 12)
            gate.append(
                torch.cat(
                    [shard.slice(state[prefix + n + ".weight"]) for n in ("gate_proj", "up_proj")]
                )
            )
            down.append(layout.expert_tensor.shard(1, 12).slice(state[prefix + "down_proj.weight"]))
        torch.testing.assert_close(
            weights["model.layers.0.mlp.experts.gate_up_proj"], torch.stack(gate)
        )
        torch.testing.assert_close(
            weights["model.layers.0.mlp.experts.down_proj"], torch.stack(down)
        )
        torch.testing.assert_close(
            weights["model.layers.0.mlp.gate.weight"], state["model.layers.0.mlp.gate.weight"]
        )


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
@pytest.mark.parametrize("tokens,remote_only", [(1, True), (3, True), (65, False)])
def test_stacked_fused_experts_ep_and_expert_tensor_parallel(tokens, remote_only):
    from minisgl.moe.fused import FusedMoe

    torch.manual_seed(47)
    x = torch.randn(tokens, 32, dtype=torch.bfloat16, device="cuda") * 0.2
    gate_up = torch.randn(8, 128, 32, dtype=x.dtype, device=x.device) * 0.1
    down = torch.randn(8, 32, 64, dtype=x.dtype, device=x.device) * 0.1
    logits = torch.randn(tokens, 8, device=x.device)
    # At least one rank sees an entirely remote selection, including M > block size.
    if remote_only:
        logits[:, :2] += 20
    backend = FusedMoe()
    expected = backend.forward(x.clone(), gate_up, down, logits, 2, True)
    actual = torch.zeros_like(x, dtype=torch.float32)
    for expert_rank in range(4):
        experts = Partition(expert_rank, 4)
        owned = experts.range(8)
        for tensor_rank in range(2):
            split = Partition(tensor_rank, 2)
            gate, up = gate_up[owned.start : owned.stop].chunk(2, 1)
            local_gate = torch.cat(
                [split.shard(1, 64).slice(w) for w in (gate, up)], 1
            ).contiguous()
            local_down = split.shard(2, 64).slice(down[owned.start : owned.stop]).contiguous()
            output = backend.forward(
                x.clone(),
                local_gate,
                local_down,
                logits,
                2,
                True,
                expert_partition=experts,
                num_global_experts=8,
            )
            if remote_only and expert_rank > 0:
                assert output.eq(0).all()
            actual += output.float()
    torch.testing.assert_close(actual, expected.float(), rtol=0.04, atol=0.002)


def _distributed_worker(rank, init_file):
    from datetime import timedelta

    import minisgl.distributed.info as info
    import torch.distributed as dist
    from minisgl.core import Context, set_global_ctx
    from minisgl.distributed.layout import configure_parallel, get_parallel_layout
    from minisgl.layers.moe import MoELayer
    from minisgl.moe.fused import FusedMoe

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + init_file,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    info._TP_INFO = DistributedInfo(rank, 4)
    ctx = Context(1)
    ctx.moe_backend = FusedMoe()
    set_global_ctx(ctx)
    try:
        torch.manual_seed(79)
        x = torch.randn(17, 32, device="cuda", dtype=torch.bfloat16) * 0.2
        gate = torch.randn(8, 64, 32, device=x.device, dtype=x.dtype) * 0.1
        up = torch.randn_like(gate) * 0.1
        down = torch.randn(8, 32, 64, device=x.device, dtype=x.dtype) * 0.1
        logits = torch.randn(17, 8, device=x.device)
        expected = ctx.moe_backend.forward(
            x.clone(), torch.cat((gate, up), 1), down, logits, 2, True
        )
        for ep in (1, 2, 4):
            configure_parallel(ep_size=ep)
            layout = get_parallel_layout()
            owned = layout.expert_range(8)
            channels = layout.expert_tensor
            with torch.device("cuda"):
                layer = MoELayer(8, 2, 32, 64)
            layer.gate_up_proj = torch.cat(
                [channels.shard(1, 64).slice(w[owned.start : owned.stop]) for w in (gate, up)], 1
            ).contiguous()
            layer.down_proj = (
                channels.shard(2, 64).slice(down[owned.start : owned.stop]).contiguous()
            )
            output = layer.forward(x.clone(), logits)
            torch.testing.assert_close(output, expected, rtol=0.04, atol=0.002)
    finally:
        info._TP_INFO = None
        configure_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
def test_existing_moe_layer_four_rank_nccl(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_worker, args=(str(tmp_path / "moe-group"),), nprocs=4, join=True)
