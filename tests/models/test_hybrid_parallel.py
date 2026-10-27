"""Real collective gates: DCP partial attention, EP ownership and prefix restore."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.distributed import DistributedInfo
from minisgl.distributed.context import init_context_parallel, merge_attention
from minisgl.distributed.layout import ParallelLayout, configure_parallel
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.models.kimi_config import cache_config
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM
from minisgl.models.kimi_parallel import owns_expert, shard_spec
from test_kimi_parallel import _batch, _config, _initialize


def test_lse_merge_empty_partitions_and_extreme_scores():
    output = torch.tensor([[[[1.0, 3.0]]], [[[9.0, 7.0]]], [[[50.0, 60.0]]]])
    lse = torch.tensor([[[1000.0]], [[1000.0]], [[-torch.inf]]])
    torch.testing.assert_close(merge_attention(output, lse), torch.tensor([[[5.0, 5.0]]]))
    assert merge_attention(output, torch.full_like(lse, -torch.inf)).eq(0).all()


def test_independent_layout_and_checkpoint_slices():
    c = _config(shared=2)
    layout = ParallelLayout(3, 4, 2, 2)
    assert (layout.dcp_rank, layout.attention_rank, layout.attention_size) == (1, 1, 2)
    assert (layout.expert_rank, layout.expert_tensor_rank, layout.expert_tensor_size) == (1, 1, 2)
    assert list(layout.expert_range(4)) == [2, 3]
    assert owns_expert("model.layers.1.mlp.experts.3.w2.weight", c, layout)
    assert not owns_expert("model.layers.1.mlp.experts.1.w2.weight", c, layout)
    q = shard_spec("model.layers.1.self_attn.q_b_proj.weight", c, 3, 4, dcp_size=2)
    out = shard_spec("model.layers.1.self_attn.o_proj.weight", c, 3, 4, dcp_size=2)
    assert (q.start, q.stop, out.start, out.stop) == (8, 16, 9, 12)
    with pytest.raises(ValueError):
        ParallelLayout(0, 4, 3, 2)
    with pytest.raises(ValueError):
        ParallelLayout(0, 4, 2, 8)


def test_packed_checkpoint_preserves_global_expert_ids(tmp_path, monkeypatch):
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout_module
    import run_kimi_runtime_gate as runtime
    from minisgl.models import create_model
    from minisgl.models.config import ModelConfig
    from minisgl.models.kimi_weight import load_kimi_weight
    from minisgl.utils import cached_load_hf_config

    config = runtime.tiny_config()
    config.linear_attn_config["head_dim"] = 16
    config.v_head_dim = 16
    monkeypatch.setattr(runtime, "tiny_config", lambda: config)
    runtime.generate_fixtures(tmp_path / "fixture")
    path = str(tmp_path / "fixture/packed")
    model_config = ModelConfig.from_hf(cached_load_hf_config(path))
    full = dict(load_kimi_weight(path, "cpu", packed=True))
    monkeypatch.setattr(layout_module, "_DCP_SIZE", 2)
    monkeypatch.setattr(layout_module, "_EP_SIZE", 2)
    expert_ids = set()
    for rank in range(2):
        monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 2))
        with torch.device("meta"):
            model = create_model(model_config)
        model.load_weights(
            load_kimi_weight(
                path, "cpu", tp_size=2, tp_rank=rank, packed=True, dcp_size=2, ep_size=2
            ),
            device=torch.device("cpu"),
        )
        local = model.state_dict()
        for name, value in local.items():
            logical = name.replace("weight_packed", "weight").replace("weight_scale", "weight")
            spec = shard_spec(logical, config, rank, 2, dcp_size=2, ep_size=2)
            expected = full[name]
            if spec is not None:
                start, length = spec.start, spec.length
                if spec.axis == 1:
                    divisor = (
                        2
                        if name.endswith("weight_packed")
                        else 32
                        if name.endswith("weight_scale")
                        else 1
                    )
                    start, length = start // divisor, length // divisor
                expected = expected.narrow(spec.axis, start, length)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        ids = set(model.model.layers.op_list[1].mlp.experts._indices)
        assert not expert_ids.intersection(ids)
        expert_ids.update(ids)
    assert expert_ids == set(range(config.num_experts))


def _worker(rank, world, init_file, device_type):
    import minisgl.distributed.info as info

    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    try:
        c = _config(shared=2)
        c.moe_intermediate_size = 12
        info._TP_INFO = None
        reference = KimiK3LinearForCausalLM(SimpleNamespace(kimi=c))
        _initialize(reference)
        state = {name: value.clone() for name, value in reference.state_dict().items()}
        tokens = [2, 17, 4, 22, 8, 13, 7]
        # Noncontiguous physical pages, while offsets within pages preserve positions.
        table = torch.arange(64).reshape(16, 4).flip(0).reshape(2, 32)
        reference_cache = HybridCache(cache_config(c), 16, 4, 2, torch.float32, "cpu")
        batch = _batch(tokens)
        batch.out_loc = table[0, : len(tokens)]
        expected = reference.forward_batch(batch, reference_cache, table, all_logits=True)
        other_tokens = [13, 7, 22]
        other = _batch(other_tokens, slot=1, uid=30)
        other.out_loc = table[1, : len(other_tokens)]
        expected_other = reference.forward_batch(other, reference_cache, table, all_logits=True)
        expected_decode = []
        for token, length, slot, uid in ((3, len(tokens), 0, 10), (9, len(other_tokens), 1, 30)):
            step = _batch([token], length, slot=slot, uid=uid)
            step.out_loc = table[slot, length : length + 1]
            expected_decode.append(
                reference.forward_batch(step, reference_cache, table, all_logits=True)
            )
        expected_decode = torch.cat(expected_decode)
        info._TP_INFO = DistributedInfo(rank, world)
        for dcp, ep in ((2, 1), (1, 2), (2, 2), (4, 4)):
            configure_parallel(dcp_size=dcp, ep_size=ep)
            init_context_parallel()
            layout = ParallelLayout(rank, world, dcp, ep)
            model = KimiK3LinearForCausalLM(SimpleNamespace(kimi=c))

            def stream():
                for name, value in state.items():
                    if owns_expert(name, c, layout):
                        spec = shard_spec(name, c, rank, world, dcp_size=dcp, ep_size=ep)
                        yield (
                            name,
                            (value if spec is None else spec.slice(value)).to(device).clone(),
                        )

            model.load_weights(stream(), device=device)
            experts = model.model.layers.op_list[1].mlp.experts
            assert list(experts._indices) == list(layout.expert_range(c.num_experts))
            for _, expert in experts.items():
                assert (
                    expert.w1.weight.shape[0]
                    == c.moe_intermediate_size // layout.expert_tensor_size
                )
            for unified in (False, True):
                cache = HybridCache(
                    cache_config(c),
                    128,
                    4,
                    2,
                    torch.float32,
                    device,
                    max_cached_states=2,
                    unified_memory=unified,
                )
                if unified:
                    cache.arena.allocate_pages(list(range(16)))
                else:
                    assert cache._mla.shape[1] == 512 // dcp
                gpu_table = table.to(device).clone()

                def run(start, stop, *, slot=0, uid=10):
                    batch = _batch(tokens[start:stop], start, slot=slot, uid=uid)
                    batch.input_ids = batch.input_ids.to(device)
                    batch.positions = batch.positions.to(device)
                    batch.out_loc = gpu_table[slot, start:stop]
                    return model.forward_batch(batch, cache, gpu_table, all_logits=True).cpu()

                full = run(0, len(tokens))
                print(
                    f"layout={dcp, ep, unified} max_abs={(full - expected).abs().max().item()} relative_l2={((full - expected).norm() / expected.norm()).item()}",
                    flush=True,
                )
                # Four-rank reduction order changes small FP32 cancellation
                # residuals near zero; also bound the whole-vector error.
                torch.testing.assert_close(full, expected, rtol=4e-5, atol=2e-5)
                assert (full - expected).norm() / expected.norm() < 1e-5
                cache.release_request(0, 10)
                # First chunk has empty local key sets on all but one DCP rank.
                chunks = [run(0, 1), run(1, 4)]
                checkpoint = cache.recurrent.snapshot(0, 10)
                cache.release_request(0, 10)
                cache.recurrent.restore(1, 20, checkpoint)
                gpu_table[1, :4] = gpu_table[0, :4]
                chunks.append(run(4, len(tokens), slot=1, uid=20))
                chunked = torch.cat(chunks)
                torch.testing.assert_close(chunked, expected, rtol=4e-5, atol=2e-5)
                assert (chunked - expected).norm() / expected.norm() < 1e-5
                cache.release_request(1, 20)
                cache.recurrent.drop_checkpoint(checkpoint)
                # Heterogeneous requests share projections and collectives,
                # while retaining separate recurrent state and KV histories.
                gpu_table.copy_(table.to(device))
                left = _batch(tokens, slot=0, uid=40)
                right = _batch(other_tokens, slot=1, uid=41)
                combined = SimpleNamespace(
                    reqs=left.reqs + right.reqs,
                    input_ids=torch.cat((left.input_ids, right.input_ids)).to(device),
                    positions=torch.cat((left.positions, right.positions)).to(device),
                    out_loc=torch.cat(
                        (gpu_table[0, : len(tokens)], gpu_table[1, : len(other_tokens)])
                    ),
                )
                actual = model.forward_batch(combined, cache, gpu_table, all_logits=True).cpu()
                expected_batch = torch.cat((expected, expected_other))
                torch.testing.assert_close(actual, expected_batch, rtol=4e-5, atol=2e-5)
                assert (actual - expected_batch).norm() / expected_batch.norm() < 1e-5
                # Decode together after different prefill lengths. Batched KDA
                # must preserve each request's convolution and recurrent history.
                left = _batch([3], len(tokens), slot=0, uid=40)
                right = _batch([9], len(other_tokens), slot=1, uid=41)
                decode = SimpleNamespace(
                    reqs=left.reqs + right.reqs,
                    input_ids=torch.cat((left.input_ids, right.input_ids)).to(device),
                    positions=torch.cat((left.positions, right.positions)).to(device),
                    out_loc=torch.cat(
                        (
                            gpu_table[0, len(tokens) : len(tokens) + 1],
                            gpu_table[1, len(other_tokens) : len(other_tokens) + 1],
                        )
                    ),
                )
                decoded = model.forward_batch(decode, cache, gpu_table, all_logits=True).cpu()
                torch.testing.assert_close(decoded, expected_decode, rtol=4e-5, atol=2e-5)
                assert (decoded - expected_decode).norm() / expected_decode.norm() < 1e-5
                cache.release_request(0, 40)
                cache.release_request(1, 41)
                if unified:
                    cache.arena.check_integrity()
    finally:
        info._TP_INFO = None
        configure_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("KIMI_RUN_GLOO_TESTS") != "1", reason="opt-in multi-process gate"
)
def test_four_rank_collectives_full_model_chunks_and_private_prefix(tmp_path):
    device = os.environ.get("K3_PARALLEL_DEVICE", "cpu")
    mp.spawn(_worker, args=(4, str(tmp_path / "collectives"), device), nprocs=4, join=True)
