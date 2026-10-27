"""TP algebra/integration gates plus opt-in real two-rank Gloo execution."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.layers.base import BaseOP, OPList
from minisgl.models.kimi_config import (
    KimiTextConfig,
    cache_bytes_per_token,
    cache_config,
    state_bytes_per_request,
)
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM
from minisgl.models.kimi_parallel import shard_spec, validate_tp_config


class _SimulatedGroup:
    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=15)
        self.inputs = [None, None]
        self.sequence = []

    def run(self, rank, op, tensor):
        self.inputs[rank] = (op, tensor)
        self.barrier.wait()
        assert self.inputs[0][0] == self.inputs[1][0]
        values = [entry[1] for entry in self.inputs]
        assert values[0].shape == values[1].shape
        assert values[0].dtype == values[1].dtype
        if rank == 0:
            self.sequence.append((op, tuple(tensor.shape), tensor.dtype))
        output = sum(values) if op == "sum" else torch.cat(values, dim=0)
        self.barrier.wait()
        return output


class _SimulatedComm:
    def __init__(self, rank, group):
        self.rank, self.group = rank, group

    def all_reduce(self, tensor):
        return self.group.run(self.rank, "sum", tensor)

    def all_gather(self, tensor):
        return self.group.run(self.rank, "gather", tensor)


def _attach_simulated_comm(op, comm):
    if hasattr(op, "_tp"):
        op._tp._comm = comm
    children = op.op_list if isinstance(op, OPList) else vars(op).values()
    for child in children:
        if isinstance(child, BaseOP):
            _attach_simulated_comm(child, comm)


def _config(*, vocab=24, full_gate=True, latent=4, shared=1, q_lora=4):
    return KimiTextConfig(
        hidden_size=8,
        vocab_size=vocab,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=12,
        max_position_embeddings=32,
        q_lora_rank=q_lora,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        linear_attn_config={
            "num_heads": 4,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
            "use_full_rank_gate": full_gate,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=2,
        num_experts=4,
        num_experts_per_token=2,
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=1.3,
        moe_intermediate_size=6,
        routed_expert_hidden_size=latent,
        latent_moe_use_norm=latent is not None,
        num_shared_experts=shared,
        first_k_dense_replace=1,
    )


def _batch(tokens, start=0, slot=0, uid=10):
    n = len(tokens)
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(
                uid=uid, table_idx=slot, cached_len=start, device_len=start + n, extend_len=n
            )
        ],
        input_ids=torch.tensor(tokens),
        positions=torch.arange(start, start + n),
        out_loc=torch.arange(slot * 32 + start, slot * 32 + start + n),
    )


def _cache(config):
    return HybridCache(cache_config(config), 64, 1, 2, torch.float32, torch.device("cpu"))


def _initialize(model):
    torch.manual_seed(47)
    for name, tensor in model.state_dict().items():
        tensor.normal_(0, 0.2)
        if "norm" in name:
            tensor.fill_(1)


def _manual_moe(x, state, c, prefix="model.layers.1.mlp."):
    """Direct full-tensor reference, with no Kimi forward or TP helper calls."""

    def weight(name):
        return state[prefix + name + ".weight"]

    scores = F.linear(x, weight("gate")).sigmoid()
    choices = scores + state[prefix + "gate.e_score_correction_bias"]
    groups = choices.reshape(len(x), c.num_expert_group, -1)
    selected_groups = groups.topk(2, dim=-1).values.sum(-1).topk(c.topk_group, dim=-1).indices
    selected = torch.zeros_like(groups[..., 0], dtype=torch.bool).scatter_(1, selected_groups, True)
    masked = choices.masked_fill(
        ~selected[..., None].expand_as(groups).reshape_as(choices), -torch.inf
    )
    ids = masked.topk(c.num_experts_per_token, dim=-1).indices
    scales = scores.gather(1, ids)
    scales = scales / scales.sum(-1, keepdim=True) * c.routed_scaling_factor
    routed = F.linear(x, weight("routed_expert_down_proj")) if c.routed_expert_hidden_size else x

    def activate(gate, up):
        return 4.0 * torch.tanh(gate / 4.0) * gate.sigmoid() * 25.0 * torch.tanh(up / 25.0)

    output = torch.zeros_like(routed)
    for row in range(len(x)):
        for slot in range(c.num_experts_per_token):
            p = "experts." + str(ids[row, slot].item())
            gate = F.linear(routed[row], weight(p + ".w1"))
            up = F.linear(routed[row], weight(p + ".w3"))
            output[row] += scales[row, slot] * F.linear(activate(gate, up), weight(p + ".w2"))
    if c.routed_expert_hidden_size:
        if c.latent_moe_use_norm:
            output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + c.rms_norm_eps)
            output = output * weight("routed_expert_norm")
        output = F.linear(output, weight("routed_expert_up_proj"))
    if c.num_shared_experts:
        gate = F.linear(x, weight("shared_experts.gate_proj"))
        up = F.linear(x, weight("shared_experts.up_proj"))
        output = output + F.linear(activate(gate, up), weight("shared_experts.down_proj"))
    return output


def _run_two_ranks(rank, init_file):
    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        for c in (
            _config(),
            _config(vocab=23, full_gate=False, latent=None, shared=2, q_lora=None),
        ):
            # Capture the same global, unsharded checkpoint on each worker first.
            info._TP_INFO = None
            reference = KimiK3LinearForCausalLM(SimpleNamespace(kimi=c))
            _initialize(reference)
            full_state = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
            table = torch.arange(64).reshape(2, 32)
            tokens = [2, 17, 4, 22, 8, 13]
            expected_cache = _cache(c)
            expected = reference.forward_batch(
                _batch(tokens), expected_cache, table, all_logits=True
            )
            x = torch.tensor(
                [
                    [1.0, 0.1, -0.2, 0.3, -0.4, 0.7, -0.2, 0.5],
                    [-0.8, -0.9, 0.4, -0.1, 0.5, 0.6, 0.3, -0.6],
                    [0.6, -0.3, -0.5, 0.2, -0.9, 0.8, -0.3, 0.1],
                ]
            )
            expected_moe = _manual_moe(x, full_state, c)
            torch.testing.assert_close(
                reference.model.layers.op_list[1].mlp.forward(x), expected_moe
            )

            info._TP_INFO = DistributedInfo(rank, 2)
            parallel = KimiK3LinearForCausalLM(SimpleNamespace(kimi=c))
            local_state = {}
            for name, tensor in full_state.items():
                spec = shard_spec(name, c, rank, 2)
                local_state[name] = (tensor if spec is None else spec.slice(tensor)).clone()
            parallel.load_state_dict(local_state)
            assert not local_state
            cache = _cache(c)
            assert cache.recurrent._recurrent.shape[2] == c.linear_attn_config["num_heads"] // 2
            assert cache._mla.shape == expected_cache._mla.shape
            state_bytes = state_bytes_per_request(c, torch.float32)
            assert state_bytes * 2 == state_bytes_per_request(c, torch.float32, tp_size=1)
            assert cache.allocated_bytes == (
                64 * cache_bytes_per_token(c, torch.float32) + 2 * state_bytes
            )
            actual = parallel.forward_batch(_batch(tokens), cache, table, all_logits=True)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            # Head-local recurrent state and replicated latent cache match TP1 too.
            torch.testing.assert_close(
                cache._mla[:, : len(tokens)], expected_cache._mla[:, : len(tokens)]
            )
            heads = c.linear_attn_config["num_heads"] // 2
            torch.testing.assert_close(
                cache.recurrent._recurrent[:, 0],
                expected_cache.recurrent._recurrent[:, 0, rank * heads : (rank + 1) * heads],
                rtol=2e-5,
                atol=2e-6,
            )
            cache.reset()
            actual_chunks = []
            for start, stop in ((0, 2), (2, 4), (4, 5), (5, 6)):
                actual_chunks.append(
                    parallel.forward_batch(
                        _batch(tokens[start:stop], start),
                        cache,
                        table,
                        all_logits=True,
                    )
                )
            torch.testing.assert_close(torch.cat(actual_chunks), expected, rtol=2e-5, atol=2e-6)
            last = parallel.forward_batch(_batch(tokens), _cache(c), table)
            torch.testing.assert_close(last, expected[-1:], rtol=2e-5, atol=2e-6)
            moe = parallel.model.layers.op_list[1].mlp
            torch.testing.assert_close(moe.forward(x), expected_moe, rtol=1e-5, atol=1e-6)
            assert all(
                expert.w1.weight.shape[0] == c.moe_intermediate_size // 2
                for expert in moe.experts.op_list
            )
            assert parallel.model.embed_tokens.weight.shape[0] == (
                c.vocab_size // 2 if c.vocab_size % 2 == 0 else c.vocab_size
            )
    finally:
        info._TP_INFO = None
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("KIMI_RUN_GLOO_TESTS") != "1",
    reason="Set KIMI_RUN_GLOO_TESTS=1 in an environment allowing local Gloo sockets",
)
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo unavailable"
)
def test_two_rank_gloo_model_cache_chunk_decode_and_manual_moe(tmp_path):
    mp.spawn(_run_two_ranks, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)


def test_shard_spec_and_invalid_splits():
    c = _config()
    validate_tp_config(c, 2)
    spec = shard_spec("model.layers.0.self_attn.q_proj.weight", c, 1, 2)
    assert (spec.axis, spec.start, spec.stop) == (0, 8, 16)
    spec = shard_spec("model.layers.1.self_attn.kv_b_proj.weight", c, 1, 2)
    assert (spec.axis, spec.start, spec.stop) == (0, 10, 20)
    spec = shard_spec("model.layers.1.mlp.experts.3.w2.weight", c, 1, 2)
    assert (spec.axis, spec.start, spec.stop) == (1, 3, 6)
    for name in (
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.1.mlp.routed_expert_up_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.0.self_attn.o_norm.weight",
        "model.layers.0.self_attn.f_a_proj.weight",
    ):
        assert shard_spec(name, c, 1, 2) is None
    assert shard_spec("lm_head.weight", _config(vocab=23), 1, 2) is None
    with pytest.raises(ValueError, match="TP rank"):
        shard_spec("lm_head.weight", c, 2, 2)
    with pytest.raises(ValueError, match="KDA heads"):
        validate_tp_config(c, 3)
    c = _config()
    c.moe_intermediate_size = 7
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        validate_tp_config(c, 2)


@pytest.mark.parametrize("odd_vocab", [False, True])
def test_simulated_collectives_model_and_independent_moe(monkeypatch, odd_vocab):
    """Supplemental algebra gate when sockets are denied; NOT distributed execution."""
    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo

    c = (
        _config(vocab=23, full_gate=False, latent=None, shared=2, q_lora=None)
        if odd_vocab
        else _config()
    )
    monkeypatch.setattr(info, "_TP_INFO", None)
    reference = KimiK3LinearForCausalLM(SimpleNamespace(kimi=c))
    _initialize(reference)
    state = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    table = torch.arange(64).reshape(2, 32)
    tokens = [2, 17, 4, 22, 8, 13]
    expected_cache = _cache(c)
    expected = reference.forward_batch(_batch(tokens), expected_cache, table, all_logits=True)
    x = torch.arange(24).reshape(3, 8).float() / 13 - 0.5
    expected_moe = _manual_moe(x, state, c)
    group, models, caches = _SimulatedGroup(), [], []
    for rank in range(2):
        monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 2))
        model = KimiK3LinearForCausalLM(SimpleNamespace(kimi=c))
        local = {}
        for name, tensor in state.items():
            spec = shard_spec(name, c, rank, 2)
            local[name] = (tensor if spec is None else spec.slice(tensor)).clone()
        model.load_state_dict(local)
        _attach_simulated_comm(model, _SimulatedComm(rank, group))
        models.append(model)
        caches.append(_cache(c))
        assert caches[-1].recurrent._recurrent.shape[2] == c.linear_attn_config["num_heads"] // 2
        state_bytes = state_bytes_per_request(c, torch.float32)
        assert 2 * state_bytes == state_bytes_per_request(c, torch.float32, tp_size=1)
        assert caches[-1].allocated_bytes == (
            64 * cache_bytes_per_token(c, torch.float32) + 2 * state_bytes
        )
    monkeypatch.setattr(info, "_TP_INFO", None)
    with ThreadPoolExecutor(max_workers=2) as pool:

        def run(function):
            futures = [pool.submit(function, rank) for rank in range(2)]
            return [future.result(timeout=30) for future in futures]

        outputs = run(
            lambda rank: models[rank].forward_batch(
                _batch(tokens), caches[rank], table, all_logits=True
            )
        )
        for output in outputs:
            torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        for rank, cache in enumerate(caches):
            torch.testing.assert_close(
                cache._mla[:, : len(tokens)], expected_cache._mla[:, : len(tokens)]
            )
            heads = c.linear_attn_config["num_heads"] // 2
            torch.testing.assert_close(
                cache.recurrent._recurrent[:, 0],
                expected_cache.recurrent._recurrent[:, 0, rank * heads : (rank + 1) * heads],
                rtol=2e-5,
                atol=2e-6,
            )
            cache.reset()
        for start, stop in ((0, 2), (2, 4), (4, 5), (5, 6)):
            outputs = run(
                lambda rank: models[rank].forward_batch(
                    _batch(tokens[start:stop], start), caches[rank], table, all_logits=True
                )
            )
            for output in outputs:
                torch.testing.assert_close(output, expected[start:stop], rtol=2e-5, atol=2e-6)
        before = len(group.sequence)
        outputs = run(lambda rank: models[rank].model.layers.op_list[1].mlp.forward(x))
        for output in outputs:
            torch.testing.assert_close(output, expected_moe, rtol=1e-5, atol=1e-6)
        # Routed and shared values share one reduction before latent norm/up.
        width = c.routed_expert_hidden_size or c.hidden_size
        assert group.sequence[before:] == [
            ("sum", (len(x), width + c.hidden_size), torch.float32),
        ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_simulated_collectives_packed_checkpoint_tp2_vs_dense_tp1(tmp_path, monkeypatch, dtype):
    """Actual packed streaming loader + TP2 math; simulated transport, not Gloo."""
    import minisgl.distributed.info as info
    import run_kimi_runtime_gate as runtime
    from minisgl.distributed import DistributedInfo
    from minisgl.models import create_model
    from minisgl.models.config import ModelConfig
    from minisgl.models.kimi_weight import load_kimi_weight
    from minisgl.utils import cached_load_hf_config, torch_dtype

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setattr(info, "_TP_INFO", None)
        config = runtime.tiny_config()
        # This fixture also packs attention output matrices. Their global input
        # width must be 64 so each TP2 input slice preserves a 32-value MXFP4 block.
        config.linear_attn_config["head_dim"] = 16
        config.v_head_dim = 16
        monkeypatch.setattr(runtime, "tiny_config", lambda: config)
        root = tmp_path / "packed-tp-fixture"
        manifest = runtime.generate_fixtures(root)
        device = torch.device("cpu")
        reference, dense_config = runtime.load_fixture(root / "dense", dtype, device)
        expected_cache = runtime.cache_for(dense_config, dtype, device)
        table = runtime.page_table(device)
        expected = reference.forward_batch(
            runtime.batch_for(runtime.PROMPT, device), expected_cache, table, all_logits=True
        )
        packed_config = ModelConfig.from_hf(cached_load_hf_config(str(root / "packed")))
        group, models, caches = _SimulatedGroup(), [], []
        for rank in range(2):
            monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 2))
            with torch.device("meta"), torch_dtype(dtype):
                model = create_model(packed_config)
            declared = model.state_dict()

            def stream():
                for name, value in load_kimi_weight(
                    str(root / "packed"),
                    device,
                    dtype=dtype,
                    tp_size=2,
                    tp_rank=rank,
                    packed=True,
                ):
                    if declared[name].is_floating_point():
                        value = value.to(declared[name].dtype)
                    else:
                        # A floating or widened packed tensor must fail, not be
                        # silently coerced back to U8 by the test harness.
                        assert value.dtype == declared[name].dtype == torch.uint8
                    yield name, value

            model.load_weights(stream())
            state = model.state_dict()
            assert all(value.device.type == "cpu" for value in state.values())
            for name in manifest["packed_names"]:
                assert name not in state
                prefix = name.removesuffix(".weight")
                assert state[prefix + ".weight_packed"].dtype == torch.uint8
                assert state[prefix + ".weight_scale"].dtype == torch.uint8
            # Both row-parallel bytes and scales are locally sliced before decode.
            expert = model.model.layers.op_list[1].mlp.experts.op_list[0]
            assert expert.w2.weight_packed.shape == (32, 16)
            assert expert.w2.weight_scale.shape == (32, 1)
            assert model.model.layers.op_list[0].self_attn.A_log.dtype == torch.float32
            assert model.model.embed_tokens.weight.dtype == dtype
            assert model.model.embed_tokens.weight.shape == (32, 32)
            _attach_simulated_comm(model, _SimulatedComm(rank, group))
            models.append(model)
            caches.append(runtime.cache_for(packed_config, dtype, device))
        monkeypatch.setattr(info, "_TP_INFO", None)
        rtol, atol = (0.04, 0.03) if dtype == torch.bfloat16 else (3e-5, 3e-6)
        with ThreadPoolExecutor(max_workers=2) as pool:

            def run(tokens, start=0):
                futures = [
                    pool.submit(
                        models[rank].forward_batch,
                        runtime.batch_for(tokens, device, start=start),
                        caches[rank],
                        table,
                        all_logits=True,
                    )
                    for rank in range(2)
                ]
                return [future.result(timeout=30) for future in futures]

            for output in run(runtime.PROMPT):
                torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
            heads = config.linear_attn_config["num_heads"] // 2
            for rank, cache in enumerate(caches):
                torch.testing.assert_close(
                    cache.recurrent._recurrent[:, 0],
                    expected_cache.recurrent._recurrent[:, 0, rank * heads : (rank + 1) * heads],
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    cache._mla[:, : len(runtime.PROMPT)],
                    expected_cache._mla[:, : len(runtime.PROMPT)],
                    rtol=rtol,
                    atol=atol,
                )
                cache.reset()
            for start, stop in ((0, 2), (2, 5), (5, 6), (6, 7)):
                for output in run(runtime.PROMPT[start:stop], start):
                    torch.testing.assert_close(output, expected[start:stop], rtol=rtol, atol=atol)
        assert any(op == "gather" for op, _, _ in group.sequence)
        assert any(op == "sum" for op, _, _ in group.sequence)
    finally:
        torch.set_num_threads(previous_threads)
