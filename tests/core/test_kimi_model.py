"""Tiny CPU numerical/lifecycle tests; these are not released-checkpoint parity."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.kvcache.kimi_pool import KimiHybridCache
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_config import KimiK3Config, KimiTextConfig
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM, attention_residual
from minisgl.models.kimi_moe import KimiLinear, KimiRMSNorm


def tiny_config():
    return KimiTextConfig(
        hidden_size=8,
        vocab_size=23,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=12,
        max_position_embeddings=32,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=6,
        routed_expert_hidden_size=4,
        latent_moe_use_norm=True,
        num_shared_experts=1,
        first_k_dense_replace=1,
    )


def setup_model(dtype=torch.float32):
    torch.manual_seed(19)
    config = ModelConfig.from_hf(KimiK3Config(text_config=tiny_config()))
    with torch.device("cpu"):
        old = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = KimiK3LinearForCausalLM(config)
        finally:
            torch.set_default_dtype(old)
    for name, value in model.state_dict().items():
        value.normal_(0, 0.13)
        if "norm" in name:
            value.fill_(1)
    return model, config


def cache_for(config, dtype=torch.float32):
    return KimiHybridCache(config.kimi, 128, 1, 3, dtype, torch.device("cpu"))


def batch_for(tokens, *, uid=10, slot=0, start=0, padding=0):
    n = len(tokens)
    req = SimpleNamespace(
        uid=uid, table_idx=slot, cached_len=start, device_len=start + n, extend_len=n
    )
    batch = SimpleNamespace(
        reqs=[req],
        input_ids=torch.tensor(tokens + [0] * padding),
        positions=torch.arange(start, start + n + padding),
        out_loc=torch.arange(slot * 32 + start, slot * 32 + start + n + padding),
    )
    return batch


def page_table():
    return torch.arange(96).reshape(3, 32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_chunk_decode_equivalence(dtype):
    model, config = setup_model(dtype)
    tokens = [2, 4, 8, 3, 9, 1]
    cache = cache_for(config, dtype)
    expected = model.forward_batch(batch_for(tokens), cache, page_table(), all_logits=True)
    cache.reset()
    actual = []
    for start, end in [(0, 2), (2, 4), (4, 5), (5, 6)]:
        actual.append(
            model.forward_batch(
                batch_for(tokens[start:end], start=start), cache, page_table(), all_logits=True
            )
        )
    torch.testing.assert_close(
        torch.cat(actual),
        expected,
        rtol=0.015 if dtype == torch.bfloat16 else 1e-5,
        atol=0.015 if dtype == torch.bfloat16 else 1e-6,
    )


def test_batch_isolation_padding_release_and_slot_reuse():
    model, config = setup_model()
    cache = cache_for(config)
    a, b = batch_for([2, 7, 4]), batch_for([9, 1], uid=11, slot=1)
    expected_a = model.forward_batch(a, cache, page_table(), all_logits=True)
    expected_b = model.forward_batch(b, cache, page_table(), all_logits=True)
    cache.reset()
    both = SimpleNamespace(
        reqs=a.reqs + b.reqs,
        input_ids=torch.cat((a.input_ids, b.input_ids, torch.tensor([99, 99]))),
        positions=torch.cat((a.positions, b.positions, torch.tensor([99, 99]))),
        out_loc=torch.cat((a.out_loc, b.out_loc, torch.tensor([32, 32]))),
    )
    actual = model.forward_batch(both, cache, page_table(), all_logits=True)
    torch.testing.assert_close(actual, torch.cat((expected_a, expected_b)), rtol=1e-5, atol=1e-6)
    cache.release_request(0, 10)
    new = batch_for([5, 6], uid=12)
    actual = model.forward_batch(new, cache, page_table(), all_logits=True)
    expected = model.forward_batch(new, cache_for(config), page_table(), all_logits=True)
    torch.testing.assert_close(actual, expected)
    # Finishing stale uid cannot erase its successor's recurrent state.
    cache.release_request(0, 10)
    model.forward_batch(batch_for([8], uid=12, start=2), cache, page_table())


def test_attnres_independent_scalar_reference():
    torch.manual_seed(5)
    norm, proj = KimiRMSNorm(4, 1e-5), KimiLinear(4, 1)
    norm.weight.normal_()
    proj.weight.normal_()
    head = torch.randn(2, 4)
    bank = [torch.randn(2, 4), torch.randn(2, 4)]
    actual = attention_residual(head, bank, norm, proj)
    expected = []
    for token in range(2):
        rows = [row[token].double() for row in [*bank, head]]
        scores = torch.stack(
            [
                (r * norm.weight.double() * proj.weight[0].double()).sum()
                / torch.sqrt((r * r).mean() + norm.eps)
                for r in rows
            ]
        )
        weights = scores.softmax(0)
        expected.append(sum(w * r for w, r in zip(weights, rows)))
    torch.testing.assert_close(actual, torch.stack(expected).float())


def test_model_roundtrip_and_config_validation():
    model, config = setup_model()
    other = KimiK3LinearForCausalLM(config)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    other.load_state_dict(state)
    assert not state
    assert config.is_moe and config.is_kimi
    assert config.kimi.linear_attn_config["head_dim"] != config.kimi.v_head_dim
    text = tiny_config()
    text.linear_attn_config["full_attn_layers"] = [1, 2]
    with pytest.raises(ValueError, match="partition"):
        ModelConfig.from_hf(text)


def test_last_logits_and_failed_forward_reset():
    model, config = setup_model()
    full = model.forward_batch(
        batch_for([1, 2, 3]), cache_for(config), page_table(), all_logits=True
    )
    last = model.forward_batch(batch_for([1, 2, 3]), cache_for(config), page_table())
    torch.testing.assert_close(last, full[-1:])
    cache = cache_for(config)
    with pytest.raises(IndexError):
        model.forward_batch(batch_for([99]), cache, page_table())
    # A failed forward must not leave an in-flight ownership lock.
    model.forward_batch(batch_for([1]), cache, page_table())


def test_multi_request_begin_failure_releases_started_requests():
    model, config = setup_model()
    cache = cache_for(config)
    a, b = batch_for([1]), batch_for([2], uid=11, slot=1, start=3)
    broken = SimpleNamespace(
        reqs=a.reqs + b.reqs,
        input_ids=torch.cat((a.input_ids, b.input_ids)),
        positions=torch.cat((a.positions, b.positions)),
        out_loc=torch.cat((a.out_loc, b.out_loc)),
    )
    with pytest.raises(RuntimeError, match="prefix"):
        model.forward_batch(broken, cache, page_table())
    model.forward_batch(a, cache, page_table())


def test_engine_guards_and_existing_model_config(tmp_path, monkeypatch):
    import json

    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.engine import _adjust_config
    from minisgl.scheduler.config import SchedulerConfig
    from minisgl.utils.hf import cached_load_hf_config
    from transformers import LlamaConfig, Qwen3MoeConfig

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    text = tiny_config()
    wrapper = KimiK3Config(text_config=text, dtype="bfloat16")
    (tmp_path / "config.json").write_text(json.dumps(wrapper.to_dict()))
    clone = cached_load_hf_config(str(tmp_path))
    assert isinstance(clone.text_config, KimiTextConfig)
    model_config = ModelConfig.from_hf(clone)
    engine = SchedulerConfig(str(tmp_path), DistributedInfo(0, 1), torch.bfloat16)
    _adjust_config(engine)
    assert engine.attention_backend == "kimi"
    assert engine.cuda_graph_bs == [] and engine.cache_type == "naive"
    engine_tp = SchedulerConfig(str(tmp_path), DistributedInfo(0, 2), torch.bfloat16)
    with pytest.raises(ValueError, match="TP=1"):
        _adjust_config(engine_tp)
    assert model_config.is_kimi
    llama = ModelConfig.from_hf(LlamaConfig())
    assert not llama.is_moe and not llama.is_kimi
    qwen = ModelConfig.from_hf(Qwen3MoeConfig())
    assert qwen.is_moe and not qwen.is_kimi


def test_complete_synthetic_checkpoint_load(tmp_path):
    import json

    from minisgl.models.kimi_weight import load_kimi_weight
    from safetensors.torch import save_file

    model, config = setup_model()
    raw = {
        "language_model." + name.replace(".mlp.", ".block_sparse_moe."): value.clone()
        for name, value in model.state_dict().items()
    }
    save_file(raw, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config.kimi.to_dict()))
    loaded = dict(load_kimi_weight(str(tmp_path), "cpu"))
    fresh = KimiK3LinearForCausalLM(config)
    fresh.load_state_dict(loaded)
    assert not loaded
    expected = model.forward_batch(batch_for([1, 2, 3]), cache_for(config), page_table())
    actual = fresh.forward_batch(batch_for([1, 2, 3]), cache_for(config), page_table())
    torch.testing.assert_close(actual, expected)
