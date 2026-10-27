"""CPU-only capacity-path coverage. All tensor values are synthetic fixtures."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.latent_moe import LatentMoE
from minisgl.layers.linear import Linear
from minisgl.models.kimi_config import KimiTextConfig
from minisgl.models.kimi_quant import (
    quantization_rules,
    replace_quantized_linears,
)
from minisgl.models.kimi_weight import decode_mxfp4, load_kimi_weight
from minisgl.quantization.mxfp4 import MXFP4Linear
from safetensors.torch import save_file


def quant_config():
    return {
        "format": "mxfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "format": "mxfp4-pack-quantized",
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "dynamic": False,
                    "scale_dtype": "torch.uint8",
                },
            }
        },
        "ignore": [
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
            "re:.*lm_head.*",
        ],
    }


def config(**overrides):
    values = {
        "hidden_size": 64,
        "vocab_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "moe_intermediate_size": 64,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_expert_group": 2,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
        "hidden_act": "situ",
        "num_shared_experts": 1,
        "routed_expert_hidden_size": 32,
        "latent_moe_use_norm": True,
        "rms_norm_eps": 1e-5,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 32,
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 32,
            "kda_layers": [1],
            "full_attn_layers": [2],
        },
        "quantization_config": quant_config(),
    }
    values.update(overrides)
    return KimiTextConfig(**values)


def fixture_weights(rows, cols):
    gen = torch.Generator().manual_seed(rows * 1000 + cols)
    return (
        torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8),
        torch.randint(122, 127, (rows, cols // 32), generator=gen, dtype=torch.uint8),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5, 64), (2, 3, 64), (64,), (0, 64)])
def test_packed_linear_matches_dense_without_dense_state(dtype, shape):
    op = MXFP4Linear(64, 19, has_bias=True, dtype=dtype, chunk_rows=5)
    op.weight_packed, op.weight_scale = fixture_weights(19, 64)
    op.bias.copy_(torch.linspace(-0.1, 0.1, 19).to(dtype))
    x = torch.randn(shape, dtype=dtype)
    expected = F.linear(x, decode_mxfp4(op.weight_packed, op.weight_scale, dtype=dtype), op.bias)
    torch.testing.assert_close(op.forward(x), expected)
    assert set(op.state_dict()) == {"weight_packed", "weight_scale", "bias"}
    assert not hasattr(op, "weight")
    assert op.weight_packed.numel() + op.weight_scale.numel() == 19 * 64 * 17 // 32


def test_decode_workspace_is_bounded_and_not_retained(monkeypatch):
    import minisgl.quantization.mxfp4 as loader

    op = MXFP4Linear(64, 65, chunk_rows=17, max_decode_elements=640)
    op.weight_packed, op.weight_scale = fixture_weights(65, 64)
    calls = []
    original = loader.decode_mxfp4

    def decode(weight, scales, **kwargs):
        calls.append(weight.shape[0])
        assert weight.shape[0] <= 10
        return original(weight, scales, **kwargs)

    monkeypatch.setattr(loader, "decode_mxfp4", decode)
    result = op.forward(torch.randn(4, 64, requires_grad=True))
    assert calls == [10] * 6 + [5]
    assert not result.requires_grad
    assert all(t.dtype == torch.uint8 for t in op.state_dict().values())


def test_meta_replacement_honors_targets_ignore_and_native_dense():
    model = BaseOP()
    with torch.device("meta"):
        model.layers = OPList([BaseOP()])
        layer = model.layers.op_list[0]
        layer.mlp = LatentMoE(config())
        layer.self_attention_res_proj = Linear(64, 1)
        layer.self_attn = BaseOP()
        layer.self_attn.q_proj = Linear(64, 64)
        model.lm_head = Linear(64, 64)
    assert replace_quantized_linears(model, config()) == 12
    assert isinstance(layer.mlp.experts.op_list[0].w1, MXFP4Linear)
    assert isinstance(layer.mlp.shared_experts.up_proj, Linear)
    assert isinstance(layer.mlp.routed_expert_down_proj, Linear)
    assert isinstance(layer.self_attention_res_proj, Linear)
    assert isinstance(layer.self_attn.q_proj, Linear)
    assert isinstance(model.lm_head, Linear)
    assert layer.mlp.gate.weight.dtype == torch.float32
    assert all(t.device.type == "meta" for t in model.state_dict().values())
    assert replace_quantized_linears(model, config()) == 0


def test_specific_selectors_and_oplist_direct_children():
    model = BaseOP()
    model.layers = OPList([Linear(32, 32), Linear(32, 32)])
    quant = quant_config()
    quant["config_groups"]["group_0"]["targets"] = ["language_model.layers.0"]
    assert replace_quantized_linears(model, SimpleNamespace(quantization_config=quant)) == 1
    assert isinstance(model.layers.op_list[0], MXFP4Linear)
    assert isinstance(model.layers.op_list[1], Linear)
    assert replace_quantized_linears(model, SimpleNamespace()) == 0


def test_tiny_latent_moe_packed_matches_dense():
    c = config()
    model = BaseOP()
    model.mlp = LatentMoE(c)
    torch.manual_seed(8)
    for name, value in model.state_dict().items():
        value.copy_(torch.randn_like(value) * 0.08)
        if name.endswith("norm.weight"):
            value.fill_(1)
    dense = copy.deepcopy(model)
    assert replace_quantized_linears(model, c, chunk_rows=7) == 12
    for packed_expert, dense_expert in zip(model.mlp.experts.op_list, dense.mlp.experts.op_list):
        for name in ("w1", "w2", "w3"):
            packed_op, dense_op = getattr(packed_expert, name), getattr(dense_expert, name)
            rows, cols = dense_op.weight.shape
            packed_op.weight_packed, packed_op.weight_scale = fixture_weights(rows, cols)
            dense_op.weight.copy_(
                decode_mxfp4(
                    packed_op.weight_packed, packed_op.weight_scale, dtype=dense_op.weight.dtype
                )
            )
    x = torch.randn(5, c.hidden_size) * 0.1
    torch.testing.assert_close(model.mlp.forward(x), dense.mlp.forward(x), atol=1e-6, rtol=2e-5)
    assert model.mlp.forward(x[:0]).shape == (0, c.hidden_size)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda q: q.update(format="int8"), NotImplementedError),
        (lambda q: q.update(config_groups={}), ValueError),
        (lambda q: q.update(ignore=["re:["]), ValueError),
        (lambda q: q["config_groups"]["group_0"].update(targets="Linear"), ValueError),
        (lambda q: q["config_groups"]["group_0"].update(input_activations={}), NotImplementedError),
        (lambda q: q["config_groups"]["group_0"]["weights"].update(group_size=16), ValueError),
        (lambda q: q["config_groups"]["group_0"]["weights"].update(dynamic=True), ValueError),
        (lambda q: q["config_groups"]["group_0"]["weights"].update(symmetric=False), ValueError),
        (
            lambda q: q["config_groups"]["group_0"]["weights"].update(scale_dtype="float32"),
            ValueError,
        ),
    ],
)
def test_malformed_quantization_rejected(mutation, error):
    quant = quant_config()
    mutation(quant)
    with pytest.raises(error):
        quantization_rules({"quantization_config": quant})


def test_overlapping_groups_fail_closed():
    model = BaseOP()
    model.linear = Linear(32, 32)
    quant = quant_config()
    quant["config_groups"]["duplicate"] = copy.deepcopy(quant["config_groups"]["group_0"])
    with pytest.raises(ValueError, match="Multiple"):
        replace_quantized_linears(model, SimpleNamespace(quantization_config=quant))


@pytest.mark.parametrize("rows,cols,chunk", [(1, 31, 4), (1, 0, 4), (0, 32, 4), (1, 32, 0)])
def test_invalid_linear_dimensions(rows, cols, chunk):
    with pytest.raises(ValueError):
        MXFP4Linear(cols, rows, chunk_rows=chunk)


def write_checkpoint(folder, tensors, c=None):
    save_file(tensors, folder / "a.safetensors")
    if c is not None:
        (folder / "config.json").write_text(json.dumps({"text_config": c.to_dict()}))


def test_loader_packed_cross_shard_no_decode(monkeypatch, tmp_path):
    import minisgl.models.kimi_weight as loader

    prefix = "language_model.model.layers.0.block_sparse_moe.experts.0.w1"
    packed, scale = fixture_weights(64, 32)
    save_file({prefix + ".weight_packed": packed}, tmp_path / "a.safetensors")
    save_file({prefix + ".weight_scale": scale[:, None, :, None]}, tmp_path / "b.safetensors")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("Packed loading must not decode")

    monkeypatch.setattr(loader, "decode_mxfp4", forbidden_decode)
    result = dict(load_kimi_weight(str(tmp_path), "cpu", packed=True))
    prefix = "model.layers.0.mlp.experts.0.w1"
    assert set(result) == {prefix + ".weight_packed", prefix + ".weight_scale"}
    torch.testing.assert_close(result[prefix + ".weight_packed"], packed)
    torch.testing.assert_close(result[prefix + ".weight_scale"], scale)
    assert all(value.dtype == torch.uint8 for value in result.values())


@pytest.mark.parametrize("projection,shape,axis", [("w1", (64, 32), 0), ("w2", (32, 64), 1)])
@pytest.mark.parametrize("keep_packed", [False, True])
def test_packed_tp_slices_rows_and_block_columns_before_read(
    tmp_path, monkeypatch, projection, shape, axis, keep_packed
):
    import minisgl.models.kimi_weight as loader

    prefix = f"model.layers.0.mlp.experts.0.{projection}"
    packed, scales = fixture_weights(*shape)
    write_checkpoint(
        tmp_path, {prefix + ".weight_packed": packed, prefix + ".weight_scale": scales}, config()
    )
    original = loader._read_packed_pair
    seen_shapes = []

    def pair(*args, **kwargs):
        result = original(*args, **kwargs)
        seen_shapes.append(result[0].shape)
        return result

    monkeypatch.setattr(loader, "_read_packed_pair", pair)
    results = []
    dense = decode_mxfp4(packed, scales, dtype=torch.float32)
    for rank in range(2):
        result = dict(
            load_kimi_weight(
                str(tmp_path),
                "cpu",
                packed=keep_packed,
                tp_size=2,
                tp_rank=rank,
                dtype=torch.float32,
            )
        )
        actual = (
            decode_mxfp4(
                result[prefix + ".weight_packed"],
                result[prefix + ".weight_scale"],
                dtype=torch.float32,
            )
            if keep_packed
            else result[prefix + ".weight"]
        )
        torch.testing.assert_close(actual, dense.chunk(2, dim=axis)[rank])
        results.append(actual)
    assert all(shape[axis] == packed.shape[axis] // 2 for shape in seen_shapes)
    torch.testing.assert_close(torch.cat(results, dim=axis), dense)


def test_tp_dense_and_legacy_a_log_sliced(tmp_path):
    c = config()
    c.linear_attn_config["num_heads"] = 4
    tensors = {
        "model.layers.0.self_attn.A_log": torch.arange(8.0).reshape(1, 1, 8, 1),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(128 * 64.0).reshape(128, 64),
        "model.layers.0.mlp.gate.weight": torch.randn(4, 64),
    }
    write_checkpoint(tmp_path, tensors, c)
    result = dict(load_kimi_weight(str(tmp_path), "cpu", packed=True, tp_size=2, tp_rank=1))
    torch.testing.assert_close(result["model.layers.0.self_attn.A_log"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(
        result["model.layers.0.self_attn.q_proj.weight"],
        tensors["model.layers.0.self_attn.q_proj.weight"][64:],
    )
    torch.testing.assert_close(
        result["model.layers.0.mlp.gate.weight"], tensors["model.layers.0.mlp.gate.weight"]
    )


def test_unaligned_packed_column_shard_rejected(tmp_path):
    prefix = "model.layers.0.mlp.experts.0.w2"
    packed, scales = fixture_weights(32, 32)
    write_checkpoint(
        tmp_path,
        {prefix + ".weight_packed": packed, prefix + ".weight_scale": scales},
        config(moe_intermediate_size=32),
    )
    with pytest.raises(ValueError, match="align to 32"):
        list(load_kimi_weight(str(tmp_path), "cpu", packed=True, tp_size=2))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda t: t.pop("model.x.weight_scale"), "Missing MXFP4"),
        (lambda t: t.pop("model.x.weight_packed"), "Orphan"),
        (
            lambda t: t.update({"model.x.weight_scale": torch.ones(32, 2, dtype=torch.uint8)}),
            "scale shape",
        ),
        (lambda t: t.update({"model.x.weight_scale": torch.ones(32, 1)}), "uint8"),
        (lambda t: t.update({"model.x.weight": torch.ones(32, 32)}), "Duplicate normalized"),
    ],
)
def test_malformed_packed_pairs_rejected(tmp_path, mutation, match):
    packed, scales = fixture_weights(32, 32)
    tensors = {"model.x.weight_packed": packed, "model.x.weight_scale": scales}
    mutation(tensors)
    write_checkpoint(tmp_path, tensors)
    with pytest.raises(ValueError, match=match):
        list(load_kimi_weight(str(tmp_path), "cpu", packed=True))


def test_remote_index_downloads_only_selected_text_shards_at_pinned_revision(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    c = config()
    c.num_hidden_layers = 1
    write_checkpoint(snapshot, {"model.norm.weight": torch.ones(64)}, c)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.norm.weight": "a.safetensors",
                    "vision_tower.weight": "vision.safetensors",
                    "language_model.mtp.weight": "mtp.safetensors",
                    "model.layers.1.mlp.gate_proj.weight": "later.safetensors",
                }
            }
        )
    )
    calls = []

    def download(repo, **kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    result = dict(load_kimi_weight("synthetic/model", "cpu", packed=True))
    assert set(result) == {"model.norm.weight"}
    assert calls == [
        {"allow_patterns": ["config.json", "model.safetensors.index.json"]},
        {"revision": snapshot.name, "allow_patterns": ["a.safetensors"]},
    ]


def test_selected_mla_kv_b_packing_fails_explicitly():
    model = BaseOP()
    model.self_attn = BaseOP()
    model.self_attn.kv_b_proj = Linear(32, 64)
    quant = quant_config()
    quant["ignore"] = []
    with pytest.raises(NotImplementedError, match="MLA kv_b_proj"):
        replace_quantized_linears(model, SimpleNamespace(quantization_config=quant))


def test_packed_metadata_slice_descriptors():
    from audit_kimi_packed_storage import sliced_descriptor

    c = SimpleNamespace(kimi=config())
    prefix = "model.layers.0.mlp.experts.0.w2"
    assert sliced_descriptor(prefix + ".weight_packed", (32, 32), "U8", c, 1, 2) == (
        (32, 16),
        "U8",
        True,
    )
    assert sliced_descriptor(prefix + ".weight_scale", (32, 1, 2, 1), "U8", c, 1, 2) == (
        (32, 1),
        "U8",
        True,
    )
    with pytest.raises(ValueError, match="exceeds"):
        sliced_descriptor(prefix + ".weight_packed", (32, 16), "U8", c, 1, 2)
    with pytest.raises(ValueError, match="Invalid MXFP4"):
        sliced_descriptor(prefix + ".weight_packed", (32, 17), "U8", c, 1, 2)


def test_optional_official_packed_storage_headers():
    import os
    from pathlib import Path

    from audit_kimi_packed_storage import audit_rank, verified_headers
    from minisgl.models.config import ModelConfig
    from minisgl.models.kimi_config import KimiK3Config

    directory = os.environ.get("KIMI_PACKED_AUDIT_HEADERS")
    config_path = os.environ.get("KIMI_PACKED_AUDIT_CONFIG")
    index_path = os.environ.get("KIMI_PACKED_AUDIT_INDEX")
    if not all((directory, config_path, index_path)):
        pytest.skip("Set KIMI_PACKED_AUDIT_HEADERS/CONFIG/INDEX to run saved-header meta audit")
    tensors, _ = verified_headers(Path(config_path), Path(index_path), Path(directory))
    c = ModelConfig.from_hf(KimiK3Config(**json.loads(Path(config_path).read_text())))
    for size in (1, 2, 4, 8):
        result = audit_rank(c, tensors, size - 1, size)
        assert result["errors"] == []
        assert result["packed_matrices"] > 0
        assert result["resident_model_bytes"] < result["equivalent_dequantized_model_bytes"]


def test_storage_auditor_reports_casts_and_rejects_packed_mismatch():
    from audit_kimi_packed_storage import audit_rank
    from minisgl.models.config import ModelConfig
    from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM

    c = config(q_lora_rank=32, kv_lora_rank=32, max_position_embeddings=64)
    c.linear_attn_config["short_conv_kernel_size"] = 4
    model_config = ModelConfig.from_hf(c)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            model = KimiK3LinearForCausalLM(model_config)
        dtype_names = {torch.uint8: "U8", torch.float32: "F32", torch.bfloat16: "BF16"}
        headers = {
            name: (tuple(t.shape), dtype_names[t.dtype]) for name, t in model.state_dict().items()
        }
    finally:
        torch.set_default_dtype(previous)
    norm_name = "model.layers.0.self_attn.o_norm.weight"
    headers[norm_name] = (headers[norm_name][0], "F32")
    result = audit_rank(model_config, headers, 0, 1)
    assert result["errors"] == []
    assert result["floating_dtype_conversions"] == [
        {"name": norm_name, "checkpoint": "F32", "runtime": "BF16"}
    ]
    assert result["loaded_checkpoint_bytes_before_floating_casts"] > result["resident_model_bytes"]
    assert result["resident_model_bytes"] < result["equivalent_dequantized_model_bytes"]
    packed_name = "model.layers.0.mlp.experts.0.w1.weight_packed"
    headers[packed_name] = (headers[packed_name][0], "BF16")
    result = audit_rank(model_config, headers, 0, 1)
    assert any("Invalid MXFP4" in error for error in result["errors"])
