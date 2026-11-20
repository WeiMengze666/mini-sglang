"""Reconstruct complete expert checkpoint bytes after real EP/ETP loading."""

import json
import re

import minisgl.distributed.layout as layouts
import pytest
import torch
from minisgl.checkpoint.loader import load_checkpoint
from minisgl.distributed import DistributedInfo, info
from minisgl.distributed.layout import ParallelLayout
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_k3.checkpoint import KimiCheckpoint
from run_kimi_runtime_gate import tiny_config
from safetensors.torch import save_file


def _checkpoint(folder, monkeypatch):
    config = tiny_config()
    config.hidden_size = config.routed_expert_hidden_size = 256
    config.intermediate_size = config.moe_intermediate_size = 512
    config.num_experts = config.num_attention_heads = 8
    config.q_lora_rank = config.kv_lora_rank = 64
    config.qk_nope_head_dim = config.qk_rope_head_dim = config.v_head_dim = 32
    config.linear_attn_config = dict(config.linear_attn_config, num_heads=8, head_dim=32)
    monkeypatch.setattr(info, "_TP_INFO", None)
    monkeypatch.setattr(layouts, "_EP_SIZE", 1)
    monkeypatch.setattr(layouts, "_DCP_SIZE", 1)
    with torch.device("meta"):
        full = create_model(ModelConfig.from_hf(config))
    tensors, scales, expected, targets = {}, {}, {}, []
    pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.w([123])\.weight$")
    for name, value in full.state_dict().items():
        raw = "language_model." + name.replace(".mlp.", ".block_sparse_moe.")
        match = pattern.fullmatch(name)
        if match is None:
            tensors[raw] = torch.zeros(value.shape, dtype=value.dtype)
            continue
        layer, expert, matrix = map(int, match.groups())
        rows, columns = value.shape
        row = torch.arange(rows)[:, None]
        col = torch.arange(columns // 2)[None, :]
        low = (row + row // 16 + col * 3 + col // 16 + expert * 5 + matrix) % 16
        high = (row * 3 + row // 128 + col * 5 + col // 32 + expert * 7 + matrix * 3) % 16
        packed = (low | (high << 4)).to(torch.uint8)
        group = torch.arange(columns // 32)[None, :]
        scale = (125 + (row + row // 128 + group + group // 4 + expert + matrix) % 5).to(
            torch.uint8
        )
        prefix = raw.removesuffix(".weight")
        tensors[prefix + ".weight_packed"] = packed
        scales[prefix + ".weight_scale"] = scale
        expected[layer, expert, matrix, "weight_packed"] = packed.clone()
        expected[layer, expert, matrix, "weight_scale"] = scale.clone()
        targets.append(name.removesuffix(".weight"))
    config.quantization_config = {
        "format": "mxfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "config_groups": {
            "experts": {
                "targets": targets,
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                },
            }
        },
    }
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config.to_dict()))
    save_file(tensors, folder / "weights.safetensors")
    save_file(scales, folder / "scales.safetensors")
    entries = [(key, "weights.safetensors") for key in tensors]
    entries += [(key, "scales.safetensors") for key in scales]
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict(reversed(entries))})
    )
    return config, expected


@pytest.mark.parametrize("ep", [2, 8])
def test_packed_checkpoint_reconstructs_global_experts_after_ep_etp_loading(
    tmp_path, monkeypatch, ep
):
    path = tmp_path / "checkpoint"
    config, expected = _checkpoint(path, monkeypatch)
    monkeypatch.setattr(layouts, "_EP_SIZE", ep)
    monkeypatch.setattr(layouts, "_DCP_SIZE", 8)
    shards = {key: {} for key in expected}
    for rank in range(8):
        monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 8))
        with torch.device("meta"):
            model = create_model(ModelConfig.from_hf(config))
        model.load_weights(
            load_checkpoint(
                str(path),
                "cpu",
                checkpoint_type=KimiCheckpoint,
                layout=ParallelLayout(DistributedInfo(rank, 8), dcp_size=8, ep_size=ep),
                packed=True,
            ),
            device=torch.device("cpu"),
        )
        # Independent coordinates: never use the production shard/range helpers.
        owner, part = rank % ep, rank // ep
        owned = set(range(owner * (8 // ep), (owner + 1) * (8 // ep)))
        for layer in (1, 2):
            experts = model.model.layers.op_list[layer].mlp.experts
            assert set(experts._indices) == owned
            for expert, op in experts.items():
                for matrix in (1, 2, 3):
                    projection = getattr(op, "w" + str(matrix))
                    for field in ("weight_packed", "weight_scale"):
                        value = getattr(projection, field)
                        assert value.dtype == torch.uint8 and value.device.type == "cpu"
                        assert value.is_contiguous()
                        parts = shards[layer, expert, matrix, field]
                        assert part not in parts
                        parts[part] = value.clone()
    for key, original in expected.items():
        parts = shards[key]
        assert set(parts) == set(range(8 // ep))
        rebuilt = torch.cat([parts[t] for t in range(8 // ep)], dim=1 if key[2] == 2 else 0)
        assert rebuilt.shape == original.shape
        assert torch.equal(rebuilt, original), key
    assert {expert for _, expert, _, _ in shards} == set(range(8))
    assert not torch.cuda.is_initialized()
