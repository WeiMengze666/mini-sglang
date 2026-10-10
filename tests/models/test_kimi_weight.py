import json

import pytest
import torch
from minisgl.models.kimi_weight import decode_mxfp4, load_kimi_weight, normalize_kimi_name
from safetensors.torch import save_file


def test_mxfp4_all_codes_low_nibble_first_and_block_scales():
    codes = torch.arange(16, dtype=torch.uint8).repeat(4)
    packed = (codes[::2] | (codes[1::2] << 4)).reshape(1, 32)
    scales = torch.tensor([[127, 129]], dtype=torch.uint8)
    result = decode_mxfp4(packed, scales, dtype=torch.float32)
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    ).repeat(4)
    table[32:] *= 4
    torch.testing.assert_close(result, table[None])
    assert torch.signbit(result[0, 8])


def test_mxfp4_scale_extremes_and_exporter_shape():
    packed = torch.full((3, 16), 0x22, dtype=torch.uint8)
    scales = torch.tensor([0, 127, 255], dtype=torch.uint8).reshape(3, 1, 1, 1)
    result = decode_mxfp4(packed, scales, dtype=torch.float32)
    torch.testing.assert_close(result[0], torch.full((32,), 2.0**-127), atol=0, rtol=0)
    assert (result[1] == 1).all()
    assert torch.isnan(result[2]).all()


@pytest.mark.parametrize(
    "packed,scales",
    [
        (torch.zeros(2, 16, dtype=torch.int32), torch.zeros(2, 1, dtype=torch.uint8)),
        (torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(2, 1)),
        (torch.zeros(2, 32, dtype=torch.uint8), torch.zeros(2, 1, dtype=torch.uint8)),
    ],
)
def test_malformed_mxfp4_rejected(packed, scales):
    with pytest.raises(ValueError):
        decode_mxfp4(packed, scales)


def test_loader_cross_shard_scales_text_filtering_and_native_names(tmp_path):
    prefix = "language_model.model.layers.1.mlp.experts.0.w1"
    dense = "language_model.model.layers.0.mlp.gate_proj.weight"
    save_file(
        {
            prefix + ".weight_packed": torch.full((2, 16), 0x32, dtype=torch.uint8),
            dense: torch.randn(3, 4),
            "vision_tower.foo.weight": torch.ones(1),
            "language_model.model.layers.2.mlp.gate_proj.weight": torch.ones(1),
        },
        tmp_path / "first.safetensors",
    )
    save_file(
        {prefix + ".weight_scale": torch.tensor([[127], [126]], dtype=torch.uint8)},
        tmp_path / "second.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 2, "dtype": "bfloat16"}})
    )
    loaded = dict(load_kimi_weight(str(tmp_path), "cpu"))
    assert set(loaded) == {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.experts.0.w1.weight",
    }
    result = loaded["model.layers.1.mlp.experts.0.w1.weight"]
    torch.testing.assert_close(result[0], torch.tensor([1.0, 1.5] * 16, dtype=torch.bfloat16))
    torch.testing.assert_close(result[1], result[0] * 0.5)
    assert loaded["model.layers.0.mlp.gate_proj.weight"].dtype == torch.float32


def test_missing_scale_duplicate_and_tp_rejected(tmp_path):
    save_file(
        {"model.x.weight_packed": torch.ones(2, 16, dtype=torch.uint8)}, tmp_path / "a.safetensors"
    )
    with pytest.raises(ValueError, match="Missing MXFP4"):
        list(load_kimi_weight(str(tmp_path), "cpu"))
    with pytest.raises(NotImplementedError, match="TP=1"):
        list(load_kimi_weight(str(tmp_path), "cpu", tp_size=2))
    save_file(
        {"model.x.weight_packed": torch.ones(2, 16, dtype=torch.uint8)}, tmp_path / "b.safetensors"
    )
    with pytest.raises(ValueError, match="Duplicate checkpoint"):
        list(load_kimi_weight(str(tmp_path), "cpu"))


def test_metadata_index_and_mtp_skip(tmp_path):
    save_file(
        {
            "language_model.model.norm.weight": torch.ones(4),
            "language_model.mtp.layers.0.weight": torch.ones(4),
        },
        tmp_path / "a.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.norm.weight": "a.safetensors",
                    "language_model.mtp.layers.0.weight": "a.safetensors",
                }
            }
        )
    )
    assert set(dict(load_kimi_weight(str(tmp_path), "cpu"))) == {"model.norm.weight"}
    assert normalize_kimi_name("mm_projector.weight") is None
    assert normalize_kimi_name("language_model.model.layers.93.mlp.w1.weight", 93) is None


@pytest.mark.parametrize("shape", [(4,), (1, 1, 4, 1)])
def test_a_log_reference_narrowing(tmp_path, shape):
    save_file(
        {"language_model.model.layers.0.self_attn.A_log": torch.arange(4.0).reshape(shape)},
        tmp_path / "a.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"linear_attn_config": {"num_heads": 3}}})
    )
    result = dict(load_kimi_weight(str(tmp_path), "cpu"))["model.layers.0.self_attn.A_log"]
    torch.testing.assert_close(result, torch.arange(3.0))


def test_public_checkpoint_moe_and_legacy_prefix_mapping():
    # Actual moonshotai/Kimi-K3 index uses block_sparse_moe for routed layers.
    assert (
        normalize_kimi_name(
            "language_model.model.layers.12.block_sparse_moe.experts.895.w3.weight_packed"
        )
        == "model.layers.12.mlp.experts.895.w3.weight_packed"
    )
    assert (
        normalize_kimi_name("language_model.layers.12.block_sparse_moe.gate.weight")
        == "model.layers.12.mlp.gate.weight"
    )


def test_loader_public_checkpoint_moe_mapping(tmp_path):
    save_file(
        {
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed": torch.full(
                (2, 16), 0x22, dtype=torch.uint8
            ),
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale": torch.full(
                (2, 1), 127, dtype=torch.uint8
            ),
        },
        tmp_path / "a.safetensors",
    )
    result = dict(load_kimi_weight(str(tmp_path), "cpu"))
    assert set(result) == {"model.layers.1.mlp.experts.0.w1.weight"}
    torch.testing.assert_close(
        result["model.layers.1.mlp.experts.0.w1.weight"], torch.ones(2, 32, dtype=torch.bfloat16)
    )


def test_remote_fetches_metadata_before_weights_at_same_revision(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / "0123456789abcdef"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"linear_attn_config": {"num_heads": 2}}))
    save_file({"model.layers.0.self_attn.A_log": torch.arange(4.0)}, snapshot / "a.safetensors")
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    result = dict(load_kimi_weight("org/test-model", "cpu"))
    assert len(calls) == 2
    assert calls[0][1]["allow_patterns"] == ["config.json", "model.safetensors.index.json"]
    assert calls[1][1]["revision"] == snapshot.name
    assert calls[1][1]["allow_patterns"] == ["*.safetensors"]
    torch.testing.assert_close(result["model.layers.0.self_attn.A_log"], torch.arange(2.0))


def test_index_allows_hf_blob_symlink_but_rejects_traversal(tmp_path):
    blob = tmp_path / "blob"
    save_file({"model.norm.weight": torch.ones(4)}, blob)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "a.safetensors").symlink_to(blob)
    index = snapshot / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"model.norm.weight": "a.safetensors"}}))
    assert set(dict(load_kimi_weight(str(snapshot), "cpu"))) == {"model.norm.weight"}
    index.write_text(json.dumps({"weight_map": {"model.norm.weight": "../blob"}}))
    with pytest.raises(ValueError, match="outside model folder"):
        list(load_kimi_weight(str(snapshot), "cpu"))
