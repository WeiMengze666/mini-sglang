import json

import torch
from minisgl.checkpoint.safetensors import checkpoint_index, shard_readers
from safetensors.torch import save_file


def test_index_insertion_order_preserves_tensor_values_dtypes_and_slices(tmp_path):
    tensors = {
        "layer.a.weight": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
        "layer.z.weight": torch.arange(24, dtype=torch.float32).reshape(4, 6) / 8,
        "layer.m.weight_packed": torch.arange(128, dtype=torch.uint8).reshape(4, 32),
        "layer.m.weight_scale": torch.tensor([[127, 128]] * 4, dtype=torch.uint8),
    }
    save_file(
        {name: tensors[name] for name in ("layer.z.weight", "layer.m.weight_scale")},
        tmp_path / "a.safetensors",
    )
    save_file(
        {name: tensors[name] for name in ("layer.m.weight_packed", "layer.a.weight")},
        tmp_path / "b.safetensors",
    )
    entries = [
        ("layer.m.weight_packed", "b.safetensors"),
        ("layer.z.weight", "a.safetensors"),
        ("layer.a.weight", "b.safetensors"),
        ("layer.m.weight_scale", "a.safetensors"),
    ]
    for insertion_order in (entries, entries[::-1], entries[1:] + entries[:1]):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict(insertion_order)})
        )
        loaded = {}
        for name, reader in shard_readers(checkpoint_index(tmp_path, lambda name: name)):
            loaded[name] = reader.get_tensor(name).clone()
            assert loaded[name].dtype == tensors[name].dtype
            torch.testing.assert_close(loaded[name], tensors[name], rtol=0, atol=0)
            torch.testing.assert_close(
                reader.get_slice(name)[1:3, 1:3], tensors[name][1:3, 1:3], rtol=0, atol=0
            )
        assert set(loaded) == set(tensors)


def test_gqa_replication_and_fused_weights_use_rank_local_checkpoint_slices(tmp_path):
    from types import SimpleNamespace

    import torch
    from minisgl.checkpoint.huggingface import HuggingFaceCheckpoint
    from minisgl.checkpoint.loader import load_checkpoint
    from minisgl.distributed import DistributedInfo
    from minisgl.distributed.layout import ParallelLayout
    from safetensors.torch import save_file

    q = torch.arange(64.0).reshape(8, 8)
    k = torch.arange(32.0).reshape(4, 8) + 100
    v = k + 100
    gate, up = q + 400, q + 500
    prefix = "language_model.model.layers.0."
    save_file(
        {
            prefix + "self_attn.v_proj.weight": v,
            prefix + "self_attn.q_proj.weight": q,
            prefix + "mlp.gate_proj.weight": gate,
        },
        tmp_path / "first.safetensors",
    )
    vocab = torch.arange(56.0).reshape(7, 8)
    save_file(
        {
            prefix + "self_attn.k_proj.weight": k,
            prefix + "mlp.up_proj.weight": up,
            "language_model.model.embed_tokens.weight": vocab,
        },
        tmp_path / "second.safetensors",
    )
    config = SimpleNamespace(num_kv_heads=2, is_moe=False)
    for rank in range(4):
        layout = ParallelLayout(DistributedInfo(rank, 4))
        loaded = dict(
            load_checkpoint(
                tmp_path, "cpu", checkpoint_type=HuggingFaceCheckpoint, config=config, layout=layout
            )
        )
        local = slice(rank * 2, rank * 2 + 2)
        kv = slice(rank // 2 * 2, rank // 2 * 2 + 2)
        torch.testing.assert_close(
            loaded["model.layers.0.self_attn.qkv_proj.weight"],
            torch.cat([q[local], k[kv], v[kv]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            loaded["model.layers.0.mlp.gate_up_proj.weight"],
            torch.cat([gate[local], up[local]]),
            rtol=0,
            atol=0,
        )
        expected_vocab = torch.cat([vocab, torch.zeros(1, 8)])[local]
        torch.testing.assert_close(
            loaded["model.embed_tokens.weight"], expected_vocab, rtol=0, atol=0
        )
