import json

import torch
from minisgl.models.checkpoint import checkpoint_index, shard_readers
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
