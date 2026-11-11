"""Native MLA local storage and post-cache decode replicas, without CUDA."""

import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.attention.config import MLAConfig
from minisgl.distributed.context import ContextParallel
from minisgl.layers.base import BaseOP
from minisgl.layers.mla import MLALayer
from minisgl.models.kimi_parallel import shard_spec


def build(monkeypatch, cp_size, *, rank=None, dtype=torch.bfloat16, qrank=4):
    import minisgl.layers.mla as module

    tp_size, heads, nope, rope, value, latent, hidden = 8, 96, 3, 2, 3, 5, 7
    rank = cp_size - 1 if rank is None else rank
    local_heads = heads // tp_size
    first_head = rank // cp_size * cp_size * local_heads
    last_head = first_head + cp_size * local_heads
    qinput = hidden if qrank is None else qrank
    full_q = (
        torch.arange(heads * (nope + rope) * qinput).reshape(heads, nope + rope, qinput) % 17 / 64
    ).to(dtype)
    full_kv = (
        torch.arange(heads * (nope + value) * latent).reshape(heads, nope + value, latent) % 13 / 64
    ).to(dtype)
    calls = []
    cp = SimpleNamespace(rank=rank % cp_size, size=cp_size, activation=None)

    def gather(x, *, dim=0):
        calls.append((dim, tuple(x.shape), x.dtype, x.stride()))
        if dim == 1:
            assert cp.activation is not None
            return cp.activation.clone()
        if x.ndim == 2:
            return full_q[first_head:last_head].flatten(0, 1).clone()
        return full_kv[first_head:last_head, :nope].contiguous().clone()

    cp.gather_heads = gather
    monkeypatch.setattr(
        module, "TensorParallel", lambda: SimpleNamespace(size=tp_size, all_reduce=lambda x: x)
    )
    monkeypatch.setattr(module, "ContextParallel", lambda: cp)
    monkeypatch.setattr(
        module,
        "get_parallel_layout",
        lambda: SimpleNamespace(mla_decode=SimpleNamespace(size=tp_size // cp_size)),
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        layer = MLALayer(MLAConfig(hidden, heads, nope, rope, value, latent, qrank, 64))
    finally:
        torch.set_default_dtype(previous)
    for name, weight in layer.state_dict().items():
        weight.fill_(1 if "norm" in name else 0.03125)
    query = layer.q_proj if qrank is None else layer.q_b_proj
    query.weight = full_q[rank * local_heads : (rank + 1) * local_heads].flatten(0, 1).clone()
    layer.kv_b_proj.weight = (
        full_kv[rank * local_heads : (rank + 1) * local_heads].flatten(0, 1).clone()
    )
    return SimpleNamespace(
        layer=layer,
        cp=cp,
        calls=calls,
        query=query,
        local_heads=local_heads,
        full_q=full_q[first_head:last_head].flatten(0, 1),
        full_kv=full_kv[first_head:last_head],
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("qrank", [None, 4])
def test_local_weights_then_late_q_and_key_only_replicas(monkeypatch, cp_size, qrank):
    f = build(monkeypatch, cp_size, qrank=qrank)
    layer = f.layer
    assert f.query.weight.shape[0] == 12 * 5
    assert layer.kv_b_proj.weight.shape == (12 * 6, 5)
    keys_before = set(layer.state_dict())
    layer.after_load()
    assert not f.calls and layer._decode_q_weight is None and layer._decode_key_weight is None
    local_kv = layer.kv_b_proj.weight.view(12, 6, 5)
    torch.testing.assert_close(layer._key_weight, local_kv[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(layer._value_weight, local_kv[:, 3:].transpose(1, 2), rtol=0, atol=0)
    assert layer._key_weight.stride() == (15, 1, 3)
    assert layer._value_weight.stride() == (15, 1, 5)
    layer.prepare_inference()
    if cp_size > 1:
        assert [item[0] for item in f.calls] == [0, 0]
        torch.testing.assert_close(layer._decode_q_weight, f.full_q, rtol=0, atol=0)
        torch.testing.assert_close(layer._decode_key_weight, f.full_kv[:, :3], rtol=0, atol=0)
        assert layer._decode_key_weight.is_contiguous()
    else:
        assert not f.calls and layer._decode_q_weight is None
        assert layer._decode_key_weight is layer._key_weight
    assert layer._value_weight.shape == (12, 5, 3)
    assert set(layer.state_dict()) == keys_before
    calls = len(f.calls)
    layer.prepare_inference()
    assert len(f.calls) == calls

    x, positions = torch.ones(3, 7, dtype=torch.bfloat16) / 8, torch.arange(3)
    source = x if qrank is None else layer.q_a_layernorm.forward(layer.q_a_proj.forward(x))
    local_q, _, _ = layer.project(x, positions)
    expected = F.linear(source, f.query.weight).view(3, 12, 5)
    torch.testing.assert_close(local_q, expected, rtol=0, atol=0)
    decode_q, _, _ = layer.project(x, positions, is_decode=True)
    expected = F.linear(source, f.full_q).view(3, 12 * cp_size, 5)
    torch.testing.assert_close(decode_q, expected, rtol=0, atol=0)
    assert len(f.calls) == calls  # No per-token query collective on the native dense path.


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_float32_projection_gathers_query_activations_without_weight_reinterpret(
    monkeypatch, cp_size
):
    f = build(monkeypatch, cp_size, dtype=torch.float32, qrank=None)
    layer = f.layer
    layer.after_load()
    layer.prepare_inference()
    assert layer._decode_q_weight is None
    assert len(f.calls) == 1  # Only the key plan, not the unsupported dense-Q replica.
    x = torch.ones(2, 7)
    f.cp.activation = F.linear(x, f.full_q).view(2, 12 * cp_size, 5)
    q, _, _ = layer.project(x, torch.arange(2), is_decode=True)
    torch.testing.assert_close(q, f.cp.activation, rtol=0, atol=0)
    assert f.calls[-1][0] == 1


def test_quantized_query_executor_is_not_read_as_a_dense_weight(monkeypatch):
    f = build(monkeypatch, 2, qrank=None)
    layer = f.layer

    class QuantizedQuery(BaseOP):
        weight_format = "mxfp4"
        bias = None

        @property
        def weight(self):
            raise AssertionError("Packed weights must never be reinterpreted")

        def forward(self, x):
            return torch.ones(len(x), 12 * 5, dtype=x.dtype)

    layer.q_proj = QuantizedQuery()
    layer.after_load()
    layer.prepare_inference()
    assert layer._decode_q_weight is None
    f.cp.activation = torch.ones(2, 24, 5, dtype=torch.bfloat16)
    q, _, _ = layer.project(torch.ones(2, 7, dtype=torch.bfloat16), torch.arange(2), is_decode=True)
    torch.testing.assert_close(q, f.cp.activation, rtol=0, atol=0)
    assert f.calls[-1][0] == 1


def test_reload_invalidates_all_derived_storage(monkeypatch):
    f = build(monkeypatch, 4)
    layer = f.layer
    layer.after_load()
    layer.prepare_inference()
    old_q = weakref.ref(layer._decode_q_weight)
    old_key = weakref.ref(layer._decode_key_weight)
    old_value = weakref.ref(layer._value_weight)
    f.query.weight = f.query.weight.clone() + 0.125
    layer.kv_b_proj.weight = layer.kv_b_proj.weight.clone() + 0.125
    layer.after_load()
    assert old_q() is None and old_key() is None and old_value() is None
    assert layer._decode_q_weight is None and not layer._inference_prepared
    assert layer._key_weight.stride() == (15, 1, 3)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dim", [0, 1])
def test_public_head_gather_preserves_rank_major_head_order(cp_size, dim):
    cp = object.__new__(ContextParallel)
    cp.group = SimpleNamespace(size=cp_size)
    local = torch.arange(24).reshape(2, 3, 4)
    gathered = torch.stack([local + 100 * rank for rank in range(cp_size)])
    cp._comm = SimpleNamespace(all_gather=lambda x: gathered)
    got = cp.gather_heads(local, dim=dim)
    expected = torch.cat(tuple(gathered.unbind(0)), dim=dim)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("rank", range(8))
def test_checkpoint_slices_remain_tp_local_for_all_dcp_sizes(cp_size, rank):
    config = SimpleNamespace(
        num_attention_heads=96,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        linear_attn_config={"kda_layers": []},
    )
    for name, width in [
        ("q_b_proj.weight", 192),
        ("q_proj.weight", 192),
        ("kv_b_proj.weight", 256),
    ]:
        spec = shard_spec("model.layers.0.self_attn." + name, config, rank, 8, dcp_size=cp_size)
        assert spec.length == 12 * width and spec.start == rank * 12 * width
