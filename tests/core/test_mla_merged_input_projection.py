"""Official MLA QKV-A merge scope, input precision and stream-load ownership (CPU)."""

import weakref

import pytest
import torch
import torch.nn.functional as F
from minisgl.attention.config import MLAConfig
from minisgl.distributed import DistributedInfo
from minisgl.layers.base import BaseOP
from minisgl.layers.mla import MLALayer


def build(monkeypatch, *, qrank=4, actual=False, rank=0):
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 8))
    monkeypatch.setattr(layout, "_DCP_SIZE", 8)
    monkeypatch.setattr(layout, "_EP_SIZE", 1)
    config = (
        MLAConfig(7168, 96, 128, 64, 128, 512, 1536, 1048576, output_gate=True)
        if actual
        else MLAConfig(8, 16, 2, 2, 2, 3, qrank, 64, output_gate=True)
    )
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            layer = MLALayer(config)
    finally:
        torch.set_default_dtype(old)
    return layer


def payload_for(layer, generation=1):
    result = {}
    for name, value in layer.state_dict().items():
        values = torch.arange(value.numel()).reshape(value.shape).remainder(7) - 3
        result[name] = (values / 16 + generation / 32).to(value.dtype)
    return result


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_canonical_qkv_a_scope_has_original_names_and_tp_replicated_meta_aliases(monkeypatch, rank):
    layer = build(monkeypatch, actual=True, rank=rank)
    assert layer._qkv_a is None
    assert layer.q_a_proj.weight.shape == (1536, 7168)
    assert layer.kv_a_proj_with_mqa.weight.shape == (576, 7168)
    q, kv = layer.q_a_proj.weight, layer.kv_a_proj_with_mqa.weight
    assert q.untyped_storage()._cdata == kv.untyped_storage()._cdata
    assert kv.storage_offset() - q.storage_offset() == q.numel()
    # q_b, kv_b and output gate are deliberately outside the official A group.
    assert layer.q_b_proj.weight.shape == (2304, 1536)
    assert layer.kv_b_proj.weight.shape == (3072, 512)
    assert layer.g_proj.weight.shape == (1536, 7168)
    assert all(
        op.weight.untyped_storage()._cdata != q.untyped_storage()._cdata
        for op in (layer.q_b_proj, layer.kv_b_proj, layer.g_proj)
    )
    assert layer._decode_heads == 96 and layer._heads == 12
    names = set(layer.state_dict())
    assert "q_a_proj.weight" in names and "kv_a_proj_with_mqa.weight" in names
    assert not any("qkv_a" in name for name in names)
    assert layer._skip_rope
    assert not torch.cuda.is_initialized()


def test_stream_load_reload_preserves_alias_arena_and_releases_derived_plan(monkeypatch):
    import minisgl.layers.weight as weight_module

    layer = build(monkeypatch)
    names = tuple(layer.state_dict())
    allocated = []
    allocate = weight_module.allocate_weight_views

    def record(expected, device, *args, **kwargs):
        views = allocate(expected, device, *args, **kwargs)
        allocated.append(
            {
                name: (value.data_ptr(), value.untyped_storage()._cdata)
                for name, value in views.items()
            }
        )
        return views

    monkeypatch.setattr(weight_module, "allocate_weight_views", record)
    old = None
    for generation in (1, 2):
        payload = payload_for(layer, generation)
        # Checkpoint streaming order need not match merge order.
        layer.load_weights(reversed(tuple(payload.items())), device=torch.device("cpu"))
        assert old is None or old() is None
        assert tuple(layer.state_dict()) == names
        plan = layer._qkv_a
        assert plan is not None and plan.widths == (4, 5) and plan.names == ("q", "kv")
        assert plan.weight.shape == (9, 8)
        q, kv = layer.q_a_proj.weight, layer.kv_a_proj_with_mqa.weight
        assert q.data_ptr() == plan.weight.data_ptr()
        assert kv.data_ptr() == q.data_ptr() + q.numel() * q.element_size()
        for name, value in layer.state_dict().items():
            assert (value.data_ptr(), value.untyped_storage()._cdata) == allocated[-1][name]
            torch.testing.assert_close(value, payload[name], rtol=0, atol=0)
        assert plan.weight.untyped_storage()._cdata == allocated[-1]["q_a_proj.weight"][1]
        pointer = plan.weight.data_ptr()
        layer.after_load()
        assert layer._qkv_a.weight.data_ptr() == pointer
        assert sum(t.numel() for t in layer.state_dict().values()) == sum(
            t.numel() for t in payload.values()
        )
        del plan
        old = weakref.ref(layer._qkv_a.weight)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("tokens", [0, 1, 5])
def test_one_input_dtype_qkv_a_gemm_matches_independent_projection_and_norm(monkeypatch, tokens):
    import minisgl.layers.linear as linear_module

    layer = build(monkeypatch)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    x = (torch.arange(tokens * 16).reshape(tokens, 16).remainder(11) / 16).bfloat16()[:, ::2]
    positions = torch.arange(tokens)
    q_a_weight = layer.q_a_proj.weight.clone()
    kv_a_weight = layer.kv_a_proj_with_mqa.weight.clone()
    expected_source = layer.q_a_layernorm.forward(F.linear(x, q_a_weight))
    expected_q = F.linear(expected_source, layer.q_b_proj.weight).reshape(tokens, 2, 4)
    expected_kv = F.linear(x, kv_a_weight)
    expected_latent = layer.kv_a_layernorm.forward(expected_kv[:, :3])
    calls = []
    original = linear_module.linear_bf16

    def record(value, weight, bias=None):
        calls.append((value.dtype, weight, value.stride()))
        return original(value, weight, bias)

    monkeypatch.setattr(linear_module, "linear_bf16", record)
    for op in (layer.q_a_proj, layer.kv_a_proj_with_mqa):
        monkeypatch.setattr(
            op, "forward", lambda value: pytest.fail("independent QKV-A projection")
        )
    monkeypatch.setattr(
        layer.g_proj, "forward", lambda value: pytest.fail("gate included in QKV-A")
    )
    actual = layer.project(x, positions)
    assert len(calls) == 1 and calls[0][1] is layer._qkv_a.weight
    assert calls[0][0] == torch.bfloat16
    for got, expected in zip(actual, (expected_q, expected_latent, expected_kv[:, 3:])):
        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("role", ["q_a_proj", "kv_a_proj_with_mqa"])
@pytest.mark.parametrize("weight_format", ["mxfp4", "dense"])
def test_packed_and_low_rank_style_executors_keep_independent_calls(
    monkeypatch, role, weight_format
):
    layer = build(monkeypatch)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    original = getattr(layer, role)
    calls = []

    class Adapter(BaseOP):
        # A stand-in for an executor with semantics beyond reading dense weight.
        @property
        def weight(self):
            raise AssertionError("Adapter weight must not be inspected or reinterpreted")

        def forward(self, x):
            calls.append(role)
            return original.forward(x) + 0.125

    adapter = Adapter()
    adapter.weight_format = weight_format
    adapter.bias = None
    setattr(layer, role, adapter)
    layer.after_load()
    assert layer._qkv_a is None
    actual = layer.project(torch.ones(2, 8, dtype=torch.bfloat16), torch.arange(2))
    assert calls == [role] and all(value.dtype == torch.bfloat16 for value in actual)
    assert not torch.cuda.is_initialized()


def test_no_q_lora_rank_keeps_direct_query_and_independent_kv(monkeypatch):
    layer = build(monkeypatch, qrank=None)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    assert layer._qkv_a is None and not hasattr(layer, "q_a_proj")
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    expected = layer.q_proj.forward(x).reshape(2, 2, 4)
    q, latent, key = layer.project(x, torch.arange(2))
    torch.testing.assert_close(q, expected, rtol=0, atol=0)
    assert latent.shape == (2, 3) and key.shape == (2, 2)
    assert not torch.cuda.is_initialized()
