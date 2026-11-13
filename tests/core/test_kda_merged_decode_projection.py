"""CPU projection/load contracts; no numerical GPU qualification."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import sgl
from minisgl.attention.kda.backend import KDAAttentionBackend, KDAMetadata
from minisgl.attention.kda.registry import KDADecodeResult, KDAKernelSpec, get_kda_kernel
from minisgl.distributed import DistributedInfo
from minisgl.layers.kda import KDALayer
from minisgl.models.kimi_parallel import shard_spec


def build_layer(monkeypatch, *, rank=0, tp=8, full=True, dtype=torch.bfloat16, actual=False):
    import minisgl.distributed.info as info

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, tp))
    config = KDAConfig(7168, 96, 128, 4, full_rank_gate=full) if actual else KDAConfig(
        8, 16, 4, 4, full_rank_gate=full
    )
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            layer = KDALayer(config)
    finally:
        torch.set_default_dtype(old)
    return layer


def payload_for(layer, value=1):
    return {
        name: torch.full(tensor.shape, value / 16, dtype=tensor.dtype)
        for name, tensor in layer.state_dict().items()
    }


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_real_stream_load_reload_keeps_tp_slices_inside_weight_arena(monkeypatch, rank):
    import minisgl.layers.weight as weight_module

    layer = build_layer(monkeypatch, rank=rank)
    original_names = tuple(layer.state_dict())
    config = SimpleNamespace(linear_attn_config={"kda_layers": [1], "num_heads": 16, "head_dim": 4})
    arenas = []
    allocate = weight_module.allocate_weight_views
    def capture(expected, device, *args, **kwargs):
        views = allocate(expected, device, *args, **kwargs)
        arenas.append({t.untyped_storage()._cdata for t in views.values()})
        return views
    monkeypatch.setattr(weight_module, "allocate_weight_views", capture)
    for generation in (1, 2):
        payload = payload_for(layer, generation)
        for part, offset in zip(("q", "k", "v", "g"), (0, 1, 2, 3)):
            full = torch.arange(64 * 8).reshape(64, 8).remainder(4).bfloat16() + offset + generation
            name = f"{part}_proj.weight"
            spec = shard_spec("model.layers.0.self_attn." + name, config, rank, 8)
            assert (spec.axis, spec.start, spec.stop) == (0, rank * 8, (rank + 1) * 8)
            payload[name] = spec.slice(full).contiguous()
        layer.load_weights(payload.items(), device=torch.device("cpu"))
        plan = layer._qkvg
        assert plan is not None and plan.weight.shape == (32, 8)
        assert tuple(layer.state_dict()) == original_names
        allocated = arenas[-1]
        assert {t.untyped_storage()._cdata for t in layer.state_dict().values()} == allocated
        assert plan.weight.untyped_storage()._cdata in allocated
        offset = 0
        for name, op in layer._qkvg_projections().items():
            assert op.weight.data_ptr() == plan.weight.data_ptr() + offset * 2
            torch.testing.assert_close(op.weight, payload[("g" if name == "gate" else name) + "_proj.weight"])
            offset += op.weight.numel()
        before = plan.weight.data_ptr()
        layer.after_load()
        assert layer._qkvg.weight.data_ptr() == before
        assert sum(t.numel() for t in layer.state_dict().values()) == sum(t.numel() for t in payload.values())
    assert not torch.cuda.is_initialized()


def test_actual_tp8_meta_shapes_require_no_extra_model_parameter(monkeypatch):
    layer = build_layer(monkeypatch, actual=True)
    ops = layer._qkvg_projections()
    assert all(op.weight.shape == (1536, 7168) for op in ops.values())
    assert len({op.weight.untyped_storage()._cdata for op in ops.values()}) == 1
    assert layer._qkvg is None
    assert not any("_qkvg" in name for name in layer.state_dict())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("full,dtype", [(False, torch.bfloat16), (True, torch.float32)])
def test_low_rank_and_other_dtypes_keep_original_projection_contract(monkeypatch, full, dtype):
    layer = build_layer(monkeypatch, full=full, dtype=dtype)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    assert layer._qkvg is None
    assert layer.project_qkvg(torch.ones(2, 8, dtype=dtype)) is None
    assert len(layer.project(torch.ones(2, 8, dtype=dtype))) == 5
    if not full:
        config = SimpleNamespace(linear_attn_config={"kda_layers": [1], "num_heads": 16, "head_dim": 4})
        assert shard_spec("model.layers.0.self_attn.g_a_proj.weight", config, 3, 8) is None
        assert shard_spec("model.layers.0.self_attn.g_b_proj.weight", config, 3, 8).start == 24
    assert not torch.cuda.is_initialized()


def test_decode_one_merged_linear_carries_gate_without_stash(monkeypatch):
    import minisgl.layers.linear as linear_module

    layer = build_layer(monkeypatch)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    calls = []
    def merged(x, weight, bias=None):
        calls.append((x, weight))
        return F.linear(x, weight, bias)
    monkeypatch.setattr(linear_module, "linear_bf16", merged)
    for op in layer._qkvg_projections().values():
        monkeypatch.setattr(op, "forward", lambda value: pytest.fail("independent Q/K/V/g decode GEMM"))
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    projected, gate = sgl._project_decode(layer, x)
    saved = gate.clone()
    assert len(calls) == 1 and calls[0][1] is layer._qkvg.weight
    assert len(projected) == 5 and all(t.dtype == torch.bfloat16 for t in projected)
    assert gate.shape == projected[0].shape == (2, 8)
    second, second_gate = sgl._project_decode(layer, x * 2)
    assert len(calls) == 2 and second_gate is not gate
    torch.testing.assert_close(gate, saved)
    torch.testing.assert_close(second_gate, gate * 2)
    assert not hasattr(layer, "_projected_gate")
    assert second[0].shape == projected[0].shape
    assert not torch.cuda.is_initialized()


def test_prefill_public_project_does_not_use_decode_merge(monkeypatch):
    layer = build_layer(monkeypatch)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    monkeypatch.setattr(layer, "project_qkvg", lambda value: pytest.fail("prefill used decode merge"))
    assert len(sgl.project(layer, torch.ones(2, 8, dtype=torch.bfloat16))) == 5
    assert not torch.cuda.is_initialized()


def test_explicit_gate_reaches_finish_without_reprojection(monkeypatch):
    import minisgl.layers.kda as kda_module

    layer = build_layer(monkeypatch)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    monkeypatch.setattr(layer.g_proj, "forward", lambda value: pytest.fail("gate recomputed"))
    monkeypatch.setattr(kda_module, "gated_rms_norm", lambda out, gate, weight, eps: out + gate)
    monkeypatch.setattr(layer.o_proj, "forward", lambda value: value)
    layer._tp = SimpleNamespace(all_reduce=lambda value: value)
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    gate = torch.ones_like(x)
    output = layer.finish(x, x.reshape(2, 2, 4), gate=gate)
    torch.testing.assert_close(output, gate)
    assert not torch.cuda.is_initialized()


def test_indexed_result_gate_is_forwarded_as_the_same_object():
    gate = torch.ones(2, 8)
    metadata = KDAMetadata(torch.arange(3), state_indices=torch.tensor([0, 1]))
    cache = SimpleNamespace(recurrent=object())
    spec = KDAKernelSpec(
        "gate-test", get_kda_kernel("torch").recurrence,
        indexed_decode=lambda *args: KDADecodeResult(torch.zeros(2, 8), gate=gate),
    )
    backend = KDAAttentionBackend(SimpleNamespace(), spec, cache=cache)
    seen = []
    def finish(x, output, normed, *, gate):
        seen.append(gate)
        return output
    layer = SimpleNamespace(finish=finish)
    batch = SimpleNamespace(
        is_decode=True, reqs=[SimpleNamespace(extend_len=1) for _ in range(2)], attn_metadata=metadata,
    )
    backend.forward_layer(layer, 0, torch.zeros(2, 8), batch, cache, None)
    assert seen == [gate]
    assert not torch.cuda.is_initialized()
