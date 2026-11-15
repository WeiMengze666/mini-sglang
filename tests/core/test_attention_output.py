"""CPU ownership and semantic routing for prepared attention projections."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.core import Context
from minisgl.layers.attn_residual import ResidualBank
from minisgl.layers.base import BaseOP
from minisgl.layers.linear import Linear, LinearRowParallel
from minisgl.layers.weight import replace_quantized_linears


class TensorParallel:
    size = 2
    rank = 0

    def __init__(self):
        self.outputs, self.events = {}, []

    def reserve_output(self, key, shape, dtype, *, enabled=True):
        self.events.append(("reserve", key, shape, dtype, enabled))
        if not enabled:
            return None
        if key not in self.outputs:
            self.outputs[key] = torch.empty(shape, dtype=dtype)
        return self.outputs[key]

    def all_reduce(self, x):
        self.events.append(("generic",))
        return x * 2

    def all_reduce_residual(self, x, residual=None, *, output_key=None):
        assert output_key in self.outputs
        self.events.append(("prepared", output_key))
        x.mul_(2)
        if residual is not None:
            x.add_(residual)
        return x


@pytest.fixture
def preparation(monkeypatch):
    import minisgl.core as core

    def forbid(*args, **kwargs):
        pytest.fail("CPU attention output test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    ctx = Context(page_size=1, max_forward_len=16384)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return ctx


def output_op(tp, *, role="attention.output", bias=False, width=16):
    op = LinearRowParallel(
        width * tp.size, width, has_bias=bias, dtype=torch.bfloat16, output_key=role, tp=tp
    )
    op.weight.zero_()
    op.weight[:, 0].fill_(1)
    if op.bias is not None:
        op.bias.fill_(3)
    return op


def test_post_cache_preparation_shares_full_capacity_and_returns_borrowed_output(preparation):
    tp = TensorParallel()
    first, second = output_op(tp), output_op(tp)
    first.prepare_inference()
    second.prepare_inference()
    assert first._output is second._output
    assert first._output.shape == (16384, 16)  # Never shrink to graph48.
    x = torch.ones(48, 16, dtype=torch.bfloat16)
    residual = torch.full_like(x, 3)
    before = len(tp.events)
    out = first.forward(x, residual=residual)
    assert out.data_ptr() == first._output.data_ptr() and out.eq(5).all()
    second.forward(x * 2)
    assert out.eq(4).all()  # Explicitly borrowed until shared-slot reuse.
    assert torch.equal(residual, torch.full_like(x, 3))
    assert all(event[0] == "prepared" for event in tp.events[before:])


@pytest.mark.parametrize("unavailable", ["abi", "unloaded", "quantized"])
def test_unavailable_producer_still_participates_in_preparation(
    preparation, monkeypatch, unavailable
):
    import minisgl.layers.linear as linear

    tp = TensorParallel()
    op = output_op(tp, width=32)
    if unavailable == "abi":
        monkeypatch.setattr(linear, "_LINEAR_OUT", None)
    elif unavailable == "unloaded":
        op.weight = torch.empty_like(op.weight, device="meta")
    else:
        op = op.new_with_weight_format("mxfp4")
    op.prepare_inference()
    assert len(tp.events) == 1 and tp.events[0][0] == "reserve"
    assert tp.events[0][2:] == ((16384, 32), torch.bfloat16, False)
    assert op._output is None and not tp.outputs


@pytest.mark.parametrize("prepared", [False, True])
def test_unprepared_or_oversize_forward_keeps_original_fallback(preparation, prepared):
    tp = TensorParallel()
    op = output_op(tp)
    preparation.max_forward_len = 2
    if prepared:
        op.prepare_inference()
        op._output.fill_(-17)
    x = torch.ones(3, 16, dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    result = op.forward(x, residual=residual)
    assert result.eq(3).all() and tp.events[-1] == ("generic",)
    if prepared:
        assert op._output.eq(-17).all()
    assert not any(event[0] == "prepared" for event in tp.events)


def test_row_parallel_replicated_bias_is_added_once_in_two_rank_sum():
    inputs = [torch.ones(2, 16, dtype=torch.bfloat16), torch.full((2, 16), 2, dtype=torch.bfloat16)]
    partials = []
    for rank, x in enumerate(inputs):
        tp = SimpleNamespace(size=2, rank=rank, all_reduce=lambda value: value)
        op = output_op(tp, role=None, bias=True)
        op.weight.zero_().fill_diagonal_(rank + 1)
        partials.append(op.forward(x))
    expected = (
        sum(
            F.linear(x, torch.eye(16, dtype=torch.bfloat16) * (rank + 1))
            for rank, x in enumerate(inputs)
        )
        + 3
    )
    torch.testing.assert_close(sum(partials), expected, rtol=0, atol=0)


def quantization():
    return {
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "all": {
                "targets": ["Linear"],
                "weights": {"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32},
            }
        },
        "ignore": [],
    }


def test_quantized_factory_preserves_row_parallel_semantics_and_checkpoint_names(
    preparation, monkeypatch
):
    class Model(BaseOP):
        def forward(self):
            raise NotImplementedError

    tp = TensorParallel()
    model = Model()
    model.o_proj = output_op(tp, bias=True, width=32)
    original_bias = model.o_proj.bias
    model.o_proj.prepare_inference()
    shared = model.o_proj._output
    assert replace_quantized_linears(model, quantization()) == 1
    packed = model.o_proj
    assert type(packed) is LinearRowParallel and packed._tp is tp
    assert packed._output_role == "attention.output" and packed._output is None
    assert packed.bias is original_bias
    assert set(model.state_dict()) == {"o_proj.weight_packed", "o_proj.weight_scale", "o_proj.bias"}
    assert not packed.supports_out(torch.bfloat16)
    packed.prepare_inference()
    assert packed._output is None and next(iter(tp.outputs.values())) is shared
    monkeypatch.setattr(packed, "_forward_mxfp4", lambda x: torch.ones(len(x), 32, dtype=x.dtype))
    before = len(tp.events)
    assert packed.forward(torch.ones(2, 32, dtype=torch.bfloat16)).eq(2).all()
    assert tp.events[before:] == [("generic",)]


def test_unknown_custom_linear_factory_is_explicitly_rejected():
    class Custom(Linear):
        def forward(self, x):
            return super().forward(x) + 1

    class Model(BaseOP):
        def forward(self):
            raise NotImplementedError

    model = Model()
    model.proj = Custom(32, 32)
    with pytest.raises(NotImplementedError, match="Custom"):
        replace_quantized_linears(model, quantization())


def test_decoder_bank_snapshots_survive_shared_slot_reuse_and_mlp_returns_owned_sum(preparation):
    from minisgl.models.kimi_k3 import KimiDecoderLayer

    tp = TensorParallel()
    projection = output_op(tp)
    projection.prepare_inference()

    class Norm:
        weight = torch.ones(16, dtype=torch.bfloat16)
        eps = 1e-6

        def forward(self, value):
            return value.clone()

    residuals = []

    class Backend:
        def forward_layer(self, layer, layer_id, x, batch, cache, page_table, *, residual=None):
            residuals.append(residual)
            return layer.forward(torch.ones_like(x), residual=residual)

    head = torch.ones(3, 16, dtype=torch.bfloat16)
    bank = ResidualBank(head, 3)
    old_heads = []
    for layer_id in range(4):
        layer = object.__new__(KimiDecoderLayer)
        layer._layer_id, layer._block_size = layer_id, 2
        layer.self_attn = projection
        layer.input_layernorm = layer.post_attention_layernorm = Norm()
        layer.self_attention_res_norm = layer.mlp_res_norm = Norm()
        layer.self_attention_res_proj = layer.mlp_res_proj = SimpleNamespace(
            weight=torch.zeros(1, 16, dtype=torch.bfloat16)
        )
        layer.mlp = SimpleNamespace(
            forward=lambda value, *, residual: residual + torch.ones_like(value)
        )
        old_heads.append((head, head.clone()))
        head = layer.forward(head, bank, None, None, None, Backend())
        assert head.data_ptr() != projection._output.data_ptr()
        assert head.eq(3 if layer_id % 2 == 0 else 6).all()
    assert residuals[0] is residuals[2] is None
    assert residuals[1] is old_heads[1][0] and residuals[3] is old_heads[3][0]
    assert list(bank)[0].eq(1).all() and list(bank)[1].eq(6).all()
    for value, snapshot in old_heads:
        assert torch.equal(value, snapshot)


@pytest.mark.parametrize(
    "path",
    [
        "indexed_prefill",
        "indexed_decode",
        "functional_decode",
        "single_fallback",
        "packed_fallback",
    ],
)
def test_kda_all_return_paths_forward_the_explicit_residual(path):
    from minisgl.attention.kda.backend import KDAAttentionBackend
    from minisgl.attention.kda.registry import KDAKernelSpec, KDAResult, get_kda_kernel
    from minisgl.layers.kda import KDALayer

    x = torch.zeros(4 if path == "packed_fallback" else 2, 3)
    residual = torch.full_like(x, 7)
    finishes = []

    class Layer:
        decode = KDALayer.decode

        def project(self, value):
            return (value,)

        def update(self, value, state, *, kernel, is_decode):
            return value + 1, "final"

        def finish(self, value, output, *, normed=False, gate=None, residual=None):
            finishes.append(residual)
            return output + residual

    recurrent = SimpleNamespace(
        batch_indices=lambda rows: torch.tensor(rows),
        get_kda_batch=lambda *args: "initial",
        set_kda_batch=lambda *args: None,
    )
    cache = SimpleNamespace(
        device=x.device,
        recurrent=recurrent,
        get_kda_state=lambda *args: "initial",
        set_kda_state=lambda *args: None,
    )
    spec = KDAKernelSpec(
        "residual-test",
        get_kda_kernel("torch").recurrence,
        decode=(lambda layer, value, state: KDAResult(value + 1, "final", normed=True))
        if path == "functional_decode"
        else None,
        indexed_decode=(lambda layer, value, *args: KDAResult(value + 1, normed=True))
        if path == "indexed_decode"
        else None,
        indexed_prefill=(lambda layer, value, *args: value + 1)
        if path == "indexed_prefill"
        else None,
    )
    backend = KDAAttentionBackend(SimpleNamespace(), spec, cache=cache)
    batch = SimpleNamespace(
        is_decode=path in ("indexed_decode", "functional_decode"),
        reqs=[
            SimpleNamespace(extend_len=2 if path == "packed_fallback" else 1, table_idx=i)
            for i in range(2)
        ],
        attn_metadata=SimpleNamespace(state_indices=torch.arange(2)),
    )
    result = backend.forward_layer(
        Layer(), 0, x, batch, cache, torch.zeros(1, 2, dtype=torch.int32), residual=residual
    )
    assert finishes == [residual]
    assert result.eq(8).all()


@pytest.mark.parametrize("path", ["paged", "packed_prefill", "fallback"])
def test_mla_all_return_paths_forward_the_explicit_residual(monkeypatch, path):
    import minisgl.attention.mla.backend as module
    from minisgl.attention.mla import MLAOutput

    monkeypatch.setattr(module, "ContextParallel", lambda: SimpleNamespace(size=1, rank=0))
    x = torch.zeros(2, 3)
    residual = torch.full_like(x, 7)
    finishes = []

    class Layer:
        def project(self, value, positions, *, is_decode=False):
            return value, value, value[:, :1]

        def attend_paged_partial(self, q, *args, **kwargs):
            return MLAOutput(q + 1)

        def attend_partial(self, q, *args, **kwargs):
            return MLAOutput(q + 1)

        def finish(self, value, partial, *, residual=None):
            finishes.append(residual)
            return partial.values + residual

    cache = SimpleNamespace(
        device=x.device,
        store_mla=lambda *args, **kwargs: None,
        mla=SimpleNamespace(paged_storage=lambda *args: None),
    )
    metadata = SimpleNamespace(
        decode=object() if path == "paged" else None,
        prefill=object() if path == "packed_prefill" else None,
    )
    batch = SimpleNamespace(
        is_decode=path == "paged",
        positions=torch.arange(2),
        out_loc=torch.arange(2),
        reqs=[SimpleNamespace(extend_len=2, cached_len=0, device_len=2, table_idx=0)],
        attn_metadata=metadata,
    )
    attention = SimpleNamespace(prefill_batch=lambda *args, **kwargs: None)
    backend = module.MLAAttentionBackend(None, attention=attention, cache=cache)
    result = backend.forward_layer(
        Layer(), 0, x, batch, cache, torch.zeros(1, 2, dtype=torch.int32), residual=residual
    )
    assert finishes == [residual] and result.eq(8).all()


def test_explicit_row_output_tensor_is_partial_while_return_is_reduced():
    tp = TensorParallel()
    op = output_op(tp, role=None)
    x = torch.ones(2, 16, dtype=torch.bfloat16)
    partial = torch.empty_like(x)
    result = op.forward(x, output_tensor=partial, residual=torch.full_like(x, 3))
    assert partial.eq(1).all() and result.eq(5).all()
    assert result.data_ptr() != partial.data_ptr()
    assert tp.events == [("generic",)]


def test_legacy_mha_mlp_and_stateful_attention_share_one_row_implementation(monkeypatch):
    import minisgl.distributed.info as info
    import minisgl.models.utils as legacy
    from minisgl.attention.config import KDAConfig, MLAConfig
    from minisgl.distributed import DistributedInfo
    from minisgl.layers.kda import KDALayer
    from minisgl.layers.mla import MLALayer

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 2))
    monkeypatch.setattr(legacy, "AttentionLayer", lambda **kwargs: SimpleNamespace())
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        hidden_act="silu",
        head_dim=2,
        num_qo_heads=4,
        num_kv_heads=2,
        rotary_config=None,
    )
    mha = legacy.RopeAttn(config, 0)
    mlp = legacy.GatedMLP(config)
    kda = KDALayer(KDAConfig(8, 2, 4, 3, full_rank_gate=True))
    mla = MLALayer(MLAConfig(8, 2, 3, 2, 3, 5, None, 32))
    assert {type(mha.o_proj), type(mlp.down_proj), type(kda.o_proj), type(mla.o_proj)} == {
        LinearRowParallel
    }
    for op in (mha.o_proj, mlp.down_proj, kda.o_proj, mla.o_proj):
        assert isinstance(op, Linear)
        assert op.full_input_size == op.local_input_size * 2
        assert op.full_output_size == op.local_output_size == 8
        assert op.weight.shape == (op.local_output_size, op.local_input_size)
    assert kda.o_proj.local_input_size == kda._heads * kda._dim
    assert mla.o_proj.local_input_size == mla._heads * mla._value
    assert mha.o_proj._output_role is mlp.down_proj._output_role is None
    assert set(mha.state_dict()) == {"qkv_proj.weight", "o_proj.weight"}
