"""Construction identity and atomic static binding; no native arithmetic or GPU."""

import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.base import BaseAttnBackend, HybridAttnBackend, PrefillDecodeAttnBackend
from minisgl.attention.config import KDAConfig, MLAConfig
from minisgl.attention.kda.registry import get_kda_kernel
from minisgl.layers.kda import KDALayer
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM, KimiModel
from test_kda_checkpoint_tracking import case as case
from test_kimi_model import tiny_config


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CPU binding initialized CUDA")
    )
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def test_spawn_pickle_and_actual_decoder_config_identity_with_heterogeneous_bounds(monkeypatch):
    import minisgl.models.kimi_k3 as model_module

    config = ModelConfig.from_hf(tiny_config())
    records = tuple(
        (i, replace(c, gate_lower_bound=-2) if i == 2 else c) for i, c in config.attention_configs
    )
    config = pickle.loads(pickle.dumps(replace(config, attention_configs=records)))
    assert isinstance(config.attention_configs, tuple)
    config.hf_config.linear_attn_config["gate_lower_bound"] = -9  # Must not be rederived.

    def derive(*args):
        pytest.fail("Declared nonempty records were rederived from HF metadata")

    monkeypatch.setattr(model_module, "kda_config", derive)
    monkeypatch.setattr(model_module, "mla_config", derive)
    model = KimiK3LinearForCausalLM(config)
    declared = dict(config.attention_configs)
    for i, decoder in model.model.layers.items():
        assert decoder._layer_id == i and decoder.self_attn._config is declared[i]
    assert declared[0].gate_lower_bound == -5 and declared[2].gate_lower_bound == -2


def test_empty_records_keep_legacy_model_construction():
    config = replace(ModelConfig.from_hf(tiny_config()), attention_configs=())
    model = KimiK3LinearForCausalLM(config)
    assert [layer._is_kda for layer in model.model.layers.op_list] == [True, False, True]


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "duplicate",
        "wrong_kind",
        "bool_id",
        "none",
        "mutable",
        "inner_list",
        "short_pair",
        "long_pair",
    ],
)
def test_nonempty_decoder_records_must_cover_actual_model(invalid):
    config = ModelConfig.from_hf(tiny_config())
    records = list(config.attention_configs)
    if invalid == "missing":
        records.pop()
    elif invalid == "duplicate":
        records.append(records[-1])
    elif invalid == "wrong_kind":
        records[2] = (2, records[1][1])
    elif invalid == "none":
        records[2] = (2, None)
    elif invalid == "bool_id":
        records[1] = (True, records[1][1])
    elif invalid == "inner_list":
        records[2] = list(records[2])
    elif invalid == "short_pair":
        records[2] = (2,)
    elif invalid == "long_pair":
        records[2] = (*records[2], "extra")
    with pytest.raises(ValueError):
        KimiModel(config.hf_config, records if invalid == "mutable" else tuple(records))


def declared(case):
    first = case.configs[0]
    last = replace(case.configs[2], gate_lower_bound=-5)
    mla = MLAConfig(4, 1, 2, 2, 2, 4, None, 128)
    return ((2, last), (1, mla), (0, first))  # Explicit sparse recurrent IDs, reordered records.


def test_bound_nonadjacent_configs_are_readonly_and_consumed_by_explicit_checkpoint_entry(case):
    records = declared(case)
    case.backend.prepare_model(records)
    bound = case.backend.checkpoint_layer_configs
    assert tuple(bound) == (0, 2) and bound[2] is records[0][1]
    assert bound[0] is records[2][1] and bound[0].gate_lower_bound != bound[2].gate_lower_bound
    with pytest.raises(TypeError):
        bound[2] = bound[0]
    case.backend.prepare_checkpoints(case.batch)
    assert case.batch.attn_metadata.checkpoints.layer_configs[2] is records[0][1]
    assert case.backend.checkpoint_quantum() is None  # strict context has no auto capability


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_first",
        "missing_last",
        "duplicate",
        "wrong_type",
        "last_unsafe",
        "last_nan",
        "last_dimension",
        "wrong_partition",
        "bool_id",
        "mutable",
        "unsupported_kernel",
        "inner_list",
        "short_pair",
        "long_pair",
    ],
)
def test_incomplete_or_unsupported_model_disables_whole_binding_before_allocation_or_write(
    case, monkeypatch, invalid
):
    records = list(declared(case))
    if invalid == "missing_first":
        records.pop(2)
    elif invalid == "missing_last":
        records.pop(0)
    elif invalid == "duplicate":
        records.append(records[0])
    elif invalid == "wrong_type":
        records[0] = (2, SimpleNamespace())
    elif invalid == "last_unsafe":
        records[0] = (2, replace(records[0][1], gate_lower_bound=None))
    elif invalid == "last_nan":
        records[0] = (2, replace(records[0][1], gate_lower_bound=float("nan")))
    elif invalid == "last_dimension":
        records[0] = (2, replace(records[0][1], head_dim=64))
    elif invalid == "wrong_partition":
        records[0], records[1] = (1, records[0][1]), (2, records[1][1])
    elif invalid == "bool_id":
        records[1] = (True, records[1][1])
    elif invalid == "unsupported_kernel":
        case.backend.kernel = get_kda_kernel("sgl")
    elif invalid == "inner_list":
        records[0] = list(records[0])
    elif invalid == "short_pair":
        records[0] = (2,)
    elif invalid == "long_pair":
        records[0] = (*records[0], "extra")
    records = records if invalid == "mutable" else tuple(records)
    state, conv = case.states._recurrent.clone(), case.states._conv.clone()
    metadata = case.batch.attn_metadata

    def allocation(*args, **kwargs):
        pytest.fail("Static invalid binding allocated tensor metadata")

    monkeypatch.setattr(torch, "tensor", allocation)
    monkeypatch.setattr(torch, "arange", allocation)
    case.backend.prepare_model(records)
    native_only = invalid in ["last_unsafe", "last_nan", "last_dimension", "unsupported_kernel"]
    assert bool(case.backend.checkpoint_layer_configs) is native_only
    assert case.backend.checkpoint_quantum(case.batch) is None
    assert case.batch.attn_metadata is metadata and metadata.checkpoints is None
    with pytest.raises((ValueError, TypeError)):
        case.backend.prepare_checkpoints(case.batch)
    assert metadata.checkpoints is None and not case.states._pending
    assert torch.equal(state, case.states._recurrent) and torch.equal(conv, case.states._conv)


def test_unbound_empty_record_model_keeps_slice_a_explicit_entry(case):
    case.backend.prepare_model(())
    assert case.backend.checkpoint_layer_configs is None
    case.backend.prepare_checkpoints(case.batch, case.configs)
    assert case.batch.attn_metadata.checkpoints is not None


def test_static_global_heads_do_not_compare_against_tp_local_heads(case, monkeypatch):
    import minisgl.layers.kda as kda

    config = KDAConfig(4, 96, 128, 4, gate_lower_bound=-5)
    monkeypatch.setattr(kda, "TensorParallel", lambda: SimpleNamespace(size=8))
    layer = KDALayer(config)
    assert layer._config is config and layer._heads == 12
    case.backend.config.cache_config = replace(case.layout, num_heads=96)
    records = ((0, config), (1, declared(case)[1][1]), (2, replace(config, gate_lower_bound=-3)))
    case.backend.prepare_model(records)
    assert case.backend.checkpoint_layer_configs[0] is config


def test_phase_and_hybrid_forward_same_public_record_object_to_all_children():
    records = ((7, KDAConfig(4, 1, 128, 4, gate_lower_bound=-5)),)
    calls = []
    child = lambda name: SimpleNamespace(prepare_model=lambda value: calls.append((name, value)))
    prefill = HybridAttnBackend(child("kda"), child("mla"), lambda i: i == 7)
    phases = PrefillDecodeAttnBackend(prefill, child("decode"))
    phases.prepare_model(records)
    assert [name for name, _ in calls] == ["kda", "mla", "decode"]
    assert all(value is records for _, value in calls)
    BaseAttnBackend.prepare_model(None, ())  # Ordinary backends inherit a no-op.


@pytest.mark.parametrize("kind", ["phase", "hybrid"])
def test_shared_startup_child_is_initialized_once(kind):
    records = ((7, KDAConfig(4, 1, 128, 4, gate_lower_bound=-5)),)
    calls = []
    child = SimpleNamespace(prepare_model=lambda value: calls.append(value))
    backend = (
        PrefillDecodeAttnBackend(child, child)
        if kind == "phase"
        else HybridAttnBackend(child, child, lambda i: True)
    )
    backend.prepare_model(records)
    assert len(calls) == 1 and calls[0] is records
