"""CPU contract gates for explicit native tracking, with numerical calls spied."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import flashinfer, prefill, sgl
from minisgl.attention.kda.backend import KDAAttentionBackend
from minisgl.attention.kda.registry import get_kda_kernel
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.recurrent_pool import RecurrentStatePool

LAYERS = (0, 2)


@pytest.fixture(params=[torch.bfloat16, torch.float32])
def case(monkeypatch, request):
    layout = HybridCacheConfig(3, (1,), LAYERS, 4, 2, 1, 128, 4)
    states = RecurrentStatePool(
        LAYERS,
        2,
        1,
        128,
        4,
        torch.bfloat16,
        "cpu",
        max_cached_states=3,
        state_dtype=request.param,
    )
    cache = SimpleNamespace(recurrent=states, device=torch.device("cpu"), supports_overlap=False)
    ctx = core.Context(page_size=64)
    ctx.kv_cache = cache
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    states.reserve_request(0, 10)
    req = Req(torch.arange(67, dtype=torch.int32), 0, 0, 3, 10, SamplingParams(), None)
    batch = Batch([req], "prefill")
    backend = KDAAttentionBackend(
        SimpleNamespace(cache_config=layout), get_kda_kernel("flashinfer"), cache=cache
    )
    backend.prepare_metadata(batch)
    plan = states.reserve_checkpoint(0, 10, 64, end_len=67)
    batch.checkpoint_plan = (plan,)
    cfg = KDAConfig(4, 1, 128, 4, gate_lower_bound=-2)
    configs = dict.fromkeys(LAYERS, cfg)
    # Metadata/ABI tests use CPU tensors; native SM/capture selection is isolated.
    monkeypatch.setattr(flashinfer, "_checkpoint_device_supported", lambda device: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    return SimpleNamespace(
        layout=layout,
        states=states,
        cache=cache,
        req=req,
        batch=batch,
        backend=backend,
        plan=plan,
        configs=configs,
    )


def test_compact_metadata_maps_only_selected_boundary_without_pool_views(case):
    case.backend.prepare_checkpoints(case.batch, case.configs)
    metadata = case.batch.attn_metadata.checkpoints
    assert metadata.plans == (case.plan,) and metadata.rows == (0,)
    assert metadata.cu_starts.tolist() == [0, 1]
    assert metadata.destinations.tolist() == [0]
    assert metadata.conv_indices.tolist() == [[61, 62, 63]]
    assert metadata.track_rows.tolist() == [0]
    for tensor in (
        metadata.conv_indices,
        metadata.cu_starts,
        metadata.destinations,
        metadata.track_rows,
    ):
        assert tensor.is_contiguous()
    assert metadata.destinations.dtype == torch.int32
    assert metadata.cu_starts.dtype == torch.int64
    assert metadata.conv_indices.dtype == metadata.track_rows.dtype == torch.int64
    assert not any("view" in key or "state" in key for key in vars(metadata))
    assert case.backend.checkpoint_quantum() is None  # no construction-bound portfolio


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_layer",
        "wrong_type",
        "later_unsafe_gate",
        "wrong_dimension",
        "all_t1",
        "end_changed",
        "overlap",
        "decode",
        "unsupported_kernel",
        "capture",
    ],
)
def test_all_layer_or_plan_rejection_is_atomic_and_releases_leases(case, monkeypatch, invalid):
    configs = dict(case.configs)
    if invalid == "missing_layer":
        configs.pop(2)
    elif invalid == "wrong_type":
        configs[2] = SimpleNamespace(**vars(configs[2]))
    elif invalid == "later_unsafe_gate":
        configs[2] = replace(configs[2], gate_lower_bound=None)
    elif invalid == "wrong_dimension":
        configs[2] = replace(configs[2], head_dim=64)
    elif invalid == "all_t1":
        case.req.device_len = 1
        case.batch.attn_metadata.lengths = (1,)
    elif invalid == "end_changed":
        case.req.device_len = 68
        case.batch.attn_metadata.lengths = (68,)
    elif invalid == "overlap":
        case.cache.supports_overlap = True
    elif invalid == "decode":
        case.batch.phase = "decode"
    elif invalid == "unsupported_kernel":
        case.backend.kernel = get_kda_kernel("sgl")
    elif invalid == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def unexpected(*args, **kwargs):
        pytest.fail("Rejected checkpoint preparation allocated tensor metadata")

    monkeypatch.setattr(torch, "tensor", unexpected)
    monkeypatch.setattr(torch, "arange", unexpected)
    with pytest.raises((ValueError, TypeError)):
        case.backend.prepare_checkpoints(case.batch, configs)
    assert case.batch.attn_metadata.checkpoints is None
    assert not case.states._pending and not case.states._checkpoints
    assert case.states._owners[0] == 10


def test_nonempty_unvalidated_plan_fails_before_any_numerical_call(case, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Unvalidated checkpoint reached numerical dispatch")

    case.backend.kernel = replace(case.backend.kernel, indexed_prefill=unexpected)
    layer = SimpleNamespace(_config=case.configs[0])
    with pytest.raises(ValueError, match="explicit validated"):
        case.backend.forward_layer(layer, 0, None, case.batch, case.cache, None)
    assert not case.states._pending and not case.states._checkpoints


def native_spies(case, monkeypatch):
    records = []
    prepared = []
    width, tokens = 128, case.req.extend_len
    q = (torch.arange(tokens * width).reshape(tokens, width) % 13).to(torch.bfloat16)
    k, v = q + 16, q + 32
    raw = torch.cat((q, k, v), dim=-1)
    layer = SimpleNamespace(
        _config=case.configs[0],
        _heads=1,
        _dim=128,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(width),
        q_conv1d=SimpleNamespace(weight=torch.zeros(width, 1, 4, dtype=torch.bfloat16)),
        k_conv1d=SimpleNamespace(weight=torch.zeros(width, 1, 4, dtype=torch.bfloat16)),
        v_conv1d=SimpleNamespace(weight=torch.zeros(width, 1, 4, dtype=torch.bfloat16)),
        prepare_kernel=lambda name, create: create(),
        finish=lambda x, output: output,
    )
    monkeypatch.setattr(sgl, "_kernels", lambda: (None, None, nullcontext))
    monkeypatch.setattr(
        sgl,
        "project",
        lambda layer, x: (
            q,
            k,
            v,
            torch.zeros(tokens, 1, width, dtype=torch.bfloat16),
            torch.zeros(tokens, 1, dtype=torch.bfloat16),
        ),
    )
    monkeypatch.setattr(prefill, "supports", lambda *args: True)

    def conv_fn(inputs, weights, bias, state, **kwargs):
        tracked = case.batch.attn_metadata.checkpoints
        if tracked is not None:
            saved = case.states._checkpoints[case.plan.generation][0]
            expected = raw[61:64].view(3, 3, width).permute(1, 2, 0)
            assert torch.equal(
                saved[len(records)], expected
            )  # before convolution updates working history
        for index in kwargs["cache_indices"].tolist():
            state[index].fill_(99)
        return inputs

    monkeypatch.setattr(prefill, "_conv_kernel", lambda: conv_fn)
    real_prepare = prefill.prepare

    def prepare(*args, **kwargs):
        prepared.append(dict(kwargs))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(prefill, "prepare", prepare)

    def extend(q, k, v, g, beta, **kwargs):
        entry = {"kwargs": dict(kwargs)}
        records.append(entry)
        # Asymmetric dyadic V,K matrix detects a missing semantic transpose.
        native = ((torch.arange(128 * 128).reshape(1, 128, 128) % 19) / 8).to(torch.bfloat16)
        native = (native + (len(records) - 1) * 4).to(kwargs["ssm_states"].dtype)
        entry["snapshot"] = native.clone()
        for index in kwargs["cache_indices"].tolist():
            kwargs["ssm_states"][index].copy_(native + 8)
        output = q + k + v
        if kwargs["return_intermediate_states"]:
            assert kwargs["num_state_checkpoints"] == 1
            assert kwargs["state_checkpoint_every_n_tokens"] == 64
            kwargs["track_state"].copy_(native.float().unsqueeze(0))
            return output, None
        return output

    monkeypatch.setattr(flashinfer, "_prefill_kernel", lambda: SimpleNamespace(extend=extend))
    return layer, raw, records, prepared


def test_native_tuple_axis_and_preconv_pair_restore_matches_written_state(case, monkeypatch):
    case.backend.prepare_checkpoints(case.batch, case.configs)
    layer, raw, records, _ = native_spies(case, monkeypatch)
    case.states.begin_request(0, 10, 0)
    for layer_id in LAYERS:
        result = case.backend.forward_layer(
            layer,
            layer_id,
            torch.zeros(67, 4, dtype=torch.bfloat16),
            case.batch,
            case.cache,
            None,
        )
        assert result.shape == (67, 1, 128)  # unpack tuple, then remove singleton batch
    case.states.finish_request(0, 10, 67)
    checkpoint = case.states.seal_checkpoint(case.plan)
    case.states.adopt_staged(checkpoint)
    case.states.restore(1, 20, checkpoint)
    case.states.begin_request(1, 20, 64)
    for layer_id, record in zip(LAYERS, records):
        restored = case.states.get_kda_state(layer_id, 1)
        assert torch.equal(restored.recurrent, record["snapshot"].transpose(-1, -2))
        assert not torch.equal(restored.recurrent, record["snapshot"])
        assert torch.equal(restored.conv, raw[61:64].view(3, 3, 128).permute(1, 2, 0))
    case.states.release_request(0, 10)
    case.states.release_request(1, 20)
    case.states.drop_checkpoint(checkpoint)


def test_empty_plan_has_no_tracking_allocations_or_extra_native_keywords(case, monkeypatch):
    case.states.abort_checkpoint(case.plan)
    case.batch.checkpoint_plan = ()
    layer, _, records, prepared = native_spies(case, monkeypatch)
    monkeypatch.setattr(
        torch.Tensor,
        "new_empty",
        lambda *args, **kwargs: pytest.fail("Empty plan allocated tracking scratch"),
    )
    # Even an unsupported layers map is ignored on the unchanged empty path.
    case.backend.prepare_checkpoints(case.batch, {})
    case.states.begin_request(0, 10, 0)
    result = case.backend.forward_layer(
        layer,
        0,
        torch.zeros(67, 4, dtype=torch.bfloat16),
        case.batch,
        case.cache,
        None,
    )
    assert result.shape == (67, 1, 128)
    assert len(records) == 1 and prepared == [{}]
    assert set(records[0]["kwargs"]) == {
        "ssm_states",
        "cache_indices",
        "query_start_loc",
        "A_log",
        "dt_bias",
        "lower_bound",
        "return_intermediate_states",
    }
    assert records[0]["kwargs"]["return_intermediate_states"] is False
    case.states.release_request(0, 10)


@pytest.mark.parametrize("failure", ["conv", "recurrence", "finish", "changed_config"])
def test_any_tracked_layer_failure_releases_undonated_plans(case, monkeypatch, failure):
    case.backend.prepare_checkpoints(case.batch, case.configs)
    layer, _, _, _ = native_spies(case, monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("component failure")

    if failure == "conv":
        monkeypatch.setattr(prefill, "_conv_kernel", lambda: fail)
    elif failure == "recurrence":
        monkeypatch.setattr(flashinfer, "_prefill_kernel", lambda: SimpleNamespace(extend=fail))
    elif failure == "finish":
        layer.finish = fail
    else:
        layer._config = replace(layer._config, gate_lower_bound=None)
    case.states.begin_request(0, 10, 0)
    with pytest.raises(ValueError):
        case.backend.forward_layer(
            layer,
            0,
            torch.zeros(67, 4, dtype=torch.bfloat16),
            case.batch,
            case.cache,
            None,
        )
    assert not case.states._pending and not case.states._checkpoints
    case.states.release_request(0, 10)


def test_compact_map_handles_permuted_request_subset_and_skipped_boundaries(case):
    second = Req(torch.arange(130, dtype=torch.int32), 1, 0, 3, 20, SamplingParams(), None)
    case.states.reserve_request(1, 20)
    second_plan = case.states.reserve_checkpoint(1, 20, 128, end_len=130)
    batch = Batch([case.req, second], "prefill")
    batch.checkpoint_plan = (second_plan, case.plan)
    case.backend.prepare_metadata(batch)
    case.backend.prepare_checkpoints(batch, case.configs)
    tracked = batch.attn_metadata.checkpoints
    assert tracked.rows == (1, 0)
    assert tracked.cu_starts.tolist() == [0, 1, 3]
    assert tracked.destinations.tolist() == [1, -1, 0]
    assert tracked.conv_indices.tolist() == [[192, 193, 194], [61, 62, 63]]
    for plan in batch.checkpoint_plan:
        case.states.abort_checkpoint(plan)


def test_warm_plan_uses_relative_kernel_offset_without_rewinding_request(case):
    case.states.abort_checkpoint(case.plan)
    case.states.begin_request(0, 10, 0)
    case.states.finish_request(0, 10, 64)
    case.req.cached_len = 64
    case.req.device_len = 131
    case.req.input_ids = torch.arange(131, dtype=torch.int32)
    plan = case.states.reserve_checkpoint(0, 10, 128, end_len=131)
    case.batch.checkpoint_plan = (plan,)
    case.backend.prepare_metadata(case.batch)
    case.backend.prepare_checkpoints(case.batch, case.configs)
    tracked = case.batch.attn_metadata.checkpoints
    assert tracked.conv_indices.tolist() == [[61, 62, 63]]
    assert tracked.cu_starts.tolist() == [0, 1]
    assert case.req.cached_len == 64 and case.req.device_len == 131
    case.states.abort_checkpoint(plan)


@pytest.mark.parametrize(
    "invalid", ["nan_gate", "device", "duplicate", "storage", "missing_lengths"]
)
def test_additional_prewrite_rejection_has_no_tensor_allocation(case, monkeypatch, invalid):
    configs = dict(case.configs)
    if invalid == "nan_gate":
        configs[2] = replace(configs[2], gate_lower_bound=float("nan"))
    elif invalid == "device":
        monkeypatch.setattr(flashinfer, "_checkpoint_device_supported", lambda device: False)
    elif invalid == "duplicate":
        case.batch.checkpoint_plan = (case.plan, case.plan)
    elif invalid == "storage":
        case.states._conv_dtype = torch.float32
    else:
        case.batch.attn_metadata.lengths = ()

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid prewrite metadata allocated a tensor")

    monkeypatch.setattr(torch, "tensor", unexpected)
    monkeypatch.setattr(torch, "arange", unexpected)
    with pytest.raises(ValueError):
        case.backend.prepare_checkpoints(case.batch, configs)
    assert case.batch.attn_metadata.checkpoints is None
    assert not case.states._pending and not case.states._checkpoints


@pytest.mark.parametrize("invalid", ["already_written", "advanced_working"])
def test_preparation_rejects_nonfresh_forward_lease_before_metadata_allocation(
    case, monkeypatch, invalid
):
    case.states.begin_request(0, 10, 0)
    if invalid == "already_written":
        with case.states.borrow_checkpoint(case.plan, 0) as (conv, state):
            conv.zero_()
            state.zero_()
    else:
        case.states.finish_request(0, 10, 1)

    def unexpected(*args, **kwargs):
        pytest.fail("Nonfresh lease allocated tracking metadata")

    monkeypatch.setattr(torch, "tensor", unexpected)
    monkeypatch.setattr(torch, "arange", unexpected)
    with pytest.raises(ValueError):
        case.backend.prepare_checkpoints(case.batch, case.configs)
    assert case.batch.attn_metadata.checkpoints is None
    assert not case.states._pending and not case.states._checkpoints
    case.states.release_request(0, 10)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mismatch", [False, True])
def test_real_k3_configs_keep_global96_while_tp8_pool_and_layer_use_local12(
    monkeypatch, state_dtype, mismatch
):
    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo
    from minisgl.layers.kda import KDALayer
    from minisgl.models.kimi_config import cache_config, kda_config

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 8))
    model = SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=3,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5,
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
        },
    )
    layout = cache_config(model)
    config = kda_config(model)
    layer = KDALayer(config)
    assert layout.num_heads == layer._config.num_heads == 96 and layer._heads == 12
    states = RecurrentStatePool(
        LAYERS,
        1,
        layer._heads,
        128,
        4,
        torch.bfloat16,
        "cpu",
        max_cached_states=1,
        state_dtype=state_dtype,
    )
    cache = SimpleNamespace(recurrent=states, device=torch.device("cpu"), supports_overlap=False)
    ctx = core.Context(page_size=512)
    ctx.kv_cache = cache
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    states.reserve_request(0, 31)
    req = Req(torch.arange(513, dtype=torch.int32), 0, 0, 2, 31, SamplingParams(), None)
    batch = Batch([req], "prefill")
    batch.checkpoint_plan = (states.reserve_checkpoint(0, 31, 512, end_len=513),)
    backend = KDAAttentionBackend(
        SimpleNamespace(cache_config=layout), get_kda_kernel("flashinfer"), cache=cache
    )
    backend.prepare_metadata(batch)
    configs = dict.fromkeys(LAYERS, layer._config)
    if mismatch:
        configs[2] = replace(configs[2], num_heads=104)
    monkeypatch.setattr(flashinfer, "_checkpoint_device_supported", lambda device: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    if mismatch:
        with pytest.raises(ValueError, match="Every checkpoint layer"):
            backend.prepare_checkpoints(batch, configs)
        assert not states._pending and batch.attn_metadata.checkpoints is None
    else:
        backend.prepare_checkpoints(batch, configs)
        metadata = batch.attn_metadata.checkpoints
        assert metadata.layer_configs[0].num_heads == 96
        assert states.indexed_kda_state(0)[1].shape[1] == 12
        assert metadata.destinations.tolist() == [-1] * 7 + [0]
        with pytest.raises(TypeError):
            metadata.layer_configs[0] = replace(config, num_heads=104)
        states.abort_checkpoint(batch.checkpoint_plan[0])
    states.release_request(0, 31)
