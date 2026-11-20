"""Indexed Triton metadata/lease/ABI contracts; recurrence math is not emulated."""

from contextlib import nullcontext
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import checkpoints, prefill, projection, triton
from minisgl.attention.kda.backend import KDAAttentionBackend
from minisgl.attention.kda.registry import get_kda_kernel
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.recurrent_pool import RecurrentStatePool


@pytest.fixture
def case(monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU protocol test attempted CUDA initialization")

    monkeypatch.setattr("tvm_ffi.use_torch_stream", nullcontext)
    monkeypatch.setattr(torch.cuda, "init", no_cuda)
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(checkpoints, "_checkpoint_device_supported", lambda device: True)
    layout = HybridCacheConfig(1, (), (0,), 4, 2, 1, 128, 4)
    states = RecurrentStatePool(
        (0,),
        3,
        1,
        128,
        4,
        torch.bfloat16,
        "cpu",
        state_slot_capacity=6,
        state_dtype=torch.bfloat16,
    )
    cache = SimpleNamespace(recurrent=states, device=torch.device("cpu"), supports_overlap=False)
    ctx = core.Context(page_size=64)
    ctx.kv_cache = cache
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    reqs = []
    for row, length in enumerate((67, 65, 130)):
        states.reserve_request(row, 10 + row)
        reqs.append(
            Req(
                torch.arange(length, dtype=torch.int32), row, 0, 3, 10 + row, SamplingParams(), None
            )
        )
    plans = (
        states.reserve_checkpoint(2, 12, 128, end_len=130),
        states.reserve_checkpoint(0, 10, 64, end_len=67),
    )
    batch = Batch(reqs, "prefill")
    batch.checkpoint_plan = plans
    backend = KDAAttentionBackend(
        SimpleNamespace(cache_config=layout), get_kda_kernel("triton"), cache=cache
    )
    backend.prepare_metadata(batch)
    config = KDAConfig(4, 1, 128, 4, gate_lower_bound=-2)
    backend.prepare_checkpoints(batch, {0: config})
    states._conv.fill_(-0.5)
    states._recurrent.fill_(0.25)
    for req in reqs:
        states.begin_request(req.table_idx, req.uid, 0)
    yield SimpleNamespace(
        states=states, batch=batch, backend=backend, cache=cache, config=config, plans=plans
    )
    for req in reqs:
        states.release_request(req.table_idx, req.uid)
    assert not torch.cuda.is_initialized()


def spies(case, monkeypatch):
    total = sum(case.batch.attn_metadata.lengths)
    q = (torch.arange(total * 128).reshape(total, 128) % 13).to(torch.bfloat16)
    k, v = q + 16, q + 32
    raw = torch.cat((q, k, v), dim=-1)
    layer = SimpleNamespace(
        _config=case.config,
        _heads=1,
        _dim=128,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(128),
        q_conv1d=SimpleNamespace(weight=torch.zeros(128, 1, 4, dtype=torch.bfloat16)),
        k_conv1d=SimpleNamespace(weight=torch.zeros(128, 1, 4, dtype=torch.bfloat16)),
        v_conv1d=SimpleNamespace(weight=torch.zeros(128, 1, 4, dtype=torch.bfloat16)),
        prepare_kernel=lambda name, create: create(),
        finish=lambda x, output, **kw: output,
    )
    monkeypatch.setattr(prefill, "supports", lambda *args: True)
    monkeypatch.setattr(
        projection,
        "project",
        lambda *args: (
            q,
            k,
            v,
            torch.zeros(total, 1, 128, dtype=torch.bfloat16),
            torch.zeros(total, 1, dtype=torch.bfloat16),
        ),
    )
    events, calls, snapshots = [], [], []

    def conv(inputs, weights, bias, state, **kwargs):
        events.append("conv")
        tracked = case.batch.attn_metadata.checkpoints
        for plan, selected in zip(tracked.plans, tracked.conv_indices):
            saved = case.states._checkpoints[plan.generation][0][0]
            expected = raw[selected].view(3, 3, 128).permute(1, 2, 0)
            assert torch.equal(saved, expected)
        for index in kwargs["cache_indices"].tolist():
            state[index].fill_(99)
        return inputs

    def chunk(**kwargs):
        events.append("chunk")
        calls.append(kwargs)
        assert kwargs["initial_state"].data_ptr() == case.states.indexed_kda_state(0)[1].data_ptr()
        assert kwargs["initial_state"].dtype == torch.bfloat16
        assert kwargs["initial_state_indices"].tolist() == [0, 1, 2]
        assert kwargs["chunk_offsets"].tolist() == [0, 2, 4, 7]
        assert kwargs["chunk_indices"].tolist() == [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
            [2, 2],
        ]
        assert kwargs["track_chunk_idx"].tolist() == [1, -1, 2]
        scratch = kwargs["track_state"]
        assert scratch.shape == (3, 1, 128, 128) and scratch.dtype == torch.float32
        scratch.fill_(float("nan"))  # Untracked row must never be copied.
        for row in (0, 2):
            # ABI marker, not a KDA numerical reference; asymmetry catches transpose.
            marker = (torch.arange(128 * 128).reshape(1, 128, 128) % 17).float() / 8 + row * 4
            scratch[row].copy_(marker)
            snapshots.append((row, marker.clone()))
        for index in kwargs["initial_state_indices"].tolist():
            kwargs["initial_state"][index].fill_(7)
        return kwargs["v"]

    def kernels():
        events.append("resolve")
        return chunk, None, nullcontext

    monkeypatch.setattr(triton, "_kernels", kernels)
    monkeypatch.setattr(prefill, "_conv_kernel", lambda: conv)
    return layer, events, calls, snapshots


def test_original_batch_rows_local_chunk_boundaries_and_selected_transpose_copy(case, monkeypatch):
    layer, events, calls, snapshots = spies(case, monkeypatch)
    output = case.backend.forward_layer(
        layer, 0, torch.zeros(262, 4, dtype=torch.bfloat16), case.batch, case.cache, None
    )
    assert output.shape == (262, 1, 128) and len(calls) == 1
    assert events.index("resolve") < events.index("conv") < events.index("chunk")
    assert calls[0]["beta_is_raw"] is True and calls[0]["use_qk_l2norm_in_kernel"] is True
    assert calls[0]["lower_bound"] == -2 and "output_intermediate_states" not in calls[0]
    expected = dict(snapshots)
    assert case.batch.attn_metadata.checkpoints.rows == (2, 0)
    for plan, row in zip(case.plans, (2, 0)):
        saved = case.states._checkpoints[plan.generation][1][0]
        assert saved.dtype == torch.bfloat16
        assert torch.equal(saved, expected[row].transpose(-1, -2).to(torch.bfloat16))
        assert not torch.equal(saved, expected[row].to(torch.bfloat16))
        assert case.states._pending[plan.generation].written == {0}
    assert case.states._owners[:3] == [10, 11, 12]
    assert torch.equal(
        case.states.indexed_kda_state(0)[1][:3],
        torch.full((3, 1, 128, 128), 7, dtype=torch.bfloat16),
    )


def test_warm_frontier_is_relative_local_chunk_not_global_chunk_offset():
    metadata = SimpleNamespace(
        lengths=(65, 130),
        kernel_data={},
        checkpoints=SimpleNamespace(
            rows=(1,),
            plans=(SimpleNamespace(start_len=64, checkpoint=SimpleNamespace(length=128)),),
            quantum=64,
        ),
    )
    plan = triton._prefill_plan(metadata, torch.device("cpu"))
    assert plan["chunk_offsets"].tolist() == [0, 2, 5]
    assert plan["track_chunk_idx"].tolist() == [-1, 1]  # not global3, not after-chunk0
    assert triton._prefill_plan(metadata, torch.device("cpu")) is plan


@pytest.mark.parametrize(
    "failure", ["support", "kernel", "metadata", "lease", "scratch", "conv", "recurrence"]
)
def test_prewrite_rejection_and_selected_failure_never_retry(case, monkeypatch, failure):
    layer, events, calls, _ = spies(case, monkeypatch)
    before_conv, before_ssm = case.states._conv.clone(), case.states._recurrent.clone()

    def fail(*args, **kwargs):
        raise ValueError("selected checkpoint failure")

    if failure == "support":
        monkeypatch.setattr(prefill, "supports", lambda *args: False)
    elif failure == "kernel":
        monkeypatch.setattr(triton, "_kernels", fail)
    elif failure == "metadata":
        case.batch.attn_metadata.lengths = (66, 65, 130)
    elif failure == "lease":
        borrow = case.states.borrow_checkpoint
        monkeypatch.setattr(
            case.states,
            "borrow_checkpoint",
            lambda plan, layer_id: fail() if plan == case.plans[1] else borrow(plan, layer_id),
        )
    elif failure == "scratch":
        original = torch.Tensor.new_empty

        def allocation(tensor, *args, **kwargs):
            if kwargs.get("dtype") == torch.float32:
                return fail()
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "new_empty", allocation)
    elif failure == "conv":
        monkeypatch.setattr(prefill, "_conv_kernel", lambda: fail)
    else:
        monkeypatch.setattr(triton, "_kernels", lambda: (fail, None, nullcontext))
    with pytest.raises(ValueError):
        case.backend.forward_layer(
            layer, 0, torch.zeros(262, 4, dtype=torch.bfloat16), case.batch, case.cache, None
        )
    assert len(calls) == 0 and not case.states._pending
    if failure in ("support", "kernel", "metadata", "lease", "scratch"):
        assert torch.equal(case.states._conv, before_conv)
        assert torch.equal(case.states._recurrent, before_ssm)
    else:
        assert events.count("conv") <= 1  # selected failure cannot repeat state advancement


def test_explicit_sgl_has_shared_checkpoint_hooks_without_auto_selector_change():
    sgl_spec, fi_spec = get_kda_kernel("triton"), get_kda_kernel("flashinfer")
    assert sgl_spec.indexed_prefill is not None
    assert sgl_spec.prepare_checkpoints is fi_spec.prepare_checkpoints
    assert sgl_spec.checkpoint_quantum is fi_spec.checkpoint_quantum
    assert sgl_spec.indexed_decode_padding is True


def test_prepared_checkpoint_cannot_borrow_or_write_inside_capture(case, monkeypatch):
    layer, events, calls, _ = spies(case, monkeypatch)
    conv, recurrent = case.states._conv.clone(), case.states._recurrent.clone()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def forbidden(*args, **kwargs):
        pytest.fail("Capture rejection reached a kernel or checkpoint borrow")

    monkeypatch.setattr(triton, "_kernels", forbidden)
    monkeypatch.setattr(case.states, "borrow_checkpoint", forbidden)
    with pytest.raises(ValueError, match="eager prefill"):
        case.backend.forward_layer(
            layer, 0, torch.zeros(262, 4, dtype=torch.bfloat16), case.batch, case.cache, None
        )
    assert not events and not calls and not case.states._pending
    assert torch.equal(case.states._conv, conv)
    assert torch.equal(case.states._recurrent, recurrent)
