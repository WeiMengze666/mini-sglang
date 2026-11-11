"""Automatic wiring CPU protocol tests; no numerical GPU kernels are invoked."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.attention.base import BaseAttnBackend, HybridAttnBackend, PrefillDecodeAttnBackend
from minisgl.core import Batch
from minisgl.engine.engine import _prepare_checkpoint_model
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.scheduler.scheduler import ForwardData, Scheduler
from test_checkpoint_writes import admit, allocate_pages, make_manager, write_all
from test_kda_checkpoint_tracking import case as case

SUPPORTED = SimpleNamespace(checkpoint_quantum=lambda batch=None: 64)


def _startup(q=64, *, error=False):
    def prepare(records):
        if error:
            raise ValueError("local binding failure")

    return SimpleNamespace(prepare_model=prepare, checkpoint_quantum=lambda: q)


def test_optional_capability_is_prefill_only_and_composes_without_decode_query():
    default = SimpleNamespace(checkpoint_quantum=BaseAttnBackend.checkpoint_quantum)
    assert BaseAttnBackend.checkpoint_quantum(default) is None
    decode = SimpleNamespace(checkpoint_quantum=lambda batch: pytest.fail("decode capability"))
    prefill = SimpleNamespace(checkpoint_quantum=lambda batch: 64)
    phase = PrefillDecodeAttnBackend(prefill, decode)
    hybrid = HybridAttnBackend(phase, decode, lambda _: True)
    assert hybrid.checkpoint_quantum(None) == 64


def test_startup_bind_errors_and_noncanonical_quantum_reject_without_gpu():
    assert _prepare_checkpoint_model(_startup(), (), None, ("bf16",)) == 64
    assert _prepare_checkpoint_model(_startup(None), (), None, ("bf16",)) is None
    with pytest.raises(RuntimeError, match="local binding failure"):
        _prepare_checkpoint_model(_startup(error=True), (), None, ("bf16",))
    with pytest.raises(RuntimeError, match="quantum"):
        _prepare_checkpoint_model(_startup(0), (), None, ("bf16",))
    assert not torch.cuda.is_initialized()


def _gloo_worker(rank, world, rendezvous, output):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=25),
    )
    group = dist.new_group(list(range(world)), backend="gloo", timeout=timedelta(seconds=25))
    try:
        assert _prepare_checkpoint_model(_startup(128), (), group, ("bf16",)) == 128
        assert (
            _prepare_checkpoint_model(_startup(128 if rank % 2 else 64), (), group, ("bf16",))
            is None
        )
        assert (
            _prepare_checkpoint_model(_startup(None if rank == 1 else 64), (), group, ("bf16",))
            is None
        )
        for kind in ("error", "signature"):
            try:
                _prepare_checkpoint_model(
                    _startup(error=kind == "error" and rank == 2),
                    (),
                    group,
                    ("bf16", rank if kind == "signature" else 0),
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("all ranks must reject startup " + kind)
        with pytest.MonkeyPatch.context() as patch:
            manager = make_manager(patch, False)
            manager._checkpoint_cpu_group = group
            manager._checkpoint_quantum = 64
            states = manager.pool.recurrent
            states._next_checkpoint = rank * 100
            reqs = [admit(manager, slot=i, uid=10 + i, end=130 + i * 64) for i in range(2)]
            original = states.reserve_checkpoint
            states.reserve_checkpoint = lambda table, *a, **kw: (
                None if rank == 0 and table == 1 else original(table, *a, **kw)
            )
            batch = Batch(reqs, "prefill")
            plans = manager.plan_checkpoints(batch, SUPPORTED)
            assert len(plans) == 1 and plans[0].table_idx == 0
            # Physical generations deliberately differ: only logical rows must agree.
            assert plans[0].checkpoint.slot == rank * 100
            manager.checkpoint_metadata_ready(batch)
            manager.release_checkpoint_plan(plans)
            assert not states._pending
            states.reserve_checkpoint = original
            for kind in ("empty", "allocation", "signature", "metadata"):
                batch = Batch(reqs, "prefill")
                local = (
                    SimpleNamespace(checkpoint_quantum=lambda batch: None)
                    if kind == "empty" and rank == 1
                    else SUPPORTED
                )
                if kind == "signature" and rank == 2:
                    reqs[0].device_len += 1
                if kind == "allocation":

                    def reserve(*args, **kwargs):
                        if rank == 2:
                            raise ValueError("allocation guard")
                        return original(*args, **kwargs)

                    states.reserve_checkpoint = reserve
                try:
                    result = manager.plan_checkpoints(batch, local)
                    if kind == "empty":
                        assert result == () and not states._pending
                    if kind == "metadata":
                        manager.checkpoint_metadata_ready(
                            batch, ValueError("local metadata") if rank == 2 else None
                        )
                except RuntimeError:
                    assert kind in ("allocation", "signature", "metadata")
                else:
                    assert kind == "empty"
                assert not states._pending and not states._checkpoints
                states.reserve_checkpoint = original
                if kind == "signature" and rank == 2:
                    reqs[0].device_len -= 1
            assert not torch.cuda.is_initialized()
        Path(output, f"rank{rank}.ok").write_text(
            "startup/mask/empty/error/signature/metadata passed\n"
        )
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


def test_real_four_rank_gloo_common_mask_errors_and_logical_signatures(tmp_path):
    mp.spawn(_gloo_worker, args=(4, str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=4)
    assert len(list(tmp_path.glob("rank*.ok"))) == 4


def test_separate_cold_admission_and_disabled_scheduler_plan_have_no_scratch(monkeypatch):
    manager = make_manager(monkeypatch, False)
    req = admit(manager, end=130)
    assert manager.pool.recurrent._owners[req.table_idx] == req.uid
    assert manager.pool.recurrent.max_staged_states == 0
    assert not manager.coordinates_checkpoints(Batch([req], "prefill"))
    manager._checkpoint_quantum = 64
    assert manager.coordinates_checkpoints(Batch([req], "prefill"))
    assert not manager.coordinates_checkpoints(Batch([req], "decode"))


@pytest.mark.parametrize("failure", [None, "layer", "partial_seal"])
def test_hybrid_is_only_seal_owner_and_rolls_back_complete_batch(monkeypatch, failure):
    manager = make_manager(monkeypatch, False)
    reqs = [admit(manager, slot=i, uid=10 + i, end=130) for i in range(2)]
    batch = Batch(reqs, "prefill")
    plans = manager.plan_checkpoints(batch, SUPPORTED)
    states = manager.pool.recurrent
    pool = object.__new__(HybridCache)
    pool.recurrent = states
    if failure:
        with pytest.raises(ValueError if failure == "layer" else RuntimeError):
            with pool.forward_batch(batch):
                for i, plan in enumerate(plans):
                    if failure != "partial_seal" or i == 0:
                        write_all(states, plan)
                if failure == "layer":
                    raise ValueError("actual model failure")
        assert not states._pending and not states._checkpoints
        assert all(owner is None for owner in states._owners)
    else:
        with pool.forward_batch(batch):
            for plan in plans:
                write_all(states, plan)
        assert all(states._pending[p.generation].sealed for p in plans)
        manager.release_checkpoint_plan(plans)


def test_cleanup_error_never_masks_original_and_aborts_remaining_leases(monkeypatch):
    manager = make_manager(monkeypatch, False)
    reqs = [admit(manager, slot=i, uid=10 + i, end=130) for i in range(2)]
    batch = Batch(reqs, "prefill")
    plans = manager.plan_checkpoints(batch, SUPPORTED)
    original = manager.pool.recurrent.abort_checkpoint

    def abort(plan):
        if plan == plans[0]:
            raise RuntimeError("cleanup failure")
        original(plan)

    monkeypatch.setattr(manager.pool.recurrent, "abort_checkpoint", abort)
    error = ValueError("original")
    manager.release_checkpoint_plan(plans, error=error)
    assert "cleanup failure" in error.__notes__[0]
    assert plans[1].generation not in manager.pool.recurrent._pending
    original(plans[0])


@pytest.mark.parametrize("excluded", [False, True])
def test_result_event_precedes_donation_and_ordinary_cache_and_lease_cleanup(monkeypatch, excluded):
    from minisgl.scheduler.prefill import ChunkedReq

    manager = make_manager(monkeypatch, False)
    manager.num_pages = 128
    manager.free_slots = torch.arange(128, dtype=torch.int32) * manager.page_size
    manager.page_table = torch.full((3, 256), -1, dtype=torch.int32)
    req = admit(manager, end=131)
    # Chunked request exercises the first ordinary cache branch with an unaligned tail.
    req = ChunkedReq(
        req.input_ids,
        req.table_idx,
        req.cached_len,
        3,
        req.uid,
        req.sampling_params,
        req.cache_handle,
    )
    allocate_pages(manager, req)
    batch = Batch([req], "prefill")
    plans = manager.plan_checkpoints(batch, SUPPORTED)
    states = manager.pool.recurrent
    pool = object.__new__(HybridCache)
    pool.recurrent = states
    with pool.forward_batch(batch):
        write_all(states, plans[0])
    req.cached_len = req.device_len
    calls = []
    scheduler = object.__new__(Scheduler)
    scheduler.cache_manager = manager
    scheduler.finished_reqs = set()
    scheduler.aborted_reqs = {req} if excluded else set()
    scheduler.send_result = lambda reply: calls.append("reply")
    original_cache = manager.cache_req

    def cache(request, **kwargs):
        calls.append("donate" if kwargs.get("checkpoint") else "ordinary")
        assert "event" in calls
        return original_cache(request, **kwargs)

    monkeypatch.setattr(manager, "cache_req", cache)
    event = SimpleNamespace(synchronize=lambda: calls.append("event"))
    inputs = SimpleNamespace(batch=batch)
    data = ForwardData(inputs, (None, torch.zeros(1, dtype=torch.int32), event), (131,), plans)
    scheduler._process_last_data(data)
    assert calls[0] == "event" and not states._pending
    if excluded:
        assert "donate" not in calls and not states._checkpoints
    else:
        assert calls.index("event") < calls.index("donate") < calls.index("ordinary")
        matched = manager.prefix_cache.match_prefix(req.input_ids).cuda_handle
        assert matched.cached_len == 128


@pytest.mark.parametrize(
    "unsupported", [None, "strict", "unified", "overlap", "device", "no_cache"]
)
def test_bound_native_capability_and_single_metadata_attachment(case, monkeypatch, unsupported):
    import minisgl.core as core
    from minisgl.attention.kda import flashinfer
    from test_checkpoint_binding import declared

    case.backend.prepare_model(declared(case))
    core.get_global_ctx().bf16_precision = "default"
    case.cache.arena = None
    if unsupported == "strict":
        core.get_global_ctx().bf16_precision = "strict"
    elif unsupported == "unified":
        case.cache.arena = object()
    elif unsupported == "overlap":
        case.cache.supports_overlap = True
    elif unsupported == "device":
        monkeypatch.setattr(flashinfer, "_checkpoint_device_supported", lambda device: False)
    elif unsupported == "no_cache":
        case.states.max_cached_states = 0
    quantum = case.backend.checkpoint_quantum(case.batch)
    expected = 64 if unsupported is None and case.states.state_dtype == torch.bfloat16 else None
    assert quantum == expected
    assert not torch.cuda.is_initialized()
    if expected == 64:
        case.backend.prepare_metadata(case.batch)
        assert case.batch.attn_metadata.checkpoints.plans == (case.plan,)


def test_generic_nonFI_checkpoint_capability128_without_FI_import(case, monkeypatch):
    import builtins
    from dataclasses import replace

    from minisgl.attention.kda.registry import get_kda_kernel
    from test_checkpoint_binding import declared

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.endswith("flashinfer"):
            pytest.fail("generic model binding imported FI")
        return original(name, *args, **kwargs)

    calls = []

    def capability(layers, config, cache, precision, batch):
        calls.append((layers, config, cache, precision, batch))
        return 128

    prepared = []
    def prepare(batch, metadata, layers, config, device, page_size, *, cache):
        prepared.append(cache)
        return SimpleNamespace(plans=tuple(batch.checkpoint_plan), quantum=128)
    case.backend.kernel = replace(
        get_kda_kernel("torch"),
        prepare_checkpoints=prepare,
        checkpoint_quantum=capability,
    )
    monkeypatch.setattr(builtins, "__import__", guarded)
    case.backend.prepare_model(declared(case))
    assert case.backend.checkpoint_quantum(case.batch) == 128
    assert calls[0][0] is case.backend.checkpoint_layer_configs
    assert calls[0][2] is case.cache and calls[0][4] is case.batch
    case.backend.prepare_checkpoints(case.batch)
    assert prepared == [case.cache] and case.batch.attn_metadata.checkpoints.quantum == 128
