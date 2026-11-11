"""Protocol only: real public traversal/forward preparation, fake FI GPU boundary."""

import gc
import json
import sys
import weakref
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.distributed.context import create_expert_cpu_group
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.layout import ParallelLayout
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.latent_moe import GroupedRouter, LatentMoE
from minisgl.moe import prepare as runtime
from minisgl.moe.fused import FusedMoe
from test_flashinfer_route_dtype import executor as cpu_boundary


def setup(monkeypatch, layout):
    op, calls = cpu_boundary.__wrapped__(monkeypatch)
    owned = layout.expert_range(8)
    op._start, op._experts = owned.start, len(owned)
    op._beta, op._linear_beta = 1.0, None
    op._validate_input = lambda value: None  # CPU storage; production CUDA guard stays intact.
    state = SimpleNamespace(group=None, is_tuning_mode=False, clears=0, entries=[])

    def clear():
        state.clears += 1

    state.clear_cache = clear

    @contextmanager
    def autotune(mode, **options):
        assert options == {"cache": None, "skip_ops": set()}
        state.entries.append(mode)
        state.is_tuning_mode = mode
        try:
            yield
        finally:
            state.is_tuning_mode = False

    api = ModuleType("flashinfer.autotuner")
    api.AutoTuner = SimpleNamespace(get=lambda: state)
    api.autotune = autotune
    api.get_autotune_process_group = lambda: state.group
    api.set_autotune_process_group = lambda group: setattr(state, "group", group)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", api)
    monkeypatch.setattr(sys.modules["flashinfer"], "__version__", "cpu-fake", raising=False)
    monkeypatch.setattr(runtime, "_domain", lambda *args: ("same-cpu-fi-domain",))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return op, calls, state


def model(op, count=1, events=None):
    root = BaseOP()
    blocks = []
    for index in range(count):
        layer = LatentMoE.__new__(LatentMoE)
        layer._backend, layer._expert_executor = FusedMoe(), op
        layer.gate = GroupedRouter.__new__(GroupedRouter)
        layer.gate.e_score_correction_bias = None
        layer.gate._config = SimpleNamespace(
            num_experts=8,
            num_experts_per_token=2,
            num_expert_group=1,
            topk_group=1,
            moe_router_activation_func="softmax",
            moe_renormalize=True,
            routed_scaling_factor=1.0,
        )
        if events is not None:
            original = layer.prepare_runtime

            def record(context, *, dry_run=False, previous=original, index=index):
                events.append((index, dry_run))
                return previous(context, dry_run=dry_run)

            layer.prepare_runtime = record
        blocks.append(layer)
    root.blocks = OPList(blocks)
    return root


def context(layout=None, **options):
    def abort(error):
        raise error

    options.setdefault("abort", abort)
    options.setdefault("local_rows", (48,))  # Caller-selected serving representative.
    return runtime.MoePreparation(
        torch.device("cpu"), layout or ParallelLayout(), fresh_worker=True, **options
    )


@pytest.mark.parametrize("mode", ["bf16", "mxfp8", "packed"])
def test_real_forward_and_startup_share_identical_prepared_arguments(monkeypatch, mode):
    op, _, _ = setup(monkeypatch, ParallelLayout())
    backend, root = FusedMoe(), None
    root = model(op)
    x = torch.randn(3, 256).bfloat16()
    routing = root.blocks.get_op("0").gate.routing(torch.randn(3, 8))
    if mode != "bf16":
        op._precision = "mxfp8"
        x = x.float()
        from minisgl.moe import sgl

        q, scale = x.to(torch.float8_e4m3fn), torch.ones(3, 8, dtype=torch.uint8)
        packed = torch.zeros(3, 2, dtype=torch.int32)
        monkeypatch.setattr(
            sgl,
            "fused_route_quant",
            lambda *args: (None, None, packed, q, scale) if mode == "packed" else None,
        )
        module = ModuleType("sglang.kernels.ops.quantization.per_token_group_quant")
        module.per_token_group_quant = lambda *args, **kw: (q, scale)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    prepared = op.prepare_inputs(backend, x, routing)
    captured = []
    monkeypatch.setattr(op, "_run", lambda *args: captured.append(args) or "same-kernel")
    assert backend.forward_experts(op, x, routing) == "same-kernel"
    for actual, expected in zip(captured[0], prepared):
        if isinstance(actual, tuple):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(actual, torch.Tensor):
            assert actual.dtype == expected.dtype and actual.stride() == expected.stride()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected
    assert not torch.cuda.is_initialized()


def test_two_public_traversals_release_all_inputs_before_cleanup(monkeypatch):
    op, calls, state = setup(monkeypatch, ParallelLayout())
    events, references, cleanup = [], [], []
    original = op.prepare_inputs

    def capture(*args):
        references.extend((weakref.ref(args[1]), weakref.ref(args[2].logits)))
        return original(*args)

    monkeypatch.setattr(op, "prepare_inputs", capture)
    collect = gc.collect

    def inspect():
        calls.clear()  # Remove fake-kernel observer-owned tensor references.
        cleanup.append([value() is not None for value in references])
        assert not any(cleanup[-1])
        return collect()

    monkeypatch.setattr(runtime.gc, "collect", inspect)
    FusedMoe().prepare(model(op, count=2, events=events), context(local_rows=(1, 8, 48)))
    assert events == [(0, True), (1, True), (0, False), (1, False)]
    assert state.entries == [True] and state.clears == 1 and state.group is None
    assert len(references) == 12 and cleanup == [[False] * 12]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "fault", ["fresh", "precision", "owner", "bundled", "rows", "placement", "supervisor"]
)
def test_refuses_unqualified_contracts_before_any_profile(monkeypatch, fault):
    op, calls, state = setup(monkeypatch, ParallelLayout())
    ctx = context()
    if fault == "fresh":
        ctx = replace(ctx, fresh_worker=False)
    elif fault == "precision":
        ctx = replace(ctx, precision="strict")
    elif fault == "owner":
        state.group = object()
    elif fault == "bundled":
        monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    elif fault == "rows":
        ctx = replace(ctx, local_rows=(0,))
    elif fault == "supervisor":
        ctx = replace(ctx, abort=None)
    else:
        op._start = 2
    previous = state.group
    with pytest.raises(ValueError):
        FusedMoe().prepare(model(op), ctx)
    assert not calls and not state.entries and state.group is previous


def test_failure_restores_group_and_requires_supervisor_not_recovery(monkeypatch):
    op, _, state = setup(monkeypatch, ParallelLayout())

    class Supervised(Exception):
        pass

    def abort(error):
        assert isinstance(error, RuntimeError)
        assert not state.is_tuning_mode and state.group is None
        raise Supervised("owned worker set terminates")

    monkeypatch.setattr(
        op, "_run", lambda *args: (_ for _ in ()).throw(RuntimeError("native failure"))
    )
    with pytest.raises(Supervised):
        FusedMoe().prepare(model(op), context(abort=abort))


def _aligned(value):
    if isinstance(value, tuple):
        return tuple(_aligned(item) for item in value)
    if not isinstance(value, torch.Tensor):
        return value
    backing = torch.empty(value.numel() + 256, dtype=value.dtype)
    start = ((-backing.data_ptr()) % 256) // value.element_size()
    return backing[start : start + value.numel()].view(value.shape).copy_(value)


def _gloo_worker(rank, directory):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    output = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{output / 'rendezvous'}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=25),
    )
    try:
        for ep in (1, 2, 4):
            patch = pytest.MonkeyPatch()
            layout = ParallelLayout(DistributedInfo(rank, 4), ep_size=ep)
            resource = create_expert_cpu_group(layout, dist.group.WORLD)

            def abort(error):
                raise RuntimeError("supervisor requested") from error

            try:
                op, calls, state = setup(patch, layout)
                original = op.prepare_inputs
                patch.setattr(op, "prepare_inputs", lambda *args: _aligned(original(*args)))
                ctx = context(layout, cpu_group=resource.group if resource else None, abort=abort)
                FusedMoe().prepare(model(op), ctx)
                assert state.group is None and len(calls) == 1
                faulted = model(op)
                if layout.expert_tensor.rank == 0:

                    def metadata_failure(context, *, dry_run=False):
                        raise RuntimeError("rank-local metadata failure")

                    faulted.prepare_runtime = metadata_failure
                with pytest.raises(ValueError, match="preflight"):
                    FusedMoe().prepare(faulted, replace(ctx, abort=None))
                assert state.group is None and len(calls) == 1 and state.clears == 1
                if ep == 2:
                    # Same static plan, different actual prepared routing-weight ABI.
                    def divergent(*args):
                        x, scale, routes, topk = _aligned(original(*args))
                        ids, weights = routes
                        return (
                            x,
                            scale,
                            (ids, weights.to(torch.bfloat16) if rank < 2 else weights),
                            topk,
                        )

                    patch.setattr(op, "prepare_inputs", divergent)
                    with pytest.raises(RuntimeError, match="supervisor requested"):
                        FusedMoe().prepare(model(op), ctx)
                    assert state.group is None and len(calls) == 1
                if resource:
                    probe = torch.tensor(rank + 1, dtype=torch.int64)
                    resource.all_reduce(probe)
                    assert probe.item() == sum(index + 1 for index in range(rank % ep, 4, ep))
            finally:
                patch.undo()
                if resource:
                    resource.close()
                    resource.close()
            dist.barrier()
        assert not torch.cuda.is_initialized()
        (output / f"rank{rank}.json").write_text(
            json.dumps(
                {"cuda_initialized": False, "ep": [1, 2, 4], "metadata_failure_consensus": True}
            )
        )
    finally:
        dist.destroy_process_group()


def test_real_four_rank_gloo_plan_and_prepared_abi_consensus(tmp_path):
    import time

    import torch.multiprocessing as mp

    workers = mp.spawn(_gloo_worker, args=(str(tmp_path),), nprocs=4, join=False)
    deadline = time.monotonic() + 50
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Owned CPU workers exceeded 50 seconds")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    assert all(
        json.loads((tmp_path / f"rank{rank}.json").read_text())["cuda_initialized"] is False
        for rank in range(4)
    )


def test_unsupported_modules_are_noop_without_fi_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "flashinfer.autotuner", raising=False)
    FusedMoe().prepare(BaseOP(), replace(context(), fresh_worker=False, abort=None))
    assert "flashinfer.autotuner" not in sys.modules and not torch.cuda.is_initialized()


def test_tactic_domain_rejects_different_fi_environment(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            major=10,
            minor=3,
            multi_processor_count=148,
        ),
    )
    before = runtime._domain(context(), "same-fi-version")
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_TEST_DOMAIN", "different")
    assert runtime._domain(context(), "same-fi-version") != before
    assert not torch.cuda.is_initialized()


def test_local_rows_are_a_required_caller_choice():
    with pytest.raises(TypeError, match="local_rows"):
        runtime.MoePreparation(torch.device("cpu"), ParallelLayout())


def test_unqualified_dry_run_only_allocates_meta(monkeypatch):
    op, calls, state = setup(monkeypatch, ParallelLayout())
    devices, empty = [], torch.empty

    def metadata_only(*args, **kwargs):
        devices.append(kwargs.get("device"))
        assert kwargs.get("device") == "meta"
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", metadata_only)
    with pytest.raises(ValueError, match="fresh exclusive"):
        FusedMoe().prepare(model(op), replace(context(), fresh_worker=False))
    assert devices == ["meta"] and not calls and not state.entries and state.clears == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("rows", [(False,), (1.5,), (0,), ()])
def test_local_rows_reject_nonpositive_or_noninteger_bounds(monkeypatch, rows):
    op, calls, state = setup(monkeypatch, ParallelLayout())
    with pytest.raises(ValueError, match="positive local row"):
        FusedMoe().prepare(model(op), context(local_rows=rows))
    assert not calls and state.clears == 0 and not state.entries


def test_weight_device_is_checked_in_metadata_preflight(monkeypatch):
    op, calls, state = setup(monkeypatch, ParallelLayout())
    op._w13 = torch.empty(2, 2, device="meta")
    with pytest.raises(ValueError, match="device differs"):
        FusedMoe().prepare(model(op), context())
    assert not calls and state.clears == 0 and not state.entries


def test_prepared_device_is_checked_before_native_run(monkeypatch):
    op, calls, state = setup(monkeypatch, ParallelLayout())
    original = op.prepare_inputs

    def wrong_device(*args):
        x, scale, routes, topk = original(*args)
        return x.to("meta"), scale, routes, topk

    monkeypatch.setattr(op, "prepare_inputs", wrong_device)
    with pytest.raises(ValueError, match="device differs"):
        FusedMoe().prepare(model(op), context())
    assert not calls and state.group is None and state.entries == [True]
    assert not torch.cuda.is_initialized()


def test_routing_bias_abi_is_part_of_metadata_plan(monkeypatch):
    op, calls, state = setup(monkeypatch, ParallelLayout())
    layer = model(op).blocks.get_op("0")
    layer.gate.e_score_correction_bias = torch.zeros(8, dtype=torch.float32)
    before = layer.prepare_runtime(context(), dry_run=True)
    layer.gate.e_score_correction_bias = torch.zeros(8, dtype=torch.bfloat16)
    after = layer.prepare_runtime(context(), dry_run=True)
    assert before != after and not calls and state.clears == 0 and not state.entries
    assert not torch.cuda.is_initialized()


def test_multiple_workload_rows_reach_existing_packed_and_unpacked_abi(monkeypatch):
    from minisgl.moe import sgl

    op, calls, state = setup(monkeypatch, ParallelLayout())
    op._precision = "mxfp8"

    def packed(x, routing):
        if len(x) > 64:
            return None
        return (
            None,
            None,
            torch.zeros((len(x), routing.topk), dtype=torch.int32),
            x.to(torch.float8_e4m3fn),
            torch.zeros((len(x), x.shape[1] // 32), dtype=torch.uint8),
        )

    monkeypatch.setattr(sgl, "fused_route_quant", packed)
    module = ModuleType("sglang.kernels.ops.quantization.per_token_group_quant")
    module.per_token_group_quant = lambda x, **kw: (
        x.to(torch.float8_e4m3fn),
        torch.zeros((len(x), x.shape[1] // 32), dtype=torch.uint8),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    FusedMoe().prepare(model(op), context(local_rows=(48, 128)))
    assert state.entries == [True] and state.clears == 1
    assert [value["hidden_states"].shape[0] for value in calls] == [48, 128]
    assert isinstance(calls[0]["topk_ids"], torch.Tensor)
    assert isinstance(calls[1]["topk_ids"], tuple)
    assert calls[0]["topk_ids"].dtype == calls[1]["topk_ids"][0].dtype == torch.int32
    assert [value["tune_max_num_tokens"] for value in calls] == [64, 128]
    assert all(value["do_finalize"] is True for value in calls)
    assert not torch.cuda.is_initialized()
