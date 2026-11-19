"""CPU-only integration contracts; native kernels and communication are not run."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.layout import ParallelLayout
from minisgl.engine.engine import Engine, WorkerOwner
from minisgl.moe import create_moe_backend, select_moe_backend
from minisgl.moe.base import TorchExpertExecutor
from minisgl.moe.ep import FlashInferEPBackend
from minisgl.moe.flashinfer import FlashInferMXFP4Experts
from minisgl.moe.fused import FusedMoe
from minisgl.moe.prepare import MoePreparation, _signature
from test_moe_engine_startup import construction as construction


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Native EP CPU tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "init", fail)
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def layout(ep=2, rank=0):
    return ParallelLayout(DistributedInfo(rank, 8), dcp_size=8, ep_size=ep)


def context(mesh=None, **options):
    options.setdefault("device", torch.device("cpu"))
    options.setdefault("layout", mesh or layout())
    options.setdefault("local_rows", (48,))
    options.setdefault("fresh_worker", True)
    options.setdefault("abort", lambda error: (_ for _ in ()).throw(error))
    options.setdefault("tp_cpu_group", object())
    options.setdefault("max_prefill_tokens", 16384)
    options.setdefault("max_decode_tokens", 48)
    return MoePreparation(**options)


def signature(ctx):
    executor = object.__new__(FlashInferMXFP4Experts)
    executor._start, executor._experts = 0, 4
    executor._global_experts, executor._hidden, executor._intermediate = 8, 256, 128
    executor._precision, executor._activation = "mxfp8", "situ"
    executor._beta, executor._linear_beta = 4.0, 25.0
    for name in ("_w13", "_w13_scale", "_w2", "_w2_scale", "_alpha", "_linear"):
        setattr(executor, name, None)
    routing = SimpleNamespace(
        topk=2,
        num_groups=1,
        topk_groups=1,
        scoring_func="sigmoid",
        renormalize=True,
        scale=2.5,
        logits=torch.empty(1, 8),
        bias=None,
    )
    return _signature(executor, routing, ctx)


def test_named_existing_record_and_explicit_backend_selection(monkeypatch):
    spec = signature(context())
    assert isinstance(spec, tuple) and spec[6] == spec.hidden == 256
    assert spec.top_k == 2 and spec.activation_precision == "mxfp8"
    assert isinstance(create_moe_backend("flashinfer-ep"), FlashInferEPBackend)
    from minisgl.server.args import parse_args

    config, _ = parse_args(
        ["--model-path", "unused", "--dtype", "bfloat16", "--moe-backend", "flashinfer-ep"]
    )
    assert config.moe_backend == "flashinfer-ep"
    assert select_moe_backend(None, dtype=torch.bfloat16, activation_precision="mxfp8") == "fused"


def test_EP1_keeps_existing_behavior_and_allocates_nothing(monkeypatch):
    import minisgl.moe.ep as module

    monkeypatch.setattr(module, "get_parallel_layout", lambda: layout(1))
    backend = FlashInferEPBackend()
    assert not backend.requires_preparation
    backend.prepare_transport((), context(layout(1)))
    assert backend._mesh is None and not backend._dispatchers
    events = []
    monkeypatch.setattr(FusedMoe, "forward_experts", lambda *a, **kw: events.append((a, kw)) or 17)
    monkeypatch.setattr(FusedMoe, "reduce_outputs", lambda *a, **kw: events.append((a, kw)) or 19)
    assert backend.forward_experts("executor", "x", "route", out="out", deferred=True) == 17
    assert events[0][1] == {"out": "out", "deferred": True}
    assert backend.reduce_outputs("parallel", "mixture", output_dtype=torch.float32) == 19
    backend.close()


def test_all_carrier_formats_are_persistent_before_tuning_or_budget(monkeypatch):
    import minisgl.distributed.expert as groups
    import minisgl.moe.ep as module
    import minisgl.moe.prepare as prepare

    ctx = context()
    events, prepared = [], []
    mesh = SimpleNamespace(info=ctx.layout.experts, cpu_group=object())

    def agree(value, actual, **kwargs):
        assert actual.cpu_group is ctx.tp_cpu_group
        events.append("CPU_agreement")

    def group(*args):
        assert events == ["CPU_agreement"]
        events.append("mesh")
        return mesh

    def transport(*args, **kwargs):
        assert events[:2] == ["CPU_agreement", "mesh"]
        prepared.append(kwargs)
        return SimpleNamespace(close=lambda: events.append("resource_close"))

    monkeypatch.setattr(prepare, "_agree", agree)
    monkeypatch.setattr(groups.ExpertParallel, "prepare", group)
    monkeypatch.setattr(module.FlashInferTokenDispatcher, "prepare", transport)
    backend = FlashInferEPBackend()
    # Two same-shape layers reuse three resources, never one allocation per layer.
    backend.prepare_transport((signature(ctx), signature(ctx)), ctx)
    assert backend._ready and len(prepared) == 3 and len(backend._dispatchers) == 3
    assert {p["payload_layout"][-1][0] for p in prepared} == {
        torch.int32,
        torch.bfloat16,
        torch.float32,
    }
    for value in prepared:
        assert value["max_prefill_tokens"] == 16384 and value["max_decode_tokens"] == 48
        assert value["output_layout"] == (torch.bfloat16, 256)
        assert value["payload_layout"][:2] == ((torch.float8_e4m3fn, 256), (torch.float8_e4m3fn, 8))
    mesh.close = lambda: events.append("group_close")
    backend.close()
    assert events[-4:] == ["resource_close"] * 3 + ["group_close"]
    assert not backend._dispatchers and backend._mesh is None


@pytest.mark.parametrize(
    "invalid", ["prefill", "decode", "bootstrap", "fresh", "owner", "frontend"]
)
def test_capability_rejection_precedes_resource_mutation(monkeypatch, invalid):
    import minisgl.distributed.expert as groups

    ctx = context()
    if invalid == "prefill":
        ctx = replace(ctx, max_prefill_tokens=None)
    elif invalid == "decode":
        ctx = replace(ctx, max_decode_tokens=0)
    elif invalid == "bootstrap":
        ctx = replace(ctx, tp_cpu_group=None)
    elif invalid == "fresh":
        ctx = replace(ctx, fresh_worker=False)
    elif invalid == "owner":
        ctx = replace(ctx, abort=None)
    plan = (signature(ctx),)
    if invalid == "frontend":
        plan = (plan[0]._replace(activation_precision="bf16"),)
    monkeypatch.setattr(groups.ExpertParallel, "prepare", lambda *a: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="explicit token budgets"):
        FlashInferEPBackend().prepare_transport(plan, ctx)


def test_portable_FP32_executor_is_rejected_without_casting_or_allocation():
    with pytest.raises(NotImplementedError, match="MXFP8 FlashInfer"):
        FlashInferEPBackend().prepare_local(TorchExpertExecutor(()), None, context(), dry_run=True)


class Mesh:
    def __init__(self, ep, rank, events):
        self.info = DistributedInfo(rank, ep)
        self.events = events

    def source_counts(self, total):
        return tuple(
            max(0, (total + self.info.size - 1 - e) // self.info.size)
            for e in range(self.info.size)
        )

    def reduce_source(self, value):
        assert value.dtype == torch.bfloat16
        self.events.append(("ETP_sum", value.shape[0]))
        return value

    def gather_source(self, value, total):
        self.events.append(("EP_gather", total))
        return torch.zeros(total, 256, dtype=value.dtype)


@pytest.mark.parametrize("carrier", [torch.int32, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,ep,rank", [(48, 2, 1), (65, 8, 0), (1, 8, 7)])
def test_full_frontend_then_owner_dispatch_compute_combine_restore(
    monkeypatch, carrier, rows, ep, rank
):
    import minisgl.moe.ep as module

    monkeypatch.setattr(module, "get_parallel_layout", lambda: layout(ep, rank))
    events = []
    backend = FlashInferEPBackend()
    backend._mesh, backend._ready = Mesh(ep, rank, events), True
    ids = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2)
    q = torch.zeros(rows, 256, dtype=torch.float8_e4m3fn)
    scale = torch.arange(rows * 8, dtype=torch.uint8).reshape(rows, 8).view(torch.float8_e4m3fn)
    route = torch.ones(rows, 2, dtype=carrier)
    original = tuple(v.view(torch.uint8).clone() for v in (ids, q, scale, route))
    counts = backend._mesh.source_counts(rows)
    received = SimpleNamespace(
        expert_ids=torch.zeros(ep, max(counts), 2, dtype=torch.int32), payloads=object()
    )

    def front(actual_backend, x, routing):
        assert actual_backend is backend and len(x) == rows
        events.append(("full_frontend", rows))
        return ids, (q, scale, route)

    def dispatch(actual_ids, payloads, *, source_counts):
        events.append(("dispatch", len(actual_ids)))
        assert source_counts == counts
        for actual, source in zip((actual_ids, *payloads), (ids, q, scale, route)):
            torch.testing.assert_close(
                actual.view(torch.uint8),
                source[rank::ep].contiguous().view(torch.uint8),
                rtol=0,
                atol=0,
            )
        assert payloads[-1].dtype == carrier
        return received

    def compute(actual_ids, payloads):
        assert actual_ids is received.expert_ids and payloads is received.payloads
        events.append(("compute", ep * max(counts)))
        return torch.zeros(ep * max(counts), 256, dtype=torch.bfloat16)

    def combine(handle, partials):
        assert handle is received and partials.dtype == torch.bfloat16
        events.append(("combine", counts[rank]))
        return torch.zeros(counts[rank], 256, dtype=partials.dtype)

    backend._dispatchers[8, 2, 256, carrier] = SimpleNamespace(dispatch=dispatch, combine=combine)
    executor = SimpleNamespace(
        _global_experts=8, prepare_dispatch_inputs=front, forward_dispatched=compute
    )
    result = backend.forward_experts(executor, torch.zeros(rows, 256), object())
    assert result.shape == (rows, 256) and result.dtype == torch.bfloat16
    assert [e[0] for e in events] == [
        "full_frontend",
        "dispatch",
        "compute",
        "combine",
        "ETP_sum",
        "EP_gather",
    ]
    assert events[0][1] == rows and events[1][1] == counts[rank]
    for source, snapshot in zip((ids, q, scale, route), original):
        torch.testing.assert_close(source.view(torch.uint8), snapshot, rtol=0, atol=0)


def test_shared_has_one_TP_sum_and_routed_has_no_second_AR(monkeypatch):
    import minisgl.moe.ep as module

    monkeypatch.setattr(module, "get_parallel_layout", lambda: layout())
    backend = FlashInferEPBackend()
    backend._ready = True
    events = []
    mixed = torch.ones(3, 2, dtype=torch.bfloat16)
    shared = torch.full((3, 2), 2, dtype=torch.bfloat16)

    def all_reduce(value):
        assert value is shared
        events.append("shared_TP_sum")
        return value * 8

    def norm(value):
        assert value.dtype == torch.bfloat16
        events.append("norm")
        return value + 1

    parallel = SimpleNamespace(all_reduce=all_reduce)
    mixture, tail = backend.forward_outputs(
        parallel,
        lambda out: mixed,
        lambda out: shared,
        norm=SimpleNamespace(forward=norm),
        output_dtype=torch.bfloat16,
    )
    projection = SimpleNamespace(forward=lambda value: events.append("up") or value * 2)
    result = backend.project_outputs(
        parallel,
        mixture,
        tail,
        projection,
        output_dtype=torch.bfloat16,
        residual=torch.ones_like(mixed),
    )
    assert events == ["shared_TP_sum", "norm", "up"]
    torch.testing.assert_close(result, torch.full_like(result, 21), rtol=0, atol=0)


def test_unknown_portfolio_fails_before_dispatch(monkeypatch):
    import minisgl.moe.ep as module

    monkeypatch.setattr(module, "get_parallel_layout", lambda: layout())
    backend = FlashInferEPBackend()
    backend._ready, backend._mesh = True, Mesh(2, 0, [])
    executor = SimpleNamespace(
        _global_experts=8,
        prepare_dispatch_inputs=lambda *a: (
            torch.zeros(1, 2, dtype=torch.int32),
            (torch.zeros(1, 256), None, torch.zeros(1, 2, dtype=torch.float16)),
        ),
    )
    with pytest.raises(NotImplementedError, match="prepared EP portfolio"):
        backend.forward_experts(executor, torch.zeros(1, 256), object())
    with pytest.raises(NotImplementedError, match="prepared expert-executor"):
        backend.forward(None, None, None, None, 1, False)


def test_actual_Engine_passes_budgets_before_KV_and_closes_before_groups(construction, monkeypatch):
    f = construction
    f.backend.requires_preparation = True
    f.backend.close = lambda: f.events.append("backend_close")
    cfg = f.config(ep=2, rows=48)
    cfg.max_forward_len = 16384
    value = Engine(cfg, worker_owner=WorkerOwner(True, lambda error: (_ for _ in ()).throw(error)))
    prepared = f.contexts[0]
    assert prepared.tp_cpu_group is f.bootstrap
    assert prepared.max_prefill_tokens == 16384 and prepared.max_decode_tokens == 48
    assert f.events.index("prepare") < f.events.index("budget") < f.events.index("KV")
    value.graph_runner.destroy_cuda_graphs = lambda: f.events.append("graphs_close")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: f.events.append("drain"))
    monkeypatch.setattr(
        f.engine, "destroy_distributed", lambda: f.events.append("distributed_close")
    )
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda: f.events.append("bootstrap_close")
    )
    value.shutdown()
    assert (
        f.events.index("graphs_close")
        < f.events.index("backend_close")
        < f.events.index("distributed_close")
        < f.events.index("bootstrap_close")
    )


@pytest.mark.parametrize("bad", ["owner", "fresh"])
def test_actual_Engine_native_requirements_fail_before_weights(construction, bad):
    f = construction
    f.backend.requires_preparation = True
    cfg = f.config(ep=2)
    owner = None if bad == "owner" else WorkerOwner(bad != "fresh", lambda error: None)
    with pytest.raises(ValueError, match="fresh owned worker"):
        Engine(cfg, worker_owner=owner)
    assert "load_weights" not in f.events and "budget" not in f.events


def test_real_ratio_capacity_accounts_for_retained_native_bytes(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    engine = object.__new__(Engine)
    engine.dtype, engine.tp_cpu_group = torch.bfloat16, None
    engine._initial_free_memory = 10000
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                cache_bytes_per_page=lambda *a: 200, state_bytes_per_request=lambda *a, **kw: 100
            )
        ),
        num_page_override=None,
        cache_dtype=torch.bfloat16,
        logical_page_size=64,
        tp_info=DistributedInfo(0, 1),
        recurrent_state_dtype=torch.bfloat16,
        memory_ratio=0.9,
        enable_unified_memory=False,
        state_memory_ratio=0.31,
        max_running_req=4,
    )
    capacities = []
    for retained in (0, 2000):
        engine._local_free_memory = 8000 - retained
        engine._sync_get_memory = lambda: (engine._local_free_memory, engine._local_free_memory)
        pages = engine._determine_num_pages(10000, config)
        capacities.append((pages, engine.state_slot_capacity))
    assert capacities[1][0] < capacities[0][0] and capacities[1][1] < capacities[0][1]


@pytest.mark.parametrize("returns", [False, True])
def test_native_failure_aborts_once_without_retry_or_group_cleanup(monkeypatch, returns):
    import minisgl.moe.ep as module

    monkeypatch.setattr(module, "get_parallel_layout", lambda: layout())
    backend = FlashInferEPBackend()
    backend._ready, backend._mesh = True, Mesh(2, 0, [])
    failure = RuntimeError("native dispatch fault")
    events = []

    class Abort(BaseException):
        pass

    def abort(error):
        assert error is failure
        events.append("abort")
        if not returns:
            raise Abort

    def dispatch(*args, **kwargs):
        events.append("dispatch")
        raise failure

    backend._abort = abort
    backend._dispatchers[8, 2, 256, torch.int32] = SimpleNamespace(
        dispatch=dispatch, combine=lambda *a: pytest.fail("retry/combine")
    )
    executor = SimpleNamespace(
        _global_experts=8,
        prepare_dispatch_inputs=lambda *a: (
            torch.zeros(2, 2, dtype=torch.int32),
            (torch.zeros(2, 256), torch.zeros(2, 8), torch.zeros(2, 2, dtype=torch.int32)),
        ),
    )
    with pytest.raises(RuntimeError if returns else Abort) as raised:
        backend.forward_experts(executor, torch.zeros(2, 256), object())
    assert events == ["dispatch", "abort"]
    if returns:
        assert "abort callable returned" in str(raised.value) and raised.value.__cause__ is failure


def test_existing_owned_prepare_invokes_transport_after_plan_before_tuning(monkeypatch):
    from test_moe_prepare_minimal import context as old_context
    from test_moe_prepare_minimal import model, setup

    executor, calls, state = setup(monkeypatch, ParallelLayout())
    backend = FusedMoe()
    seen = []

    def transport(plan, context):
        assert len(plan) == 1 and isinstance(plan[0], tuple) and plan[0].hidden == 256
        assert not calls and not state.entries  # No expert launch/tuning scope before this hook.
        seen.append(plan)

    monkeypatch.setattr(backend, "prepare_transport", transport)
    backend.prepare(model(executor), old_context())
    assert len(seen) == 1 and len(calls) == 1 and state.entries


def test_failed_resource_close_does_not_destroy_groups_or_continue(monkeypatch):
    backend = FlashInferEPBackend()
    failure = RuntimeError("failed channel cannot close collectively")
    events = []

    class Abort(BaseException):
        pass

    def abort(error):
        assert error is failure
        events.append("abort")
        raise Abort

    def fail():
        events.append("close_failed")
        raise failure

    backend._abort = abort
    backend._dispatchers = {
        "first": SimpleNamespace(close=fail),
        "next": SimpleNamespace(close=lambda: pytest.fail("continued cleanup")),
    }
    backend._mesh = SimpleNamespace(close=lambda: pytest.fail("destroyed borrowed bootstrap"))
    with pytest.raises(Abort):
        backend.close()
    assert events == ["close_failed", "abort"]
