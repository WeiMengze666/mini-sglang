"""CPU lifecycle tests: real Engine/Scheduler constructors, public485 hooks and parent supervision.

CUDA construction interfaces are substituted with CPU storage; no kernel, model
inference, or GPU initialization is performed by these tests.
"""

import multiprocessing as mp
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.layout import ParallelLayout
from minisgl.engine.engine import Engine, StartupOwner
from minisgl.layers.base import BaseOP
from minisgl.moe.fused import FusedMoe
from minisgl.server.launch import _abort_startup, _run_scheduler, _stop_workers, _wait_for_workers


class Stream:
    def wait_stream(self, _other):
        pass


@pytest.fixture
def construction(monkeypatch):
    import minisgl.distributed.context as groups
    import minisgl.distributed.info as info
    import minisgl.engine.engine as engine
    import torch.distributed as dist

    events = []
    bootstrap = SimpleNamespace(ranks=list(range(4)), backend="gloo", name="bootstrap")
    contexts = []
    calls = []
    real_zeros = torch.zeros

    class Leaf(BaseOP):
        ready = False

        def after_load(self):
            self.ready = True
            events.append("after_load")

    class Model(BaseOP):
        def __init__(self):
            self.leaf = Leaf()

        def load_weights(self, _weights, *, device):
            events.append("load_weights")
            self.apply(lambda op: op.after_load())

        def load_state_dict(self, _weights):
            events.append("load_state_dict")
            self.apply(lambda op: op.after_load())

    class Backend:
        fail = False

        def validate_precision(self, precision):
            assert precision == "mxfp8"

        def prepare(self, model, context):
            assert model.leaf.ready and "budget" not in events and "graphs" not in events
            contexts.append(context)
            events.append("prepare")
            if self.fail:
                raise RuntimeError("device failure: worker owner terminates, no cleanup barrier")

    backend = Backend()

    def device_zeros(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            kwargs["device"] = "cpu"
            events.append("base_buffer")
        return real_zeros(*args, **kwargs)

    def memory(self):
        self._local_free_memory = 1 << 38
        return 1 << 38, 1 << 38

    def pages(self, *args):
        events.append("budget")
        self.cached_state_capacity = 0
        return 4

    def new_group(ranks, **kwargs):
        assert kwargs == {"backend": "gloo", "use_local_synchronization": True}
        events.append(("new_group", tuple(ranks)))
        return (
            SimpleNamespace(ranks=ranks, backend="gloo", name="etp")
            if 0 in ranks
            else dist.GroupMember.NON_GROUP_MEMBER
        )

    def destroy(group):
        calls.append(group)
        events.append("close_owned_group")

    monkeypatch.setattr(engine, "_adjust_config", lambda config: None)
    monkeypatch.setattr(engine, "set_tp_info", lambda **kwargs: None)
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 4))
    monkeypatch.setattr(engine, "configure_parallel", lambda **kwargs: None)
    monkeypatch.setattr(engine, "init_context_parallel", lambda **kwargs: None)
    monkeypatch.setattr(engine, "set_global_ctx", lambda ctx: None)
    monkeypatch.setattr(engine, "set_rope_device", lambda device: None)
    monkeypatch.setattr(engine, "create_model", lambda config: Model())
    monkeypatch.setattr(engine, "create_moe_backend", lambda name: backend)
    monkeypatch.setattr(
        engine, "create_kvcache_pool", lambda **kwargs: events.append("KV") or SimpleNamespace()
    )
    monkeypatch.setattr(engine, "create_attention_backend", lambda *args: SimpleNamespace())
    monkeypatch.setattr(engine, "Sampler", lambda *args: SimpleNamespace())
    monkeypatch.setattr(
        engine, "GraphRunner", lambda **kwargs: events.append("graphs") or SimpleNamespace()
    )
    monkeypatch.setattr(engine, "native_precision_scope", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(Engine, "_init_communication", lambda self, config: bootstrap)
    monkeypatch.setattr(Engine, "_sync_get_memory", memory)
    monkeypatch.setattr(Engine, "_check_capacity", lambda *args: events.append("precheck"))
    monkeypatch.setattr(Engine, "_iter_weights", lambda *args: iter(()))
    monkeypatch.setattr(Engine, "_load_weight_state_dict", lambda *args: {})
    monkeypatch.setattr(Engine, "_determine_num_pages", pages)
    monkeypatch.setattr(torch, "zeros", device_zeros)
    monkeypatch.setattr(torch, "manual_seed", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "set_stream", lambda *args: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda group: group.backend)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dist, "get_rank", lambda *args: 0)
    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)

    def config(*, ep=1, rows=37, precision="auto", hybrid=True, is_moe=True):
        value = SimpleNamespace(
            tp_info=DistributedInfo(0, 4),
            dcp_size=1,
            ep_size=ep,
            attention_provider=None,
            dtype=torch.bfloat16,
            logical_page_size=64,
            bf16_precision=precision,
            model_config=SimpleNamespace(
                is_moe=is_moe, cache_config=object() if hybrid else None, vocab_size=8
            ),
            moe_backend="fused",
            mxfp4_backend="flashinfer",
            mxfp4_activation_precision="mxfp8",
            max_running_req=rows,
            cache_dtype=torch.bfloat16,
            max_seq_len=8,
            cuda_graph_bs=[],
            cuda_graph_max_bs=0,
            enable_unified_memory=False,
            enable_cache_overlap=False,
            recurrent_state_dtype=torch.bfloat16,
            attention_backend="auto",
        )
        monkeypatch.setattr(
            engine, "get_parallel_layout", lambda: ParallelLayout(value.tp_info, ep_size=ep)
        )
        return value

    return SimpleNamespace(
        events=events,
        contexts=contexts,
        backend=backend,
        config=config,
        bootstrap=bootstrap,
        destroyed=calls,
        groups=groups,
        engine=engine,
    )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("ep", [1, 2, 4])
@pytest.mark.parametrize("rows", [11, 37])
def test_preparation_follows_weights_and_releases_only_owned_ETP_before_budget(
    construction, hybrid, ep, rows
):
    f = construction
    owner = StartupOwner(True, _abort_startup)
    value = Engine(f.config(ep=ep, rows=rows, hybrid=hybrid), startup_owner=owner)
    assert len(f.contexts) == 1
    context = f.contexts[0]
    assert context.device == value.device and context.layout.ep_size == ep
    assert context.local_rows == (rows,) and context.precision == "default"
    assert context.fresh_worker and context.abort is owner.abort
    assert (
        f.events.index("after_load")
        < f.events.index("prepare")
        < f.events.index("budget")
        < f.events.index("KV")
        < f.events.index("graphs")
    )
    if ep == 1:
        assert context.cpu_group is f.bootstrap and not f.destroyed
    elif ep == 2:
        assert context.cpu_group.ranks == [0, 2] and f.destroyed == [context.cpu_group]
        assert (
            f.events.index("prepare")
            < f.events.index("close_owned_group")
            < f.events.index("budget")
        )
    else:
        assert context.cpu_group is None and not f.destroyed
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mode", ["no_owner", "strict", "non_moe"])
def test_optional_unsupported_paths_keep_original_initialization(construction, monkeypatch, mode):
    f = construction
    monkeypatch.setattr(
        f.groups,
        "create_expert_cpu_group",
        lambda *args: pytest.fail("optional path created prep group"),
    )
    config = f.config(precision="strict" if mode == "strict" else "auto", is_moe=mode != "non_moe")
    owner = None if mode == "no_owner" else StartupOwner(True, _abort_startup)
    value = Engine(config, startup_owner=owner)
    assert not f.contexts and "prepare" not in f.events and "graphs" in f.events
    assert value.ctx.bf16_precision == ("strict" if mode == "strict" else "default")


def test_nonFI_empty_public_plan_does_not_profile_or_allocate_extra_GPU(construction, monkeypatch):
    import builtins

    f = construction
    monkeypatch.setattr(f.engine, "create_moe_backend", lambda name: FusedMoe())
    # ETP singleton avoids even CPU timing collectives; all leaf hooks returnNone.
    config = f.config(ep=4)
    original = builtins.__import__

    def forbid(name, *args, **kwargs):
        if name == "flashinfer" or name.startswith("flashinfer."):
            pytest.fail("empty public plan imported FlashInfer")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid)
    Engine(config, startup_owner=StartupOwner(True, _abort_startup))
    assert not f.contexts and not f.destroyed and "graphs" in f.events
    assert f.events.count("base_buffer") == 1  # Existing page table only, no preparation buffer.
    assert not torch.cuda.is_initialized()


class OwnedAbort(BaseException):
    pass


@pytest.mark.parametrize("stage", ["create", "prepare", "close"])
def test_preparation_failure_aborts_once_without_unknown_cleanup(construction, monkeypatch, stage):
    f = construction
    error = RuntimeError(f"{stage} failure")
    calls = []

    def abort(cause):
        calls.append(cause)
        raise OwnedAbort

    if stage == "prepare":

        def fail(*args):
            raise error

        monkeypatch.setattr(f.backend, "prepare", fail)
    else:
        original = f.groups.create_expert_cpu_group

        def group(*args):
            if stage == "create":
                raise error
            resource = original(*args)

            def close():
                f.events.append("close_failure")
                raise error

            resource.close = close
            return resource

        monkeypatch.setattr(f.groups, "create_expert_cpu_group", group)
    with pytest.raises(OwnedAbort):
        Engine(f.config(ep=2), startup_owner=StartupOwner(True, abort))
    assert calls == [error]
    assert "budget" not in f.events and "graphs" not in f.events
    assert not f.destroyed
    assert f.events.count("close_failure") == int(stage == "close")


def test_returning_abort_callback_is_rejected(construction, monkeypatch):
    f = construction
    error = RuntimeError("preparation failure")
    calls = []

    def fail(*args):
        raise error

    monkeypatch.setattr(f.backend, "prepare", fail)
    with pytest.raises(RuntimeError, match="Startup abort callback returned") as raised:
        Engine(f.config(ep=2), startup_owner=StartupOwner(True, calls.append))
    assert raised.value.__cause__ is error and calls == [error]
    assert not f.destroyed and "budget" not in f.events and "graphs" not in f.events


@pytest.mark.parametrize("has_owner", [False, True])
def test_scheduler_passes_borrowed_owner_and_existing_CPU_group(monkeypatch, has_owner):
    import minisgl.engine as engine
    import minisgl.scheduler.scheduler as scheduler

    seen = []
    group = object()
    stream = Stream()
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        stream=stream,
        page_table=torch.zeros(2, 32, dtype=torch.int32),
        num_pages=4,
        kv_cache=object(),
        tp_cpu_group=group,
    )
    monkeypatch.setattr(
        engine, "Engine", lambda config, *, startup_owner=None: seen.append(startup_owner) or fake
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setattr(
        scheduler, "TableManager", lambda *args: SimpleNamespace(token_pool=object())
    )
    monkeypatch.setattr(scheduler, "CacheManager", lambda *args, **kwargs: object())
    monkeypatch.setattr(scheduler, "DecodeManager", lambda *args: object())
    monkeypatch.setattr(scheduler, "PrefillManager", lambda *args: object())
    monkeypatch.setattr(
        scheduler, "load_tokenizer", lambda *args, **kwargs: SimpleNamespace(eos_token_id=0)
    )
    monkeypatch.setattr(scheduler, "get_stop_token_ids", lambda token: {0})
    config = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=8),
        max_running_req=1,
        logical_page_size=64,
        cache_type="naive",
        model_path="cpu-fixture",
        trust_remote_code=False,
        max_extend_tokens=17,
        tp_info=DistributedInfo(0, 1),
        offline_mode=True,
    )
    owner = StartupOwner(True, _abort_startup) if has_owner else None
    result = scheduler.Scheduler(config, startup_owner=owner)
    assert seen == [owner] and result.tp_cpu_group is group


def test_startup_owner_is_small_and_pickleable():
    owner = StartupOwner(True, _abort_startup)
    restored = pickle.loads(pickle.dumps(owner))
    assert restored.fresh_worker and restored.abort is _abort_startup


def _metadata_failure_scheduler(args, *, startup_owner):
    from minisgl.moe.prepare import MoePreparation

    assert startup_owner.fresh_worker and startup_owner.abort is _abort_startup
    root = BaseOP()

    def bad(context, *, dry_run=False):
        assert dry_run
        Path(args.evidence).write_text(
            __import__("json").dumps(
                {
                    "fresh_worker": context.fresh_worker,
                    "rows": context.local_rows,
                    "dry_run": dry_run,
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "native_FI_imported": any(
                        x == "flashinfer" or x.startswith("flashinfer.") for x in sys.modules
                    ),
                }
            )
        )
        raise ValueError("rank-local metadata preflight failure")

    root.prepare_runtime = bad
    FusedMoe().prepare(
        root,
        MoePreparation(
            torch.device("cpu"),
            ParallelLayout(),
            (args.rows,),
            fresh_worker=startup_owner.fresh_worker,
            abort=startup_owner.abort,
        ),
    )
    raise AssertionError("fatal owner returned")


def _fresh_spawn_metadata_failure(queue, evidence):
    from types import ModuleType

    fake = ModuleType("minisgl.scheduler")
    fake.Scheduler = _metadata_failure_scheduler
    sys.modules["minisgl.scheduler"] = fake
    args = SimpleNamespace(
        evidence=evidence, rows=37, tp_info=DistributedInfo(0, 1), silent_output=False
    )
    _run_scheduler(args, queue)


def _wait_peer(event):
    event.wait()


def test_real_fresh_spawn_fatal_preflight_is_seen_by_existing_parent_and_peers_end(tmp_path):
    context = mp.get_context("spawn")
    queue, event = context.Queue(), context.Event()
    evidence = tmp_path / "metadata.json"
    failed = context.Process(
        target=_fresh_spawn_metadata_failure,
        args=(queue, str(evidence)),
        name="owned-preparation-failure",
    )
    peer = context.Process(target=_wait_peer, args=(event,), name="owned-waiting-peer")
    unrelated = context.Process(target=_wait_peer, args=(event,), name="unrelated-peer")
    for proc in (failed, peer, unrelated):
        proc.start()
    try:
        with pytest.raises(RuntimeError, match="owned-preparation-failure.*1"):
            list(_wait_for_workers(queue, [failed, peer], 1))
        _stop_workers([failed, peer])
        assert failed.exitcode == 1 and not peer.is_alive() and unrelated.is_alive()
        value = __import__("json").loads(evidence.read_text())
        assert value == {
            "fresh_worker": True,
            "rows": [37],
            "dry_run": True,
            "cuda_initialized": False,
            "native_FI_imported": False,
        }
    finally:
        _stop_workers([failed, peer, unrelated])
        queue.close()
