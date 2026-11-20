"""CPU ABI, real Gloo dimensions, and native-gather lifecycle checks; no GPU execution."""

import multiprocessing
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import test_moe_engine_startup as startup_tests
import torch
from minisgl.distributed.impl import (
    DistributedCommunicator,
    FallbackDistributedImpl,
    TorchDistributedImpl,
)
from minisgl.distributed.mnnvl import MNNVLDistributedImpl
from minisgl.distributed.multimem.provider import MultimemProvider
from minisgl.distributed.parallel import TensorParallel
from minisgl.engine.engine import Engine, _prepare_head_all_gather
from minisgl.layers.linear import Linear
from test_collective_outputs import cuda_tensor

construction = startup_tests.construction


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU test attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.fixture
def native(monkeypatch):
    events = []
    cpu_group, device_group = object(), object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 1)

    def agreement(peers, packet, *, group):
        assert group is cpu_group
        peers[:] = [packet, packet]
        events.append(("agreement", packet))

    monkeypatch.setattr(torch.distributed, "all_gather_object", agreement)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("drain", device)))
    comm = SimpleNamespace(disabled=False, has_multicast=True, close=lambda: events.append("close"))
    provider = MultimemProvider(
        comm,
        group=device_group,
        output_group=cpu_group,
        device="cuda:0",
        execution_context=nullcontext,
    )

    def create(group, rank, rows, width, *, device):
        assert group is device_group and rank == 1 and device == torch.device("cuda:0")
        events.append(("create", rows, width))
        return SimpleNamespace(
            hidden_dim=width,
            comm_buff=torch.empty(rows, width, dtype=torch.bfloat16),
            symm_mem_hdl=SimpleNamespace(multicast_ptr=128),
        )

    def gather(state, x, width, *, skip_entry_sync, safe):
        assert width == state.hidden_dim and skip_entry_sync is False and safe is True
        events.append(("gather", tuple(x.shape)))
        # Model a reusable SDK workspace and its safe clone contract.
        state.comm_buff[: x.shape[0]] = torch.cat([x, x + 1], dim=-1)
        return state.comm_buff[: x.shape[0]].clone()

    provider.__dict__["_all_gather_ops"] = SimpleNamespace(
        create_state=create, all_gather_inner=gather
    )
    return provider, events


def test_legacy_provider_and_duck_backend_keep_zero_keyword_abi(monkeypatch):
    calls = []

    class Legacy:
        def all_gather(self, x):
            calls.append(tuple(x.shape))
            return torch.cat([x, x + 10], dim=0)

        def all_reduce(self, x):
            return x

    x = torch.arange(6).view(2, 3)
    backend = MNNVLDistributedImpl(Legacy())
    assert torch.equal(backend.all_gather(x), torch.cat([x, x + 10], 0))
    assert torch.equal(backend.all_gather(x, dim=-1), torch.cat([x, x + 10], -1))
    assert calls == [(2, 3), (3, 2)]
    assert backend.prepare_all_gather() is False
    monkeypatch.setattr(DistributedCommunicator, "plugins", [Legacy()])
    comm = DistributedCommunicator()
    assert torch.equal(comm.all_gather(x, dim=-1), torch.cat([x, x + 10], -1))
    assert comm.prepare_all_gather() is False
    fallback = FallbackDistributedImpl(
        SimpleNamespace(supports=lambda x: False, all_gather=lambda x: pytest.fail("primary")),
        SimpleNamespace(supports=lambda x: True, all_gather=Legacy().all_gather),
    )
    assert torch.equal(fallback.all_gather(x, dim=-1), torch.cat([x, x + 10], -1))
    assert fallback.prepare_all_gather() is False


@pytest.mark.parametrize("dim", [1, -2, True, None])
def test_dimensions_reject_unsupported_values_before_transport(dim):
    backend = MNNVLDistributedImpl(SimpleNamespace(all_gather=lambda x: pytest.fail("transport")))
    with pytest.raises(ValueError, match="dim"):
        backend.all_gather(torch.ones(2, 3), dim=dim)


def test_single_rank_lastdim_is_identity_and_scalar_is_rejected():
    parallel = TensorParallel()
    parallel.group = SimpleNamespace(size=1, rank=0)
    x = torch.ones(2, 3)
    assert parallel.all_gather(x, dim=-1) is x
    assert parallel.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16) is False
    with pytest.raises(ValueError):
        parallel.all_gather(torch.tensor(1))


def test_native_preparation_is_group_bound_warmed_and_reused(native):
    provider, events = native
    assert provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)
    assert ("create", 128, 16) in events and ("gather", (1, 8)) in events
    assert events[-1] == ("drain", torch.device("cuda:0"))
    assert provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)
    assert len([x for x in events if isinstance(x, tuple) and x[0] == "create"]) == 1
    with pytest.raises(ValueError, match="resize"):
        provider.prepare_all_gather(max_rows=256, shard_width=8, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_rows": 128, "shard_width": 8, "dtype": torch.float32},
        {"max_rows": 128, "shard_width": 7, "dtype": torch.bfloat16},
    ],
)
def test_unsupported_metadata_declines_before_sdk(native, options):
    provider, events = native
    assert provider.prepare_all_gather(**options) is False
    assert not any(x[0] == "create" for x in events)


@pytest.mark.parametrize(
    "options",
    [
        {"max_rows": True, "shard_width": 8, "dtype": torch.bfloat16},
        {"max_rows": 0, "shard_width": 8, "dtype": torch.bfloat16},
        {"max_rows": 128, "shard_width": 8},
    ],
)
def test_invalid_metadata_fails_before_sdk(native, options):
    provider, events = native
    with pytest.raises(ValueError, match="positive"):
        provider.prepare_all_gather(**options)
    assert not any(x[0] == "create" for x in events)


def test_import_error_is_only_prelaunch_decline(native, monkeypatch):
    provider, events = native
    del provider.__dict__["_all_gather_ops"]
    import minisgl.distributed.multimem.provider as sgl

    def unavailable(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(sgl.importlib, "import_module", unavailable)
    assert provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16) is False
    assert not any(x[0] == "create" for x in events)


def test_native_outputs_are_owned_inputs_readonly_and_previous_result_retained(native):
    provider, events = native
    provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)
    x = cuda_tensor(3, 8)
    original = x.clone()
    first = provider.all_gather_last_dim(x)
    held = first.clone()
    second = provider.all_gather_last_dim(x * 3)
    assert first.data_ptr() != second.data_ptr() != provider._gather_state.comm_buff.data_ptr()
    assert torch.equal(first, held) and torch.equal(x, original)
    assert first.shape == (3, 16) and first.dtype == torch.bfloat16
    provider.close()
    assert events[-2:] == [("drain", torch.device("cuda:0")), "close"]
    assert provider._gather_state is None
    assert torch.equal(first, held)


@pytest.mark.parametrize("violation", ["stride", "alignment", "grad"])
def test_selected_storage_violation_never_changes_collective(native, violation):
    provider, events = native
    provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)
    if violation == "stride":
        x = cuda_tensor(8, 3).T
    elif violation == "alignment":
        x = cuda_tensor(25)[1:].view(3, 8)
    else:
        x = cuda_tensor(3, 8).requires_grad_()
    assert provider.supports_all_gather_last_dim(x)
    count = len(events)
    with pytest.raises(ValueError, match="storage"):
        provider.all_gather_last_dim(x)
    assert len(events) == count


def test_selected_launch_error_propagates_without_fallback(native):
    provider, events = native
    provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)

    def fail(*args, **kwargs):
        raise RuntimeError("selected device error")

    provider._all_gather_ops.all_gather_inner = fail
    provider.all_gather = lambda x: pytest.fail("retry")
    with pytest.raises(RuntimeError, match="selected device error"):
        provider.all_gather_last_dim(cuda_tensor(3, 8))


def test_capture_preparation_and_unwarmed_use_rejected(native):
    provider, _ = native
    provider._capturing = True
    with pytest.raises(RuntimeError, match="precede"):
        provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)
    provider._capturing = False
    provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)
    provider._gather_warm = False
    with pytest.raises(RuntimeError, match="warmed"):
        provider.all_gather_last_dim(cuda_tensor(3, 8))


def test_close_failure_retains_mapping(native):
    provider, _ = native
    provider.prepare_all_gather(max_rows=128, shard_width=8, dtype=torch.bfloat16)
    state = provider._gather_state

    def fail():
        raise RuntimeError("close failed")

    provider.comm.close = fail
    with pytest.raises(RuntimeError, match="close failed"):
        provider.close()
    assert not provider._closed and provider._gather_state is state


@pytest.mark.parametrize(
    "kind", ["dense", "biased", "float", "packed", "subclass", "unsharded", "absent"]
)
def test_engine_prepares_only_exact_meta_dense_sharded_bf16_head(monkeypatch, kind):
    calls = []

    class CustomLinear(Linear):
        pass

    if kind == "absent":
        head = None
    elif kind == "packed":
        head = Linear(32, 8, weight_format="mxfp4", device="meta")
    else:
        cls = CustomLinear if kind == "subclass" else Linear
        head = cls(
            4,
            16 if kind == "unsharded" else 8,
            has_bias=kind == "biased",
            device="meta",
            dtype=torch.float32 if kind == "float" else torch.bfloat16,
        )
    fake = SimpleNamespace(size=2, prepare_all_gather=lambda **kw: calls.append(kw))
    monkeypatch.setattr("minisgl.distributed.parallel.TensorParallel", lambda: fake)
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        max_running_req=48,
        cuda_graph_max_bs=160,
        cuda_graph_bs=[192],
        model_config=SimpleNamespace(vocab_size=16),
    )
    _prepare_head_all_gather(SimpleNamespace(lm_head=head), config)
    expected = (
        {"max_rows": 192, "shard_width": 8, "dtype": torch.bfloat16}
        if kind == "dense"
        else {"max_rows": None, "shard_width": None, "dtype": None}
    )
    assert calls == [expected]


def test_real_engine_prepares_before_first_memory_snapshot(construction, monkeypatch):
    f = construction
    memory = Engine._sync_get_memory

    def snapshot(self):
        f.events.append("snapshot")
        return memory(self)

    def prepare(model, config):
        assert model is not None and "load_weights" not in f.events and "snapshot" not in f.events
        f.events.append("gather_prepare")

    monkeypatch.setattr(f.engine, "_prepare_head_all_gather", prepare)
    monkeypatch.setattr(Engine, "_sync_get_memory", snapshot)
    Engine(f.config(), worker_owner=None)
    assert (
        f.events.index("gather_prepare") < f.events.index("snapshot") < f.events.index("precheck")
    )
    assert f.events.index("snapshot") < f.events.index("load_weights") < f.events.index("budget")


def test_real_engine_prepare_error_uses_existing_worker_abort(construction, monkeypatch):
    f = construction

    def fail(model, config):
        raise RuntimeError("native setup failure")

    def abort(error):
        assert str(error) == "native setup failure" and "load_weights" not in f.events
        raise LookupError("owned worker abort")

    monkeypatch.setattr(f.engine, "_prepare_head_all_gather", fail)
    with pytest.raises(LookupError, match="owned worker"):
        Engine(f.config(), worker_owner=SimpleNamespace(abort=abort))


def _gloo_worker(rank, init_path, mode, results):
    dist = torch.distributed
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(
            torch.cuda, "_lazy_init", lambda: (_ for _ in ()).throw(AssertionError("CUDA"))
        )
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        dist.init_process_group(
            "gloo",
            init_method="file://" + init_path,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=10),
        )
        group = dist.group.WORLD
        if mode == "dimensions":
            backend = TorchDistributedImpl(group)
            for shape in [(2, 3), (2, 3, 4)]:
                x = (
                    torch.arange(torch.Size(shape).numel(), dtype=torch.bfloat16).view(shape)
                    + rank * 100
                )
                values = [x - rank * 100, x - rank * 100 + 100]
                for dim in (0, -1):
                    actual = backend.all_gather(x, dim=dim)
                    assert torch.equal(actual, torch.cat(values, dim=dim))
                    assert actual.is_contiguous()
            results.put((rank, "exact"))
        else:
            provider = MultimemProvider(
                SimpleNamespace(
                    disabled=False, has_multicast=not (mode == "capability" and rank == 0)
                ),
                group=group,
                output_group=group,
                device="cuda:0",
            )

            def unexpected(*args, **kwargs):
                raise AssertionError("declined/mismatched ranks must not allocate")

            provider.__dict__["_all_gather_ops"] = SimpleNamespace(create_state=unexpected)
            width = 16 if mode == "metadata" and rank == 0 else 8
            try:
                value = provider.prepare_all_gather(
                    max_rows=128, shard_width=width, dtype=torch.bfloat16
                )
                results.put((rank, "declined" if value is False else "unexpected"))
            except ValueError as exc:
                results.put((rank, str(exc)))
    except Exception as exc:
        results.put((rank, "FAILED: " + repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        patch.undo()


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("dimensions", "exact"),
        ("capability", "declined"),
        ("metadata", "All-gather preparation differs across communicator ranks"),
    ],
)
def test_real_gloo_dimensions_and_group_consistent_selection(tmp_path, mode, expected):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_gloo_worker, args=(rank, str(tmp_path / "init"), mode, results))
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(25)
        assert all(not worker.is_alive() for worker in workers), "CPU Gloo worker hung"
        assert [worker.exitcode for worker in workers] == [0, 0]
        assert sorted(results.get(timeout=2) for _ in workers) == [(0, expected), (1, expected)]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
        results.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sharded", [True, False])
def test_head_gathers_original_dtype_then_casts_exactly(dtype, sharded):
    from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM

    model = object.__new__(KimiK3LinearForCausalLM)
    shard = torch.tensor([[1.25, -2.5], [3.0, 0.125]], dtype=dtype)
    model.model = SimpleNamespace(forward=lambda *args: torch.empty(2, 4, dtype=dtype))
    model.lm_head = SimpleNamespace(forward=lambda hidden: shard)
    model._sharded_vocab = sharded
    calls = []

    def gather(value, *, dim):
        assert value is shard and value.dtype == dtype and dim == -1
        calls.append("gather")
        return torch.cat([value, value + 1], dim=-1)

    model._tp = SimpleNamespace(all_gather=gather)
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(extend_len=2)], input_ids=torch.arange(2), positions=torch.arange(2)
    )
    actual = model._forward_batch(batch, None, None, all_logits=True, backend=object())
    expected = torch.cat([shard.float(), (shard + 1).float()], dim=-1) if sharded else shard.float()
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    assert calls == (["gather"] if sharded else [])
