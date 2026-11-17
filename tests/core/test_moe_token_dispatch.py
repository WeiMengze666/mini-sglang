"""CPU contracts/API spies only; these do not execute native GPU transport."""

import importlib.util
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.distributed import fi_moe as F
from minisgl.distributed.info import DistributedInfo
from minisgl.moe.token_dispatch import FlashInferTokenDispatcher


class DeviceTensor(torch.Tensor):
    """CPU storage with CUDA metadata, used only to inspect adapter calls."""

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda:0")

    def record_stream(self, stream):
        pass


def tensor(value, dtype=None):
    return torch.as_tensor(value, dtype=dtype).as_subclass(DeviceTensor)


@pytest.fixture(autouse=True)
def host_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: "stream")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test initialized CUDA"))


class Spy:
    def __init__(self):
        self.calls = []
        self.stream = nullcontext

    def dispatch(self, ids, fields, workspace, meta, m, rank, ep, k, experts, **kw):
        self.calls.append(("dispatch", ids, fields, m, rank, ep, k, experts, kw))
        # Deliberate poison padding: only the independent ID field is sanitized.
        result = [value.new_full((ep, m, value.shape[1]), 7) for value in fields]
        for value, out in zip(fields, result):
            out[rank, : value.shape[0]].copy_(value)
        return result, 123, None

    def sanitize(self, ids, workspace, meta, rank, invalid, **kw):
        self.calls.append(("sanitize", invalid, kw))
        ids[1, -1].fill_(invalid)

    def combine(self, value, n, workspace, meta, m, rank, ep, k, offset, **kw):
        self.calls.append(("combine", value, n, m, rank, ep, k, offset, kw))
        return value[rank, :n].clone()


def dispatcher(monkeypatch, *, rank=0, ep=2, layout=None, output=torch.float32):
    layout = layout or ((torch.float32, 3), (torch.bfloat16, 2), (torch.int32, 2), (torch.uint8, 5))
    r = object.__new__(F.FlashInferMoEWorkspace)
    r.api, r.info, r.device = Spy(), DistributedInfo(rank, ep), torch.device("cuda:0")
    r.capacity, r.stream, r.workspace, r.metainfo, r.group = (
        4,
        "stream",
        object(),
        object(),
        "group",
    )
    r.closed = r.failed = r.busy = False
    r.views = {}
    r.memory = object()
    return FlashInferTokenDispatcher(r, 16, 2, layout, (output, 3))


def inputs(d, n):
    ids = tensor([[0, 15]] * n, torch.int32).reshape(n, 2)
    fields = tuple(
        torch.arange(n * width).reshape(n, width).to(dtype).as_subclass(DeviceTensor)
        for dtype, width in d.payload_layout
    )
    return ids, fields


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_typed_payloads_keep_values_bits_and_source_order(monkeypatch, dtype):
    d = dispatcher(monkeypatch, output=dtype)
    ids, fields = inputs(d, 2)
    saved = tuple(value.clone() for value in (ids, *fields))
    h = d.dispatch(ids, fields, source_counts=(2, 1))
    call = d.resource.api.calls[0]
    assert [v.element_size() for v in call[2]] == sorted(
        [v.element_size() for v in (ids, *fields)], reverse=True
    )
    assert {id(v) for v in call[2]} == {id(v) for v in (ids, *fields)}
    for original, actual in zip(fields, h.payloads):
        assert actual.dtype == original.dtype and actual.shape == (2, 2, original.shape[1])
        torch.testing.assert_close(actual[0], original, rtol=0, atol=0)
    assert h.expert_ids[0, 0, 1] == 15  # Foreign expert ID survives.
    assert h.expert_ids[1, -1].eq(-1).all()
    assert h.payloads[2][1, -1].eq(7).all()  # Opaque packed carrier is not sanitized.
    for before, after in zip(saved, (ids, *fields)):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    partials = tensor(torch.arange(12).reshape(4, 3), dtype)
    actual = d.combine(h, partials)
    torch.testing.assert_close(actual, partials[:2], rtol=0, atol=0)
    assert actual.dtype == dtype and h._consumed and not d.resource.busy
    c = d.resource.api.calls[-1]
    assert c[2:8] == (2, 2, 0, 2, 2, 123)
    assert c[-1] == {
        "payload_in_workspace": False,
        "output_dtype": dtype,
        "use_low_precision": False,
        "backend": "trtllm",
    }
    with pytest.raises(RuntimeError, match="single-use"):
        d.combine(h, partials)


def test_fp8_and_opaque_scale_bytes_are_not_cast(monkeypatch):
    d = dispatcher(
        monkeypatch, layout=((torch.float8_e4m3fn, 4), (torch.uint8, 8), (torch.int32, 2))
    )
    ids, fields = inputs(d, 1)
    h = d.dispatch(ids, fields, source_counts=(1, 1))
    for a, b in zip(fields, h.payloads):
        assert b.dtype == a.dtype
        assert torch.equal(a.view(torch.uint8), b[0].view(torch.uint8))


def test_local_empty_participates_but_common_empty_skips_native(monkeypatch):
    d = dispatcher(monkeypatch)
    ids, fields = inputs(d, 0)
    h = d.dispatch(ids, fields, source_counts=(0, 2))
    assert d.resource.api.calls[0][0] == "dispatch" and h.expert_ids.shape == (2, 2, 2)
    assert d.combine(h, tensor(torch.zeros(4, 3))).shape == (0, 3)
    d.resource.api.calls.clear()
    h = d.dispatch(ids, fields, source_counts=(0, 0))
    assert d.combine(h, tensor(torch.empty(0, 3))).shape == (0, 3)
    assert d.resource.api.calls == []


def test_outstanding_handle_blocks_reuse_wrong_owner_and_close(monkeypatch):
    d = dispatcher(monkeypatch)
    ids, fields = inputs(d, 1)
    h = d.dispatch(ids, fields, source_counts=(1, 1))
    with pytest.raises(RuntimeError, match="previous"):
        d.dispatch(ids, fields, source_counts=(1, 1))
    with pytest.raises(RuntimeError, match="single-use"):
        dispatcher(monkeypatch).combine(h, tensor(torch.zeros(2, 3)))
    with pytest.raises(RuntimeError, match="Finish"):
        d.close()
    assert h._owner is d and h._inputs[0] is ids and d.resource.memory is not None


@pytest.mark.parametrize("kind", ["shape", "dtype", "stride", "counts", "capacity", "cpu"])
def test_bad_producer_contract_poisoned_before_native(monkeypatch, kind):
    d = dispatcher(monkeypatch)
    ids, fields = inputs(d, 1)
    counts = (1, 1)
    if kind == "shape":
        ids = ids[:, :1]
    elif kind == "dtype":
        ids = ids.long()
    elif kind == "stride":
        ids, fields = inputs(d, 2)
        fields = (tensor(torch.zeros(2, 6))[:, ::2], *fields[1:])
        counts = (2, 1)
    elif kind == "counts":
        counts = (0, 1)
    elif kind == "capacity":
        counts = (1, 5)
    else:
        ids = ids.as_subclass(torch.Tensor)
    with pytest.raises(ValueError, match="producer"):
        d.dispatch(ids, fields, source_counts=counts)
    assert d.resource.failed and d.resource.api.calls == []
    with pytest.raises(RuntimeError, match="cannot be retried"):
        d.dispatch(ids, fields, source_counts=counts)


@pytest.mark.parametrize("stage", ["dispatch", "sanitize", "combine"])
def test_native_failure_is_fatal_and_never_retries(monkeypatch, stage):
    d = dispatcher(monkeypatch)
    ids, fields = inputs(d, 1)

    def fail(*args, **kw):
        raise RuntimeError("native failure")

    monkeypatch.setattr(d.resource.api, stage, fail)
    with pytest.raises(RuntimeError, match="native failure"):
        h = d.dispatch(ids, fields, source_counts=(1, 1))
        d.combine(h, tensor(torch.zeros(2, 3)))
    assert d.resource.failed
    with pytest.raises(RuntimeError, match="cannot be retried"):
        d.dispatch(ids, fields, source_counts=(1, 1))
    with pytest.raises(RuntimeError, match="worker abort"):
        d.close()


def test_precision_change_at_combine_is_fatal(monkeypatch):
    d = dispatcher(monkeypatch)
    ids, fields = inputs(d, 1)
    h = d.dispatch(ids, fields, source_counts=(1, 1))
    with pytest.raises(ValueError, match="precision"):
        d.combine(h, tensor(torch.zeros(2, 3), torch.bfloat16))
    assert d.resource.failed


def test_close_orders_drain_barrier_and_keeps_native_owner(monkeypatch):
    d = dispatcher(monkeypatch)
    events, owner = [], d.resource.memory
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("drain"))
    monkeypatch.setattr(F.dist, "barrier", lambda group: events.append("barrier"))
    d.resource.views["borrowed"] = object()
    d.close()
    d.close()
    assert events == ["drain", "barrier"] and d.resource.closed
    assert d.resource.memory is owner and not d.resource.views
    with pytest.raises(RuntimeError, match="cannot be retried"):
        d.resource.check()


def test_one_workspace_cannot_silently_cross_execution_streams(monkeypatch):
    d = dispatcher(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: "other")
    with pytest.raises(RuntimeError, match="stream/channel"):
        d.resource.check()


def test_prepare_derives_capacity_and_ep1_never_loads_provider(monkeypatch):
    calls = []
    monkeypatch.setattr(F.FlashInferMoEWorkspace, "prepare", lambda *a, **kw: calls.append(kw))
    kwargs = {
        "num_experts": 896,
        "top_k": 16,
        "payload_layout": (
            (torch.float8_e4m3fn, 3584),
            (torch.float8_e4m3fn, 112),
            (torch.int32, 16),
        ),
        "output_layout": (torch.bfloat16, 7168),
        "max_prefill_tokens": 16385,
        "max_decode_tokens": 48,
    }
    assert FlashInferTokenDispatcher.prepare(DistributedInfo(0, 1), None, "cpu", **kwargs) is None
    assert calls == []
    assert FlashInferTokenDispatcher.prepare(DistributedInfo(0, 8), None, "cpu", **kwargs) is None
    spec = calls[0]
    assert (
        spec["capacity"] == 2049
        and spec["dispatch_bytes"] == 3824
        and spec["combine_bytes"] == 14336
    )


def _gloo_worker(rank, rendezvous):
    import minisgl.distributed.fi_moe as f

    torch.set_num_threads(1)
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(AssertionError("CUDA init"))
    f._native_api = lambda: (_ for _ in ()).throw(AssertionError("CPU decline imported FI"))
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    groups = [dist.new_group(members, backend="gloo") for members in ([0, 1], [2, 3])]
    group = groups[rank // 2]
    info = DistributedInfo(rank % 2, 2)
    kwargs = {"capacity": 4, "dispatch_bytes": 32, "combine_bytes": 16, "layout": "same"}
    assert f.FlashInferMoEWorkspace.prepare(info, group, "cpu", **kwargs) is None
    with pytest.raises(ValueError, match="disagreed"):
        f.FlashInferMoEWorkspace.prepare(info, group, "cpu", **{**kwargs, "layout": rank})
    # Actual mature CPU backend: subgroup root 0 is global rank 2 in second group.
    source = Path(torch.__file__).parent.parent / "flashinfer/comm/comm_backend.py"
    spec = importlib.util.spec_from_file_location("fi_comm_backend_ep_cpu", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.TorchDistBackend(group)
    assert backend.Get_rank() == rank % 2 and backend.Get_size() == 2
    assert backend.bcast(rank if rank % 2 == 0 else None, root=0) == rank // 2 * 2
    assert backend.allgather(rank) == (rank // 2 * 2, rank // 2 * 2 + 1)
    dist.barrier()
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in multiprocess gate"
)
def test_real_gloo_subgroups_agree_before_allocation_and_translate_root(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)


@pytest.mark.parametrize("ep", [2, 8])
def test_instance_native_workspace_uses_group_owner_and_strided_bytes(monkeypatch, ep):
    events = []
    group = object()

    class Memory:
        mapped = True

        def __init__(self, **kw):
            self.kw = kw
            events.append(("memory", kw))

        def get_allocation_size(self):
            return 128

        def get_buffer_ptrs_host(self):
            return [1024 + i * 128 for i in range(ep)]

    def pack(ptr, width, stride, size, dtype, device):
        events.append(("pack", ptr, width, stride, size, dtype, device))
        return torch.empty_strided((size, width), (stride, 1), dtype=dtype).as_subclass(
            DeviceTensor
        )

    def initialize(workspace, rank, size, capacity, **kw):
        events.append(("initialize", rank, size, capacity, kw))
        return torch.zeros(8, dtype=torch.int64)

    api = SimpleNamespace(
        size=lambda *a, **kw: 64,
        memory=Memory,
        backend=lambda g: ("backend", g),
        pack=pack,
        initialize=initialize,
        stream=nullcontext,
    )
    monkeypatch.setattr(F.dist, "barrier", lambda group: events.append(("barrier", group)))
    r = F.FlashInferMoEWorkspace(
        api, DistributedInfo(0, ep), group, torch.device("cuda:0"), 4, 32, 16
    )
    assert r.workspace.shape == (ep, 64) and r.workspace.stride() == (128, 1)
    assert r.workspace._moe_memory_owner is r.memory
    assert r.memory.kw == {
        "buf_size": 64,
        "group_size": ep,
        "group_rank": 0,
        "device_idx": 0,
        "comm_backend_for_handle_transfer": ("backend", group),
        "enable_multicast": False,
        "allocate_signal_pads": False,
    }
    assert [v[0] for v in events] == ["memory", "pack", "initialize", "barrier"]


@pytest.mark.parametrize(
    "mode", ["unsupported", "mixed", "mismatch", "capture", "capability_error"]
)
def test_prepare_collective_decline_or_error_precedes_native_allocation(monkeypatch, mode):
    events = []
    monkeypatch.setattr(F.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(F.dist, "get_process_group_ranks", lambda group: [2, 3])
    monkeypatch.setattr(F.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode == "capture")

    def supported(**kw):
        assert kw == {"dcp_size": 2, "tp_size": 2, "pp_size": 1, "nnodes": 1}
        if mode == "capability_error":
            raise RuntimeError("capability failed")
        return mode != "unsupported"

    monkeypatch.setattr(F, "_native_api", lambda: SimpleNamespace(supported=supported))

    def gather(rows, local, group):
        events.append("agreement")
        rows[:] = [local, local]
        if mode == "mixed":
            rows[1] = (*local[:2], False)
        elif mode == "mismatch":
            rows[1] = (("different",), None, True)

    monkeypatch.setattr(F.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        F.FlashInferMoEWorkspace,
        "__init__",
        lambda *a: pytest.fail("declined preparation allocated"),
    )
    call = lambda: F.FlashInferMoEWorkspace.prepare(
        DistributedInfo(0, 2),
        object(),
        "cuda:0",
        capacity=4,
        dispatch_bytes=32,
        combine_bytes=16,
        layout="same",
    )
    if mode in ("capture", "capability_error", "mismatch"):
        with pytest.raises(ValueError, match="disagreed"):
            call()
    else:
        assert call() is None
    assert events == ["agreement"]


@pytest.mark.parametrize("missing", ["flashinfer", "flashinfer.comm.trtllm_moe_alltoall"])
def test_missing_optional_provider_target_declines(monkeypatch, missing):
    F._native_api.cache_clear()

    def unavailable(name):
        raise ModuleNotFoundError("optional provider absent", name=missing)

    with monkeypatch.context() as patch:
        patch.setattr(F.importlib, "import_module", unavailable)
        assert F._native_api() is None
    F._native_api.cache_clear()


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("internal dependency absent", name="cuda.bindings.runtime"),
        ImportError("installed provider ABI mismatch"),
        OSError("installed provider shared library failed"),
    ],
)
def test_broken_installed_provider_cannot_silently_decline(monkeypatch, failure):
    F._native_api.cache_clear()

    def broken(name):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(F.importlib, "import_module", broken)
        with pytest.raises(type(failure), match=str(failure)):
            F._native_api()
    F._native_api.cache_clear()
