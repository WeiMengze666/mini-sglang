"""CPU protocol checks; fake CUDA metadata must never initialize CUDA."""

import multiprocessing
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed.impl import DistributedImpl, FallbackDistributedImpl
from minisgl.distributed.mnnvl import MNNVLAllReduceProvider, MNNVLDistributedImpl
from minisgl.distributed.sgl import SGLangMNNVLProvider


class CudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda:0")


def cuda_tensor(*shape, dtype=torch.bfloat16):
    return torch.ones(shape, dtype=dtype).as_subclass(CudaTensor)


@pytest.fixture
def host(monkeypatch):
    events = []

    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU protocol test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("drain"))
    group = SimpleNamespace(group_name="owned-gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda candidate: 2)

    def gather(peers, packet, *, group):
        assert group.group_name == "owned-gloo"
        events.append("agreement")
        peers[:] = [packet, packet]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    class Handle:
        multicast_ptr = 1 << 40

        def __del__(self):
            events.append("release")

    class Memory:
        @staticmethod
        def empty_strided_p2p(shape, strides, dtype, device, group_name):
            assert strides == [1] and group_name == "owned-gloo"
            assert device == torch.device("cuda:0")
            assert not torch.is_inference_mode_enabled()
            events.append("allocate")
            return cuda_tensor(*shape, dtype=dtype)

        @staticmethod
        def rendezvous(tensor):
            events.append("rendezvous")
            return Handle()

    monkeypatch.setattr(torch._C._distributed_c10d, "_SymmetricMemory", Memory)
    comm = SimpleNamespace(
        disabled=False,
        has_multicast=True,
        pull_enabled=True,
        max_push_size=768 * 1024,
        world_size=2,
        obj=object(),
        should_custom_ar=lambda x: True,
        custom_all_reduce=lambda x: x * 2,
        close=lambda: events.append("comm-close"),
    )
    provider = SGLangMNNVLProvider(
        comm,
        group=group,
        output_group=group,
        device=torch.device("cuda:0"),
        execution_context=nullcontext,
    )
    return provider, events, group


def test_reserve_identity_alias_and_close_order(host):
    provider, events, _ = host
    with torch.inference_mode():
        output = provider.reserve_output("projection", (64, 7168), torch.bfloat16)
    assert provider.reserve_output("projection", (64, 7168), torch.bfloat16) is output
    assert events == ["agreement", "allocate", "agreement", "rendezvous", "agreement", "agreement"]
    alias = output[2:50]
    assert provider._output_mc_ptr(alias) == (1 << 40) + 2 * 7168 * 2
    assert provider._output_mc_ptr(cuda_tensor(48, 7168)) is None
    provider.close()
    assert events[-3:] == ["drain", "comm-close", "release"]
    provider.close()
    assert events.count("comm-close") == 1


def test_reservation_metadata_and_capture_rejected(host):
    provider, events, _ = host
    provider.reserve_output("output", (8, 16), torch.bfloat16)
    with pytest.raises(ValueError, match="resize"):
        provider.reserve_output("output", (9, 16), torch.bfloat16)
    with pytest.raises(ValueError, match="resize"):
        provider.reserve_output("output", (8, 16), torch.float32)
    with provider.capture(), pytest.raises(RuntimeError, match="precede"):
        provider.reserve_output("another", (8, 16), torch.bfloat16)
    assert events.count("allocate") == 1


@pytest.mark.parametrize("mismatch", [False, True])
def test_all_rank_agreement_before_allocation(host, monkeypatch, mismatch):
    provider, events, _ = host

    def gather(peers, packet, *, group):
        spec, action, error, available = packet
        peer_spec = ("other", *spec[1:]) if mismatch else spec
        peers[:] = [packet, (peer_spec, action, error, False)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if mismatch:
        with pytest.raises(ValueError, match="differs"):
            provider.reserve_output("output", (8, 16), torch.bfloat16)
    else:
        assert provider.reserve_output("output", (8, 16), torch.bfloat16) is None
    assert events == []


@pytest.mark.parametrize("failure", ["drain", "comm-close"])
def test_teardown_failure_retains_mapping(host, monkeypatch, failure):
    provider, events, _ = host
    provider.reserve_output("output", (8, 16), torch.bfloat16)

    def fail(*args):
        raise RuntimeError(failure)

    if failure == "drain":
        monkeypatch.setattr(torch.cuda, "synchronize", fail)
    else:
        provider.comm.close = fail
    with pytest.raises(RuntimeError, match=failure):
        provider.close()
    assert provider._outputs and "release" not in events


@pytest.fixture
def native_ops(host, monkeypatch):
    from minisgl.distributed import sgl

    provider, events, _ = host

    class Kernels:
        def push_res(self, comm, x, residual):
            assert comm is provider.comm.obj
            events.append("push")
            x.mul_(2)
            if residual is not None:
                x.add_(residual)

        def pull_res(self, comm, x, residual, mc_ptr, blocks, unroll):
            assert comm is provider.comm.obj
            assert mc_ptr == provider._output_mc_ptr(x)
            assert (blocks, unroll) == (4, 8)
            events.append("pull")
            x.mul_(2)
            if residual is not None:
                x.add_(residual)

    kernels = Kernels()
    ops = SimpleNamespace(
        _jit_module=lambda world_size: kernels,
        RES_TUNING=object(),
        _resolve_tuning=lambda table, **kwargs: SimpleNamespace(num_blocks=4, unroll=8),
    )
    original = sgl.importlib.import_module
    monkeypatch.setattr(
        sgl.importlib,
        "import_module",
        lambda name: (
            ops
            if name.endswith(".all_reduce_residual")
            else SimpleNamespace(_PUSH_MAX_BYTES=512 * 1024)
            if name.endswith(".k3_ar_fusion")
            else original(name)
        ),
    )
    return provider, events, kernels


def test_instance_bound_push_and_pull(native_ops):
    provider, events, _ = native_ops
    x = cuda_tensor(8, 16)
    result = provider.all_reduce_residual(x, cuda_tensor(8, 16))
    assert result is x and torch.equal(result, torch.full_like(result, 3))
    output = provider.reserve_output("projection", (64, 7168), torch.bfloat16)
    alias = output[:48]
    assert provider.supports_all_reduce_residual(alias, output_key="projection")
    assert provider.all_reduce_residual(alias, output_key="projection") is alias
    assert events.count("push") == 1 and events.count("pull") == 1


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "dtype",
        "empty",
        "tail",
        "residual_shape",
        "residual_dtype",
        "norm",
        "disabled",
        "no_pull",
    ],
)
def test_native_selector_declines_before_dispatch(native_ops, case):
    provider, events, _ = native_ops
    x, residual, norm = cuda_tensor(8, 16), None, None
    if case == "cpu":
        x = torch.ones((8, 16), dtype=torch.bfloat16)
    elif case == "dtype":
        x = cuda_tensor(8, 16, dtype=torch.float32)
    elif case == "empty":
        x = cuda_tensor(0, 16)
    elif case == "tail":
        x = cuda_tensor(7)
    elif case == "residual_shape":
        residual = cuda_tensor(16, 8)
    elif case == "residual_dtype":
        residual = cuda_tensor(8, 16, dtype=torch.float32)
    elif case == "norm":
        norm = lambda y: y
    elif case == "disabled":
        provider.comm.disabled = True
    elif case == "no_pull":
        x = provider.reserve_output("projection", (48, 7168), torch.bfloat16)
        provider.comm.pull_enabled = False
    kwargs = {"output_key": "projection"} if case == "no_pull" else {}
    assert not provider.supports_all_reduce_residual(x, residual, norm=norm, **kwargs)
    with pytest.raises(ValueError, match="does not cover"):
        provider.all_reduce_residual(x, residual, norm=norm, **kwargs)
    assert "push" not in events and "pull" not in events


def test_decline_composes_generic_fallback(native_ops):
    provider, events, _ = native_ops
    backend = MNNVLDistributedImpl(provider)
    x = cuda_tensor(8, 16, dtype=torch.float32)
    result = backend.all_reduce_residual(x, cuda_tensor(8, 16, dtype=torch.float32))
    assert torch.equal(result, torch.full_like(result, 3))
    assert "push" not in events and "pull" not in events


def test_started_dispatch_error_never_falls_back(native_ops):
    provider, events, kernels = native_ops

    def fail(*args):
        raise RuntimeError("native collective failure")

    kernels.push_res = fail
    provider.comm.custom_all_reduce = lambda x: pytest.fail("fell back after native dispatch")
    with pytest.raises(RuntimeError, match="native collective"):
        MNNVLDistributedImpl(provider).all_reduce_residual(cuda_tensor(8, 16))


def test_plain_provider_optional_reservation_and_wrapper_fallback(host):
    provider, _, _ = host
    plain = MNNVLAllReduceProvider(provider.comm)
    assert plain.reserve_output("output", (8, 16), torch.bfloat16) is None

    class Plain(DistributedImpl):
        def all_reduce(self, x):
            return x

        def all_gather(self, x):
            return x

    wrapped = MNNVLDistributedImpl(provider)
    backend = FallbackDistributedImpl(Plain(), wrapped)
    assert backend.reserve_output("output", (8, 16), torch.bfloat16) is not None


def test_same_world_size_keeps_instance_communicators(host):
    provider, _, group = host
    other_comm = SimpleNamespace(**vars(provider.comm))
    other_comm.obj = object()
    other = SGLangMNNVLProvider(
        other_comm, group=group, output_group=group, device=torch.device("cuda:0")
    )
    observed = []

    def push(comm, x, residual):
        observed.append(comm)

    ops = SimpleNamespace(_jit_module=lambda size: SimpleNamespace(push_res=push))
    for current in (provider, other):
        current._residual_ops = ops
        current._push_max_bytes = 512 * 1024
        current.all_reduce_residual(cuda_tensor(8, 16))
    assert observed == [provider.comm.obj, other.comm.obj]
    assert observed[0] is not observed[1]


@pytest.mark.parametrize("local_reuse", [False, True])
def test_reuse_new_disagreement_is_collective(host, monkeypatch, local_reuse):
    provider, events, _ = host
    if local_reuse:
        provider.reserve_output("output", (8, 16), torch.bfloat16)
    before = list(events)

    def gather(peers, packet, *, group):
        spec, action, error, available = packet
        events.append("agreement")
        peers[:] = [packet, (spec, "new" if local_reuse else "reuse", error, available)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="reuse/new"):
        provider.reserve_output("output", (8, 16), torch.bfloat16)
    assert events == before + ["agreement"]


@pytest.mark.parametrize("invalid", ["shape", "dtype", "closed", "capture", "resize"])
def test_local_error_is_agreed_before_raising(host, monkeypatch, invalid):
    provider, events, _ = host
    if invalid == "resize":
        provider.reserve_output("output", (8, 16), torch.bfloat16)
    before = list(events)

    def gather(peers, packet, *, group):
        assert packet[1] == "error" and packet[2] is not None
        events.append("agreement")
        peers[:] = [packet, (("output", (8, 16), "torch.bfloat16"), "new", None, True)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if invalid == "closed":
        provider._closed = True
    if invalid == "capture":
        provider._capturing = True
    shape = (0, 16) if invalid == "shape" else (9, 16) if invalid == "resize" else (8, 16)
    dtype = "bf16" if invalid == "dtype" else torch.bfloat16
    with pytest.raises((ValueError, RuntimeError)):
        provider.reserve_output("output", shape, dtype)
    assert events == before + ["agreement"]


@pytest.mark.parametrize("layout", ["strided", "unaligned"])
def test_push_local_layout_normalizes_without_selector_divergence(native_ops, layout):
    provider, events, _ = native_ops
    x = cuda_tensor(16, 8).t() if layout == "strided" else cuda_tensor(129)[1:].view(8, 16)
    residual = cuda_tensor(129)[1:].view(8, 16)
    assert provider.supports_all_reduce_residual(x, residual)
    result = provider.all_reduce_residual(x, residual)
    assert result is not x and result.is_contiguous() and result.data_ptr() % 16 == 0
    assert torch.equal(result, torch.full_like(result, 3))
    assert torch.equal(x, torch.ones_like(x))
    assert events.count("push") == 1


@pytest.mark.parametrize("layout", ["unowned", "strided", "unaligned"])
def test_pull_local_storage_fault_fails_without_fallback(native_ops, layout):
    provider, events, _ = native_ops
    output = provider.reserve_output("output", (48, 7168), torch.bfloat16)
    x = (
        cuda_tensor(48, 7168)
        if layout == "unowned"
        else output.t()
        if layout == "strided"
        else output.flatten()[1:-7]
    )
    assert provider.supports_all_reduce_residual(x, output_key="output")
    provider.comm.custom_all_reduce = lambda x: pytest.fail("rank-local fallback")
    with pytest.raises(ValueError, match="reserved multicast"):
        MNNVLDistributedImpl(provider).all_reduce_residual(x, output_key="output")
    assert "push" not in events and "pull" not in events


def test_fusion_requires_pre_capture_warmup(native_ops, monkeypatch):
    provider, events, _ = native_ops
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="Warm native"):
        provider.all_reduce_residual(cuda_tensor(8, 16))
    assert "push" not in events
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    provider.all_reduce_residual(cuda_tensor(8, 16))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    provider.all_reduce_residual(cuda_tensor(8, 16))
    assert events.count("push") == 2


@pytest.mark.parametrize("reservation", ["none", "unrelated", "same_input"])
def test_plain_large_input_preserves_generic_fallback(native_ops, reservation):
    provider, events, _ = native_ops
    x = cuda_tensor(48, 7168)
    if reservation != "none":
        output = provider.reserve_output("output", (48, 7168), torch.bfloat16)
        if reservation == "same_input":
            x = output
    assert not provider.supports_all_reduce_residual(x)
    result = MNNVLDistributedImpl(provider).all_reduce_residual(x, torch.ones_like(x))
    assert torch.equal(result, torch.full_like(result, 3))
    assert "push" not in events and "pull" not in events


@pytest.mark.parametrize("key", ["missing", "other"])
def test_explicit_pull_key_must_match_its_own_reservation(native_ops, key):
    provider, events, _ = native_ops
    output = provider.reserve_output("output", (48, 7168), torch.bfloat16)
    provider.reserve_output("other", (48, 7168), torch.bfloat16)
    assert provider.supports_all_reduce_residual(output, output_key=key)
    provider.comm.custom_all_reduce = lambda x: pytest.fail("fallback after explicit route")
    with pytest.raises(ValueError, match="reserved multicast"):
        MNNVLDistributedImpl(provider).all_reduce_residual(output, output_key=key)
    assert "pull" not in events


@pytest.mark.parametrize("algorithm", ["push", "pull"])
@pytest.mark.parametrize("overlap", ["exact", "partial"])
def test_native_residual_overlap_uses_prewrite_snapshot(native_ops, algorithm, overlap):
    provider, events, _ = native_ops
    if algorithm == "pull":
        storage = provider.reserve_output("output", (49, 7168), torch.bfloat16)
        x = storage[:48]
        residual = x if overlap == "exact" else storage[1:]
        kwargs = {"output_key": "output"}
    else:
        storage = cuda_tensor(9, 16)
        x = storage[:8]
        residual = x if overlap == "exact" else storage[1:]
        kwargs = {}
    expected = x.clone() * 2 + residual.clone()
    result = provider.all_reduce_residual(x, residual, **kwargs)
    assert result is x and torch.equal(result, expected)
    assert events.count(algorithm) == 1


def test_native_forward_paths_never_use_cpu_collectives(native_ops, monkeypatch):
    provider, events, _ = native_ops
    output = provider.reserve_output("output", (48, 7168), torch.bfloat16)

    def fail(*args, **kwargs):
        pytest.fail("forward performed a host collective")

    monkeypatch.setattr(torch.distributed, "all_gather_object", fail)
    provider.all_reduce_residual(cuda_tensor(8, 16))
    provider.all_reduce_residual(output, output_key="output")
    assert events.count("push") == events.count("pull") == 1


def test_output_key_reaches_communicator_and_tensor_parallel(monkeypatch):
    from minisgl.distributed.impl import DistributedCommunicator
    from minisgl.distributed.info import DistributedInfo
    from minisgl.distributed.parallel import TensorParallel

    calls = []
    backend = SimpleNamespace(
        all_reduce_residual=lambda x, residual=None, *, norm=None, output_key=None: (
            calls.append(output_key) or x
        )
    )
    monkeypatch.setattr(DistributedCommunicator, "plugins", [backend])
    x = torch.ones(8)
    assert DistributedCommunicator().all_reduce_residual(x, output_key="slot") is x
    tp = TensorParallel()
    tp.group = DistributedInfo(0, 2)
    assert tp.all_reduce_residual(x, output_key="slot") is x
    assert calls == ["slot", "slot"]


def _reservation_failure_worker(rank, init_path, failure, results):
    """Real Gloo control flow; symmetric allocation/mapping stay CPU fakes."""
    dist = torch.distributed
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=3),
    )
    patch = pytest.MonkeyPatch()
    rendezvous_count = 0
    provider = None
    try:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

        def no_cuda(*args, **kwargs):
            raise AssertionError("CPU Gloo test initialized CUDA")

        patch.setattr(torch.cuda, "_lazy_init", no_cuda)

        class Memory:
            @staticmethod
            def empty_strided_p2p(shape, strides, dtype, device, group_name):
                if failure == "allocation" and rank == 0:
                    raise RuntimeError("simulated rank-local allocation OOM")
                return torch.empty(shape, dtype=dtype)

            @staticmethod
            def rendezvous(tensor):
                nonlocal rendezvous_count
                rendezvous_count += 1
                peers = [None, None]
                dist.all_gather_object(peers, rank)
                if failure == "mapping_exception" and rank == 0:
                    raise RuntimeError("simulated rank-local mapping failure")
                return SimpleNamespace(
                    multicast_ptr=0 if failure == "mapping_zero" and rank == 0 else 1 << 40
                )

        patch.setattr(torch._C._distributed_c10d, "_SymmetricMemory", Memory)
        provider = MNNVLAllReduceProvider(
            SimpleNamespace(disabled=False, has_multicast=True),
            output_group=dist.group.WORLD,
            device="cuda:0",
        )
        if rank == 0 and failure == "reuse_new":
            provider._outputs["output"] = (torch.empty(8, 16, dtype=torch.bfloat16), object())
        if rank == 0 and failure == "closed":
            provider._closed = True
        shape = (0, 16) if rank == 0 and failure == "invalid" else (8, 16)
        provider.reserve_output("output", shape, torch.bfloat16)
        results.put((rank, "unexpected success", rendezvous_count, "output" in provider._outputs))
    except Exception as exc:
        results.put((rank, str(exc), rendezvous_count, bool(provider and provider._outputs)))
    finally:
        patch.undo()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "failure, expected, rendezvous_count",
    [
        ("allocation", "allocation failed on rank 0", 0),
        ("mapping_exception", "mapping failed on rank 0", 1),
        ("mapping_zero", "mapping failed on rank 0", 1),
        ("reuse_new", "reuse/new state differs", 0),
        ("closed", "closed communicator", 0),
        ("invalid", "positive integer shape", 0),
    ],
)
def test_reservation_failures_have_matching_terminal_states(
    tmp_path, failure, expected, rendezvous_count
):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_reservation_failure_worker,
            args=(rank, str(tmp_path / "gloo-init"), failure, results),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(15)
        assert all(not process.is_alive() for process in processes), "Gloo worker hung"
        assert [process.exitcode for process in processes] == [0, 0]
        observed = sorted(results.get(timeout=2) for _ in processes)
        assert observed[0][1] == observed[1][1]
        assert expected in observed[0][1]
        assert [item[2] for item in observed] == [rendezvous_count] * 2
        if failure != "reuse_new":
            assert not any(item[3] for item in observed), "partial reservation was published"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        results.close()


@pytest.mark.parametrize("world_size", [1, 2])
def test_tensor_parallel_reservation_facade(monkeypatch, world_size):
    from minisgl.distributed.impl import DistributedCommunicator
    from minisgl.distributed.info import DistributedInfo
    from minisgl.distributed.parallel import TensorParallel

    output = torch.empty(8, 16, dtype=torch.bfloat16)
    calls = []

    def reserve(key, shape, dtype):
        calls.append((key, shape, dtype))
        return output

    monkeypatch.setattr(
        DistributedCommunicator, "plugins", [SimpleNamespace(reserve_output=reserve)]
    )
    tp = TensorParallel()
    tp.group = DistributedInfo(0, world_size)
    actual = tp.reserve_output("slot", (8, 16), torch.bfloat16)
    if world_size == 1:
        assert actual is None and calls == []
    else:
        assert actual is output and calls == [("slot", (8, 16), torch.bfloat16)]
