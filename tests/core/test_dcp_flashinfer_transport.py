"""Host protocol/source ABI only: no FI kernel or CUDA execution."""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.distributed import context as C
from minisgl.distributed import flashinfer_dcp as F
from minisgl.distributed.impl import DistributedCommunicator, destroy_distributed
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.layout import configure_parallel


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CPU DCP protocol initialized CUDA")
    )


@pytest.mark.parametrize(
    "dtype,dim", [(torch.float32, 512), (torch.bfloat16, 3), (torch.bfloat16, 2048)]
)
def test_unsupported_partial_contract_declines_before_optional_import(monkeypatch, dtype, dim):
    monkeypatch.setattr(
        F, "_native_api", lambda: pytest.fail("unsupported metadata loaded native ABI")
    )
    assert (
        F.native_candidate(
            F.AttentionExchangeSpec(dtype, dim), torch.device("cuda:1"), tp_size=8, cp_size=4
        )
        is None
    )


@pytest.mark.parametrize("base", ["e", "2"])
@pytest.mark.parametrize("empty", [False, True])
def test_merge_matches_native_dtype_and_invalid_lse_contract(base, empty):
    values = torch.tensor(
        [
            [[[0.0, 0.0], [5.0, 7.0], [5.0, 7.0]]],
            [[[0.0, 0.0] if empty else [4.0, 6.0], [9.0, 11.0], [9.0, 11.0]]],
        ],
        dtype=torch.bfloat16,
    )
    lse = torch.tensor([[[-torch.inf, 0.0, 0.0]], [[-torch.inf if empty else 0.0, 0.0, torch.inf]]])
    saved = values.clone()
    actual = C.merge_attention(values, lse, lse_base=base)
    assert actual.dtype == values.dtype
    torch.testing.assert_close(
        actual[:, 0],
        torch.tensor([[0.0, 0.0] if empty else [4.0, 6.0]]).to(values.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual[:, 1], torch.tensor([[7.0, 9.0]]).to(values.dtype), rtol=0, atol=0
    )
    torch.testing.assert_close(actual[:, 2], values[0, :, 2], rtol=0, atol=0)
    torch.testing.assert_close(values, saved, rtol=0, atol=0)


def test_only_selected_decode_backend_prepares_communication():
    from minisgl.attention.base import HybridAttnBackend, PrefillDecodeAttnBackend

    calls = []
    leaf = lambda name: SimpleNamespace(
        prepare_communication=lambda **kwargs: calls.append((name, kwargs))
    )
    prefill = leaf("prefill")
    decoder = HybridAttnBackend(leaf("KDA"), leaf("MLA"), lambda _: False)
    PrefillDecodeAttnBackend(prefill, decoder).prepare_communication(dtype=torch.bfloat16)
    assert calls == [("MLA", {"dtype": torch.bfloat16})]


def test_actual_decode_producer_declares_latent512_not_final128():
    from minisgl.attention.config import MLAConfig
    from minisgl.attention.mla.flashinfer import FlashInferMLAAdapter

    config = MLAConfig(7168, 64, 128, 64, 128, 512, 1536, 16384)
    producer = FlashInferMLAAdapter()
    records = tuple((index, config) for index in range(24))
    assert producer.decode_partial_spec(records, dtype=torch.bfloat16) == F.AttentionExchangeSpec(
        torch.bfloat16, 512, "e"
    )
    assert producer.decode_partial_spec((), dtype=torch.bfloat16) is None
    assert producer.decode_partial_spec(records, dtype=torch.float32) is None
    from dataclasses import replace

    assert producer.decode_partial_spec(
        ((0, replace(config, value_dim=256, nope_dim=64)),), dtype=torch.bfloat16
    ) == F.AttentionExchangeSpec(torch.bfloat16, 512, "e")
    cp = object.__new__(C.ContextParallel)
    cp.group = DistributedInfo(0, 4)
    calls = []
    cp._comm = SimpleNamespace(
        _backend=lambda: SimpleNamespace(
            prepare_exchange=lambda spec, enabled: calls.append(spec) or True
        )
    )
    assert cp.prepare_exchange(producer.decode_partial_spec(records, dtype=torch.bfloat16))
    assert len(calls) == 1


class FakeAPI:
    """CPU protocol stand-in, not native execution or numerical qualification."""

    def __init__(self, events):
        self.events = events
        self.Mapping = lambda **kwargs: SimpleNamespace(**kwargs)
        self.MnnvlConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        # Load the real standalone FI backend source, with CUDA forbidden.
        import importlib.util

        source = Path(torch.__file__).parent.parent / "flashinfer/comm/comm_backend.py"
        module_spec = importlib.util.spec_from_file_location("fi_comm_backend_cpu", source)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.CommBackend = module.TorchDistBackend

    def allocate(self, mapping, *, mnnvl_config):
        backend = mnnvl_config.comm_backend
        assert backend.Split(0, mapping.rank) is backend
        with pytest.raises(ValueError, match="partition"):
            backend.Split(1, mapping.rank)
        assert backend.Get_rank() == mapping.rank and backend.Get_size() == mapping.cp_size
        assert tuple(backend.allgather(mapping.rank)) == tuple(range(mapping.cp_size))
        # A nonzero subgroup-local root must become the correct global rank.
        value = backend.bcast(
            mapping.rank if mapping.rank == mapping.cp_size - 1 else None, root=mapping.cp_size - 1
        )
        assert value == mapping.cp_size - 1
        self.events.append(("allocate", vars(mapping)))
        return {"CPU_workspace": True}

    def initialize(self, workspace, rank, size):
        self.events.append(("initialize", rank, size))


def _gloo_worker(rank, world, cp_size, mode, rendezvous):
    import minisgl.distributed.info as info

    torch.set_num_threads(1)
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(
        AssertionError("CPU subgroup initialized CUDA")
    )
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=45),
    )
    info._TP_INFO = DistributedInfo(rank, world)
    configure_parallel(dcp_size=cp_size)
    events = []
    api = FakeAPI(events)
    C.native_candidate = lambda spec, device, **kw: (
        (api, world) if spec.native_layout() and not (mode == "unavailable" and rank == 1) else None
    )
    C.init_context_parallel(backend="gloo", cpu_group=dist.group.WORLD)
    cp = C.ContextParallel()
    owner = DistributedCommunicator.groups["context"]
    owner.device = torch.device("cuda", rank)  # Metadata only; no set_device.
    assert owner.members == tuple(range(rank // cp_size * cp_size, (rank // cp_size + 1) * cp_size))
    spec = F.AttentionExchangeSpec(torch.bfloat16, 512, "e")
    if mode == "unsupported":
        spec = F.AttentionExchangeSpec(torch.float32, 512, "e")
    if mode == "mismatch" and rank == 1:
        spec = F.AttentionExchangeSpec(torch.bfloat16, 128, "e")
    enabled = not (mode == "mixed" and rank == 1)
    # A generic producer exchanges before a native contract is prepared.
    values = torch.full((2, cp_size * 2, 8), float(rank), dtype=torch.float32)
    lse = torch.zeros(2, cp_size * 2)
    actual = cp.exchange(values, lse)
    expected_value = sum(owner.members) / cp_size
    assert actual.dtype == torch.float32 and actual.eq(expected_value).all()
    try:
        if mode == "mismatch":
            with pytest.raises(RuntimeError, match="agreement"):
                cp.prepare_exchange(spec, enabled=enabled)
            assert events == []
        else:
            ready = cp.prepare_exchange(spec, enabled=enabled)
            expected = mode != "unsupported" and not (
                mode in ("mixed", "unavailable") and rank < cp_size
            )
            assert ready == expected
            assert len(events) == (2 if ready else 0)
            if ready:
                mapping = events[0][1]
                assert mapping["world_size"] == cp_size and mapping["rank"] == rank % cp_size
                assert (
                    mapping["gpus_per_node"] == world
                    and mapping["tp_size"] == mapping["pp_size"] == 1
                )
                assert owner.spec == spec and owner.transport.workspace == {"CPU_workspace": True}
                with pytest.raises(RuntimeError, match="once"):
                    cp.prepare_exchange(spec)
            if ready:
                with pytest.raises(ValueError, match="declared spec"):
                    cp.exchange(values, lse)
            else:
                # Collectively declined preparation retains the generic route.
                actual = cp.exchange(values, lse)
                assert actual.dtype == torch.float32 and actual.eq(expected_value).all()
        transport = owner.transport
        owner.close()
        owner.close()
        assert owner.closed and not owner.owns_group
        if transport is not None:
            assert transport.closed and transport.workspace == {"CPU_workspace": True}
        # Borrowed TP Gloo is still usable after the owned CP group closes.
        dist.barrier()
    finally:
        destroy_distributed()
        info._TP_INFO = None
        configure_parallel()
        dist.destroy_process_group()
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in multiprocess gate"
)
@pytest.mark.parametrize(
    "world,cp_size,mode",
    [
        (2, 2, "ready"),
        (2, 2, "mixed"),
        (2, 2, "mismatch"),
        (2, 2, "unavailable"),
        (2, 2, "unsupported"),
        (8, 2, "ready"),
        (8, 4, "ready"),
        (8, 4, "mixed"),
        (8, 8, "ready"),
    ],
)
def test_real_gloo_startup_subgroups_and_fallback(tmp_path, world, cp_size, mode):
    mp.spawn(
        _gloo_worker,
        args=(world, cp_size, mode, str(tmp_path / "rendezvous")),
        nprocs=world,
        join=True,
    )


def test_native_initialization_error_never_enters_another_agreement_or_fallback(monkeypatch):
    calls = []
    group = C._ContextGroup(
        "device", members=[0, 1], tp_members=[0, 1], cpu_group="cpu", device=torch.device("cuda:0")
    )
    monkeypatch.setattr(C, "native_candidate", lambda *a, **kw: ("API", 8))
    monkeypatch.setattr(
        C.dist,
        "all_gather_object",
        lambda rows, local, group: (
            calls.append("agreement"),
            rows.__setitem__(slice(None), [local, local]),
        ),
    )
    monkeypatch.setattr(C.dist, "get_rank", lambda group: 0)

    def fail(*a, **kw):
        calls.append("native_init")
        raise RuntimeError("native init failure")

    monkeypatch.setattr(C, "FlashInferDCPExchange", fail)
    with pytest.raises(RuntimeError, match="native init failure"):
        group.prepare_exchange(F.AttentionExchangeSpec(torch.bfloat16, 512))
    assert calls == ["agreement", "native_init"] and group.transport is None
    with pytest.raises(RuntimeError, match="once"):
        group.prepare_exchange(F.AttentionExchangeSpec(torch.bfloat16, 512))


def test_workspace_close_has_no_private_keepalive_release_or_gpu_operation(monkeypatch):
    transport = object.__new__(F.FlashInferDCPExchange)
    transport.workspace = object()
    saved = transport.workspace
    transport.closed = False
    transport.close()
    assert transport.closed and transport.workspace is saved
    source = Path(F.__file__).read_text()
    assert "_workspace_keepalive" not in source and "close_mnnvl_memory" not in source


@pytest.mark.parametrize("base", ["e", "2"])
def test_nonzero_lse_base_and_native_flag_are_explicit(monkeypatch, base):
    import math

    from minisgl.kernel import dcp

    values = torch.tensor([[[[4.0]]], [[[8.0]]]])
    lse = torch.tensor([[[0.0]], [[math.log(3) if base == "e" else math.log2(3)]]])
    torch.testing.assert_close(
        C.merge_attention(values, lse, lse_base=base), torch.tensor([[[7.0]]])
    )

    class DeviceInput(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    calls = []

    def native(output, stats, *, is_lse_base_on_e):
        calls.append(is_lse_base_on_e)
        return torch.zeros(1, 1, 8), None

    monkeypatch.setattr(dcp, "_native_lse_combine", lambda: native)
    dcp.combine_partial_attention(
        torch.zeros(2, 1, 1, 8).as_subclass(DeviceInput),
        torch.zeros(2, 1, 1).as_subclass(DeviceInput),
        is_lse_base_on_e=base == "e",
    )
    assert calls == [base == "e"]


def test_missing_optional_abi_declines_without_loading_native_workspace(monkeypatch):
    loader = F._native_api
    loader.cache_clear()

    def missing(name):
        raise ImportError("optional DCP ABI absent")

    with monkeypatch.context() as m:
        m.setattr(F.importlib, "import_module", missing)
        assert loader() is None
    loader.cache_clear()


class DeviceInput(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def test_real_transport_pack_layout_uses_only_stats_slot_zero():
    calls = []
    x = torch.arange(2 * 4 * 8).view(2, 4, 8).bfloat16().as_subclass(DeviceInput)
    lse = torch.arange(8).view(2, 4).float()
    saved = x.clone()
    transport = object.__new__(F.FlashInferDCPExchange)
    transport.closed, transport.rank, transport.size, transport.device = False, 0, 2, x.device
    transport.workspace = object()

    def pack(output, stats, dest, dest_stats):
        assert output is x and stats is lse
        dest.copy_(output.view(2, 2, 2, 8).permute(1, 0, 2, 3))
        dest_stats.copy_(stats.view(2, 2, 2).permute(1, 0, 2))
        calls.append("official_pack_ABI")

    def exchange(partial, stats, workspace, rank, size):
        assert partial.shape == (2, 2, 2, 8) and partial.dtype == torch.bfloat16
        assert stats.shape == (2, 2, 2, 2) and stats.dtype == torch.float32
        assert workspace is transport.workspace and (rank, size) == (0, 2)
        stats[..., 1].fill_(torch.nan)
        calls.append("FI_exchange_ABI")
        return partial, stats

    transport.api = SimpleNamespace(pack=pack, exchange=exchange)
    output, stats = transport.exchange(x, lse)
    torch.testing.assert_close(output, x.view(2, 2, 2, 8).permute(1, 0, 2, 3), rtol=0, atol=0)
    torch.testing.assert_close(stats, lse.view(2, 2, 2).permute(1, 0, 2), rtol=0, atol=0)
    assert torch.equal(x, saved) and calls == ["official_pack_ABI", "FI_exchange_ABI"]
    with pytest.raises(ValueError, match="storage contract"):
        transport.exchange(
            torch.empty(2, 4, 16).bfloat16()[:, :, ::2].as_subclass(DeviceInput), lse
        )
    assert len(calls) == 2


def test_native_launch_failure_never_retries_torch_transport():
    calls = []

    def fail(output, lse):
        calls.append("native_launch")
        raise RuntimeError("unknown native DCP failure")

    owner = C._ContextGroup(
        "device", members=[0, 1], tp_members=[0, 1], cpu_group=None, device=None
    )
    owner.spec = F.AttentionExchangeSpec(torch.bfloat16, 8)
    owner.transport = SimpleNamespace(exchange=fail)
    cp = object.__new__(C.ContextParallel)
    cp.group = DistributedInfo(0, 2)
    cp._comm = SimpleNamespace(
        _backend=lambda: owner, all_to_all=lambda x: pytest.fail("native failure retried Torch")
    )
    with pytest.raises(RuntimeError, match="unknown native DCP failure"):
        cp.exchange(torch.ones(2, 4, 8).bfloat16().as_subclass(DeviceInput), torch.zeros(2, 4))
    assert calls == ["native_launch"]


def test_capture_preparation_is_rejected_before_cpu_agreement(monkeypatch):
    owner = C._ContextGroup(
        "device", members=[0, 1], tp_members=[0, 1], cpu_group="cpu", device=torch.device("cuda:0")
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        C.dist,
        "all_gather_object",
        lambda *a, **kw: pytest.fail("capture entered startup collective"),
    )
    with pytest.raises(RuntimeError, match="before graph capture"):
        owner.prepare_exchange(F.AttentionExchangeSpec(torch.bfloat16, 512))
    assert owner.transport is None and not owner.prepared


def test_prepared_local_device_violation_cannot_switch_collective():
    transport = object.__new__(F.FlashInferDCPExchange)
    transport.closed, transport.device = False, torch.device("cuda:0")
    owner = C._ContextGroup(
        "device", members=[0, 1], tp_members=[0, 1], cpu_group=None, device=transport.device
    )
    owner.spec, owner.transport = F.AttentionExchangeSpec(torch.bfloat16, 8), transport
    cp = object.__new__(C.ContextParallel)
    cp.group = DistributedInfo(0, 2)
    cp._comm = SimpleNamespace(
        _backend=lambda: owner, all_to_all=lambda x: pytest.fail("local device switched collective")
    )
    with pytest.raises(ValueError, match="storage contract"):
        cp.exchange(torch.zeros(2, 4, 8).bfloat16(), torch.zeros(2, 4))


@pytest.mark.parametrize("violation", ["dtype", "value_dim", "lse_dtype", "lse_base"])
def test_prepared_semantic_violation_never_switches_collective(violation):
    owner = C._ContextGroup(
        "device", members=[0, 1], tp_members=[0, 1], cpu_group=None, device=None
    )
    owner.spec = F.AttentionExchangeSpec(torch.bfloat16, 8, "e")
    owner.transport = SimpleNamespace(
        exchange=lambda *args: pytest.fail("invalid spec reached native")
    )
    cp = object.__new__(C.ContextParallel)
    cp.group = DistributedInfo(0, 2)
    cp._comm = SimpleNamespace(
        _backend=lambda: owner,
        all_to_all=lambda x: pytest.fail("spec violation switched collective"),
    )
    output, lse, base = torch.zeros(2, 4, 8).bfloat16(), torch.zeros(2, 4), "e"
    if violation == "dtype":
        output = output.float()
    elif violation == "value_dim":
        output = output[..., :4]
    elif violation == "lse_dtype":
        lse = lse.double()
    else:
        base = "2"
    with pytest.raises(ValueError, match="declared spec"):
        cp.exchange(output, lse, lse_base=base)
