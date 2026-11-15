"""CPU projection/gate/capture resource contracts; no CUDA or native math run."""

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.attention.kda import prefill, sgl
from minisgl.attention.kda.backend import KDAAttentionBackend, KDAMetadata
from minisgl.attention.kda.registry import KDAKernelSpec, KDAResult, get_kda_kernel
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from test_kda_merged_decode_projection import build_layer, payload_for


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU KDA projection test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)


def test_prefill_merge_preserves_gate_through_conv_without_reprojection(monkeypatch):
    layer = build_layer(monkeypatch)
    layer.load_weights(payload_for(layer).items(), device=torch.device("cpu"))
    x = torch.arange(16).reshape(2, 8).bfloat16() / 16
    expected = {name: F.linear(x, op.weight) for name, op in layer._qkvg_projections().items()}
    for op in layer._qkvg_projections().values():
        monkeypatch.setattr(op, "forward", lambda value: pytest.fail("independent wide projection"))
    monkeypatch.setattr(sgl, "project", lambda *a: pytest.fail("old five-tensor prefill path"))
    monkeypatch.setattr(sgl, "_kernels", lambda: (None, None, nullcontext))
    raw_inputs = []

    def conv(raw, *args, **kwargs):
        raw_inputs.append(raw)
        assert kwargs["activation"] == "silu"
        return raw

    monkeypatch.setattr(prefill, "_conv_kernel", lambda: conv)
    pool = RecurrentStatePool(
        (0,),
        2,
        2,
        4,
        4,
        torch.bfloat16,
        "cpu",
        state_slot_capacity=6,
        state_dtype=torch.bfloat16,
    )
    pool.reserve_request(0, 10)
    pool.begin_request(0, 10, 0)
    indices = pool.batch_indices([0])
    metadata = SimpleNamespace(
        lengths=(2,),
        has_initial_states=(False,),
        cu_seqlens_q=torch.tensor([0, 2]),
        kernel_data={},
        checkpoints=None,
    )
    try:
        prepared, gate = prefill.prepare(layer, x, pool, 0, indices, metadata)
        q, k, v, _, _, _, plan, actual_indices = prepared
        assert len(raw_inputs) == 1 and gate.shape == (2, 8)
        for actual, key in zip((q, k, v), ("q", "k", "v")):
            torch.testing.assert_close(actual.reshape(2, 8), expected[key], rtol=0, atol=0)
        torch.testing.assert_close(gate, expected["gate"], rtol=0, atol=0)
        assert actual_indices is plan["indices"]
        before = gate.clone()
        raw_inputs[0].fill_(7)  # Convolution's separate packed storage cannot overwrite the gate.
        assert torch.equal(gate.view(torch.uint8), before.view(torch.uint8))
        assert not hasattr(layer, "_projected_gate")
    finally:
        pool.release_request(0, 10)


@pytest.fixture
def capture_projection(monkeypatch):
    events, active = [], ["main"]
    main = SimpleNamespace(name="main", device=torch.device("cpu"))

    class Stream:
        name = "side"
        device = torch.device("cpu")

        def wait_stream(self, other):
            events.append(("wait", "side", other.name))

    side = Stream()
    main.wait_stream = lambda other: events.append(("wait", "main", other.name))

    class Input:
        dtype = torch.bfloat16
        device = torch.device("cpu")
        is_cuda = True
        ndim = 2

        def __init__(self, rows):
            self.shape = (rows, 8)

        def __len__(self):
            return self.shape[0]

        def record_stream(self, stream):
            events.append(("record-input", stream.name))

    class Result(torch.Tensor):
        def record_stream(self, stream):
            events.append(("record-result", stream.name))

    @contextmanager
    def use(stream):
        previous, active[0] = active[0], stream.name
        events.append(("enter", stream.name))
        try:
            yield
        finally:
            events.append(("exit", stream.name))
            active[0] = previous

    def layer_for(x):
        wide = {key: torch.ones(len(x), 8, dtype=torch.bfloat16) for key in ("q", "k", "v", "gate")}
        layer = SimpleNamespace(
            _qkvg=SimpleNamespace(weight=torch.ones(32, 8, dtype=torch.bfloat16)),
            _full_gate=True,
            _heads=2,
            _dim=4,
            f_b_proj=SimpleNamespace(weight=torch.ones(8, 4, dtype=torch.bfloat16)),
        )

        def project(actual):
            assert actual is x
            events.append(("wide", active[0]))
            return wide

        layer.project_qkvg = project
        return layer

    def gemm(x, weight):
        events.append(("tiny", active[0], weight.shape[0]))
        return torch.ones(len(x), weight.shape[0], dtype=torch.bfloat16).as_subclass(Result)

    monkeypatch.setattr(
        sgl,
        "_projection_plan",
        lambda layer: SimpleNamespace(weight=torch.ones(8, 8, dtype=torch.bfloat16), widths=(4, 2)),
    )
    monkeypatch.setattr(sgl, "_projection_kernels", lambda: (gemm, nullcontext))
    monkeypatch.setattr(sgl, "_projection_overlap_limit", lambda device: 128)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: main)
    monkeypatch.setattr(torch.cuda, "stream", use)
    return events, Input, layer_for, side


@pytest.mark.parametrize(
    "capturing,rows,limit,overlap",
    [
        (False, 48, 128, False),
        (True, 48, 128, True),
        (True, 129, 128, False),
        (True, 64, 64, True),
        (True, 65, 64, False),
    ],
)
def test_actual_capture_and_official_size_choose_fork_before_wide(
    capture_projection, monkeypatch, capturing, rows, limit, overlap
):
    events, Input, layer_for, side = capture_projection
    monkeypatch.setattr(sgl, "_projection_overlap_limit", lambda device: limit)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    x = Input(rows)
    projected, gate = sgl._project_with_gate(layer_for(x), x, projection_stream=side)
    if overlap:
        assert events == [
            ("wait", "side", "main"),
            ("record-input", "side"),
            ("wide", "main"),
            ("enter", "side"),
            ("tiny", "side", 8),
            ("tiny", "side", 8),
            ("exit", "side"),
            ("wait", "main", "side"),
            ("record-result", "main"),
            ("record-result", "main"),
        ]
    else:
        assert events == [("wide", "main"), ("tiny", "main", 8), ("tiny", "main", 8)]
    assert len(projected) == 5 and gate.shape == (rows, 8)


def test_wrong_selected_stream_device_raises_before_any_projection(capture_projection, monkeypatch):
    events, Input, layer_for, side = capture_projection
    side.device = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    x = Input(4)
    with pytest.raises(ValueError, match="stream differs"):
        sgl._project_with_gate(layer_for(x), x, projection_stream=side)
    assert events == []


@pytest.mark.parametrize(
    "is_cuda,project,sizes,expected",
    [
        (False, True, [1], 0),
        (True, False, [1], 0),
        (True, True, [], 0),
        (True, True, [1, 8], 1),
    ],
)
def test_backend_owns_one_stream_and_capture_metadata_only_borrows(
    monkeypatch, is_cuda, project, sizes, expected
):
    made = []
    resource = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kw: made.append(kw) or resource)
    cache = SimpleNamespace(
        device=torch.device("cuda:0" if is_cuda else "cpu"),
        recurrent=SimpleNamespace(conv_dtype=torch.bfloat16, is_idle=True),
    )
    spec = KDAKernelSpec(
        "projection-resource",
        get_kda_kernel("torch").recurrence,
        project=(lambda layer, x: ()) if project else None,
    )
    backend = KDAAttentionBackend(
        SimpleNamespace(cache_config=SimpleNamespace(head_dim=128)), spec, cache=cache
    )
    backend.init_capture_graph(64, sizes)
    backend.init_capture_graph(64, sizes)
    assert len(made) == expected
    if sizes:
        monkeypatch.setattr(
            backend,
            "prepare_metadata",
            lambda batch: setattr(batch, "attn_metadata", KDAMetadata(torch.tensor([0, 1]))),
        )
        batch = SimpleNamespace(size=1, reqs=[])
        backend.prepare_for_capture(batch)
        assert batch.attn_metadata.projection_stream is (resource if expected else None)
        backend.finish_capture(batch)
        assert backend._projection_stream is (resource if expected else None)


@pytest.mark.parametrize("structured", [False, True])
def test_prefill_result_gate_reaches_finish_exactly_once_and_legacy_tensor_remains_valid(
    structured,
):
    gate, output, seen = torch.ones(2, 8), torch.zeros(2, 2, 4), []
    result = KDAResult(output, gate=gate) if structured else output
    spec = KDAKernelSpec(
        "prefill-output",
        get_kda_kernel("torch").recurrence,
        indexed_prefill=lambda *args: result,
    )
    cache = SimpleNamespace(recurrent=object())
    backend = KDAAttentionBackend(SimpleNamespace(), spec, cache=cache)
    metadata = KDAMetadata(torch.tensor([0, 2]), state_indices=torch.tensor([0]))
    batch = SimpleNamespace(
        is_decode=False, reqs=[SimpleNamespace(extend_len=2)], attn_metadata=metadata
    )

    def finish(x, actual, *, normed=False, gate=None, residual=None):
        assert actual is output and gate is (result.gate if structured else None)
        seen.append((gate, normed))
        return actual

    actual = backend.forward_layer(
        SimpleNamespace(finish=finish), 0, torch.zeros(2, 8), batch, cache, None
    )
    assert actual is output and len(seen) == 1 and seen[0][1] is False


@pytest.mark.parametrize("blackwell,expected", [(False, 64), (True, 128)])
def test_projection_policy_uses_official_pure_hardware_predicate(monkeypatch, blackwell, expected):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils.common",
        SimpleNamespace(is_blackwell_supported=lambda: blackwell),
    )
    sgl._projection_overlap_limit.cache_clear()
    try:
        assert sgl._projection_overlap_limit(torch.device("cuda:0")) == expected
    finally:
        sgl._projection_overlap_limit.cache_clear()
