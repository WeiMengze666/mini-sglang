"""CPU arithmetic, ownership and routing checks for generic merged KDA projection."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda.registry import KDAProjection, get_kda_kernel
from minisgl.layers.kda import KDALayer
from minisgl.layers.linear import Linear, MergedLinear
from minisgl.utils import torch_dtype


def dyadic(shape, dtype=torch.float32, shift=0):
    return ((torch.arange(torch.tensor(shape).prod().item()).reshape(shape) + shift) % 9 - 4).to(
        dtype
    ) / 16


def layer(dtype=torch.float32, full_gate=True):
    with torch_dtype(dtype):
        op = KDALayer(KDAConfig(8, 2, 4, 3, full_rank_gate=full_gate, gate_lower_bound=-5))
    state = {
        name: dyadic(value.shape, value.dtype, index)
        for index, (name, value) in enumerate(op.state_dict().items())
    }
    op.load_state_dict(state)
    assert not state
    return op


@pytest.mark.parametrize("rows", [1, 8, 12, 48, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merged_wide_views_match_independent_dyadic_dot_and_one_dispatch(monkeypatch, rows, dtype):
    import torch.nn.functional as functional

    op = layer(dtype)
    expected_weights = [
        getattr(op, name).weight.double().clone()
        for name in ("q_proj", "k_proj", "v_proj", "g_proj")
    ]
    x = dyadic((rows, 16), dtype)[:, ::2]
    assert not x.is_contiguous()
    calls, original = [], functional.linear

    def run(value, weight, bias=None):
        calls.append((weight.shape, value.dtype))
        return original(value, weight, bias)

    monkeypatch.setattr(functional, "linear", run)
    values = op.project_qkv(x)
    assert calls == [(torch.Size((32, 8)), dtype)]
    for value, weight in zip(values, expected_weights):
        reference = (x.double() @ weight.T).to(dtype)
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert value.stride() == (32, 1)
        assert value.is_contiguous() == (rows == 1)
        assert value.untyped_storage()._cdata == values[0].untyped_storage()._cdata
    weights = [getattr(op, name).weight for name in ("q_proj", "k_proj", "v_proj", "g_proj")]
    assert all(
        w.untyped_storage()._cdata == op._qkvg.weight.untyped_storage()._cdata for w in weights
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merged_output_dtype_is_explicit_and_default_front_stays_fp32(dtype):
    with torch_dtype(dtype):
        ops = {"left": Linear(8, 3), "right": Linear(8, 5)}
    for i, op in enumerate(ops.values()):
        op.weight.copy_(dyadic(op.weight.shape, dtype, i))
    plan = MergedLinear.try_create(ops)
    x = dyadic((8, 8), dtype)
    default = plan.forward(x)
    explicit = plan.forward(x, output_dtype=dtype)
    assert all(value.dtype == torch.float32 for value in default.values())
    assert all(value.dtype == dtype for value in explicit.values())
    for name, op in ops.items():
        reference = x.double() @ op.weight.double().T
        torch.testing.assert_close(default[name], reference.float(), rtol=0, atol=0)
        torch.testing.assert_close(explicit[name], reference.to(dtype), rtol=0, atol=0)
    with pytest.raises(ValueError, match="input dtype or float32"):
        plan.forward(x, output_dtype=torch.int32)


@pytest.mark.parametrize("partial", [False, True])
def test_after_load_rebuild_releases_old_plan_and_all_weight_views(partial):
    op = layer()
    old_weight = weakref.ref(op._qkvg.weight)
    old_storage = op._qkvg.weight.untyped_storage()._cdata
    if partial:
        expected = op.q_proj.weight.clone() + 1 / 16
        op.q_proj.weight = expected
        op._kernel_plans["stale"] = object()
        op.after_load()
    else:
        state = {name: value.clone() + 1 / 16 for name, value in op.state_dict().items()}
        expected = state["q_proj.weight"].clone()
        op.load_state_dict(state)
        assert not state
    gc.collect()
    assert old_weight() is None
    assert not op._kernel_plans
    assert all(
        getattr(op, name).weight.untyped_storage()._cdata != old_storage
        for name in ("q_proj", "k_proj", "v_proj", "g_proj")
    )
    torch.testing.assert_close(op.q_proj.weight, expected, rtol=0, atol=0)
    actual = op.project_qkv(dyadic((8, 8)))[0]
    torch.testing.assert_close(actual, dyadic((8, 8)) @ expected.T, rtol=0, atol=0)


@pytest.mark.parametrize(
    "unsupported", ["nonfull", "layout", "quantized", "bias", "missing_weight"]
)
def test_unsupported_projection_keeps_original_linear_calls(unsupported):
    op = layer(full_gate=unsupported != "nonfull")
    if unsupported == "layout":
        op.q_proj.weight = op.q_proj.weight.T
    elif unsupported == "bias":
        op.q_proj.bias = torch.zeros(8)
    elif unsupported in ("quantized", "missing_weight"):
        weight = op.q_proj.weight.clone()
        fields = {
            "weight_format": "mxfp4",
            "bias": None,
            "forward": lambda x: torch.nn.functional.linear(x, weight),
        }
        if unsupported == "quantized":
            fields["weight"] = weight
        op.q_proj = SimpleNamespace(**fields)
    op.after_load()
    assert op._qkvg is None
    x = dyadic((8, 8))
    projected = op.project(x)
    assert isinstance(projected, KDAProjection) and projected.gate is None
    for actual, name in zip(projected[:3], ("q_proj", "k_proj", "v_proj")):
        torch.testing.assert_close(actual, getattr(op, name).forward(x), rtol=0, atol=0)


def test_explicit_gate_is_per_forward_and_finish_does_not_reproject(monkeypatch):
    import minisgl.layers.kda as kda

    op = layer()
    x = dyadic((8, 8))
    first, second = op.project(x), op.project(x + 1 / 16)
    assert first.gate is not None and second.gate is not None
    assert first.gate.data_ptr() != second.gate.data_ptr()
    assert not torch.equal(first.gate, second.gate)
    monkeypatch.setattr(op.g_proj, "forward", lambda x: pytest.fail("explicit gate reprojected"))
    seen = []

    def norm(value, gate, weight, eps):
        seen.append(gate)
        return value

    monkeypatch.setattr(kda, "gated_rms_norm", norm)
    y = torch.zeros(8, 2, 4)
    op.finish(x, y, gate=first.gate)
    op.finish(x, y, gate=second.gate)
    assert (
        seen[0].data_ptr() == first.gate.data_ptr() and seen[1].data_ptr() == second.gate.data_ptr()
    )
    assert not any(
        isinstance(value, torch.Tensor) and value.data_ptr() == first.gate.data_ptr()
        for value in vars(op).values()
    )


@pytest.mark.parametrize("backend", ["torch", "flashinfer", "sgl"])
def test_functional_path_carries_explicit_gate_to_finish_without_sdk(monkeypatch, backend):
    import minisgl.layers.kda as kda

    op = layer()
    x = dyadic((7, 8))
    monkeypatch.setattr(op.g_proj, "forward", lambda x: pytest.fail("functional gate reprojected"))
    seen = []
    original = kda.gated_rms_norm

    def norm(output, gate, weight, eps):
        seen.append(gate)
        return original(output, gate, weight, eps)

    monkeypatch.setattr(kda, "gated_rms_norm", norm)
    out, state = op.forward(x, kernel=get_kda_kernel(backend))
    assert out.shape == x.shape and state.recurrent.shape == (2, 4, 4)
    assert len(seen) == 1 and seen[0].shape == (7, 2, 4)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("lengths,is_decode", [((3, 4), False), ((1, 1), False), ((1, 1), True)])
def test_real_backend_preserves_per_request_state_and_explicit_gate(
    monkeypatch, lengths, is_decode
):
    from minisgl.attention.kda.backend import KDAAttentionBackend
    from minisgl.kvcache.recurrent_pool import RecurrentStatePool

    actual, reference = layer(), layer()
    reference.load_state_dict({name: value.clone() for name, value in actual.state_dict().items()})
    reference._qkvg = None
    states = RecurrentStatePool((0,), 2, 2, 4, 3, torch.float32, "cpu")
    cache = SimpleNamespace(
        recurrent=states,
        device=torch.device("cpu"),
        get_kda_state=states.get_kda_state,
        set_kda_state=states.set_kda_state,
    )
    reqs = [
        SimpleNamespace(table_idx=i, uid=i + 10, cached_len=0, extend_len=n, device_len=n)
        for i, n in enumerate(lengths)
    ]
    for req in reqs:
        states.begin_request(req.table_idx, req.uid, 0)
    batch = SimpleNamespace(reqs=reqs, is_decode=is_decode)
    kernel = get_kda_kernel("torch")
    backend = KDAAttentionBackend(SimpleNamespace(), kernel, cache=cache)
    backend.prepare_metadata(batch)
    x = dyadic((sum(lengths), 8))
    expected, expected_states, offset = [], [], 0
    for n in lengths:
        result, state = reference.forward(x[offset : offset + n], kernel=kernel)
        expected.append(result)
        expected_states.append(state)
        offset += n
    monkeypatch.setattr(actual.g_proj, "forward", lambda x: pytest.fail("backend gate reprojected"))
    output = backend.forward_layer(actual, 0, x, batch, cache, None)
    torch.testing.assert_close(output, torch.cat(expected), rtol=1e-5, atol=1e-6)
    for req, expected_state in zip(reqs, expected_states):
        state = states.get_kda_state(0, req.table_idx)
        torch.testing.assert_close(state.conv, expected_state.conv, rtol=0, atol=0)
        torch.testing.assert_close(state.recurrent, expected_state.recurrent, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float32),
        (torch.float64, torch.bfloat16),
    ],
)
def test_default_fp32_merged_projection_preserves_mixed_input_weight_dtypes(
    input_dtype, weight_dtype
):
    with torch_dtype(weight_dtype):
        projections = {"q": Linear(8, 3), "kv": Linear(8, 5)}
    for i, projection in enumerate(projections.values()):
        projection.weight.copy_(dyadic(projection.weight.shape, weight_dtype, i))
    plan = MergedLinear.try_create(projections)
    x = dyadic((8, 8), input_dtype)
    actual = plan.forward(x)
    for name, projection in projections.items():
        reference = torch.nn.functional.linear(x.float(), projection.weight.float())
        assert actual[name].dtype == torch.float32
        torch.testing.assert_close(actual[name], reference, rtol=0, atol=0)
