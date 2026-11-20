"""CPU policy/ABI checks, with no CUDA initialization or native execution."""

import ast
import importlib.util
import os
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.layers import bf16_gemm as gemm
from minisgl.layers import linear


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU policy test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    policy_load, splitk_load = gemm._load_policy, gemm._load_splitk
    policy_load.cache_clear()
    splitk_load.cache_clear()
    yield
    policy_load.cache_clear()
    splitk_load.cache_clear()
    assert not torch.cuda.is_initialized()


def source_oracle():
    # Extract only pure official definitions for an independent CPU oracle.
    # Product code imports these functions/table objects; it copies no policy.
    root = os.environ.get("MINISGL_SGLANG_SOURCE")
    if root:
        package = Path(root) / "python/sglang"
    else:
        spec = importlib.util.find_spec("sglang")
        if spec is None:
            pytest.skip("Optional SGLang SDK source is absent")
        package = Path(next(iter(spec.submodule_search_locations)))
    namespaces = []
    for relative, names in [
        (
            "srt/layers/quantization/unquant.py",
            {"_BF16_SPLITK_TUNED_TACTICS", "use_bf16_splitk_gemm"},
        ),
        (
            "kernels/ops/gemm/cutedsl_bf16_gemm.py",
            {"_K3_TGV_WIN_SHAPES", "use_cutedsl_bf16_gemm"},
        ),
    ]:
        tree = ast.parse((package / relative).read_text())
        nodes = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
            )
        ]
        namespace = {}
        exec(
            compile(ast.Module(nodes, type_ignores=[]), str(package / relative), "exec"), namespace
        )
        assert names <= namespace.keys()
        namespaces.append(SimpleNamespace(**namespace))
    return namespaces


def test_policy_matches_pinned_official_pure_selectors_and_original_table():
    sdk, tgv = source_oracle()
    table = sdk._BF16_SPLITK_TUNED_TACTICS
    policy = gemm.Policy(sdk, tgv, True, lambda: False)
    assert policy.sdk._BF16_SPLITK_TUNED_TACTICS is table
    # Cover every SDK tuned point and the physical canonical/ragged boundaries.
    points = set(table) | {
        (m, n, k)
        for m in [0, 1, 8, 9, 16, 25, 32, 47, 48, 49, 63, 64, 72, 128, 512]
        for n, k in [(7168, 1536), (7168, 3584), (6144, 7168), (2112, 7168), (1024, 4096)]
    }
    for m, n, k in points:
        tgv_selected = tgv.use_cutedsl_bf16_gemm(m, n, k)
        for bias in (False, True):
            for caller_out in (False, True):
                expected = (
                    "splitk"
                    if sdk.use_bf16_splitk_gemm(m, n, k) and not (bias and caller_out)
                    else "tgv"
                    if tgv_selected
                    else None
                )
                assert (
                    policy.select(
                        m, n, k, output_dtype=torch.bfloat16, bias=bias, caller_out=caller_out
                    )
                    == expected
                )
                assert policy.select(
                    m, n, k, output_dtype=torch.float32, bias=bias, caller_out=caller_out
                ) == ("tgv" if tgv_selected and not bias else None)
    assert (
        policy.select(48, 7168, 1536, output_dtype=torch.bfloat16, bias=False, caller_out=False)
        is None
    )
    assert (
        policy.select(47, 7168, 3584, output_dtype=torch.bfloat16, bias=False, caller_out=False)
        == "tgv"
    )
    assert (
        policy.select(48, 7168, 3584, output_dtype=torch.bfloat16, bias=False, caller_out=False)
        is None
    )
    disabled = gemm.Policy(sdk, tgv, False, lambda: False)
    for m, n, k in table:
        assert disabled.select(
            m, n, k, output_dtype=torch.bfloat16, bias=False, caller_out=False
        ) == ("tgv" if tgv.use_cutedsl_bf16_gemm(m, n, k) else None)


class CudaMetadata(torch.Tensor):
    """CPU storage with CUDA metadata: exercise host dispatch, never a kernel."""

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda:0")


def fake(shape, dtype=torch.bfloat16):
    return torch.ones(shape, dtype=dtype).as_subclass(CudaMetadata)


def install_policy(monkeypatch, route="tgv", *, invariant=False):
    calls = []

    def functional(x, weight, bias):
        calls.append(("tgv", x, weight, bias))
        return F.linear(x, weight, bias)

    def destination(x, weight, out, bias):
        calls.append(("tgv_out", x, weight, out, bias))
        out.copy_(F.linear(x.float(), weight.float(), None if bias is None else bias.float()))
        return out

    sdk = SimpleNamespace(
        _BF16_SPLITK_TUNED_TACTICS={(2, 32, 16): (1, 2, 3, 4)},
        use_bf16_splitk_gemm=lambda *shape: route == "splitk",
    )
    tgv = SimpleNamespace(
        use_cutedsl_bf16_gemm=lambda *shape: route == "tgv",
        cutedsl_bf16_gemm=functional,
        cutedsl_bf16_gemm_out=destination,
    )
    policy = gemm.Policy(sdk, tgv, True, lambda: invariant)
    monkeypatch.setattr(gemm, "_load_policy", cache(lambda: policy))
    monkeypatch.setattr(gemm, "_allocate", lambda x, m, n, dtype: fake((m, n), dtype))
    return policy, calls


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_tgv_functional_destination_and_fp32_use_one_policy(monkeypatch, output_dtype):
    _, calls = install_policy(monkeypatch)
    x, weight = fake((2, 16)), fake((32, 16))
    result = gemm.try_bf16_gemm(x, weight, output_dtype=output_dtype)
    assert result.shape == (2, 32) and result.dtype == output_dtype
    assert calls[-1][0] == ("tgv" if output_dtype == torch.bfloat16 else "tgv_out")
    storage = fake((6, 32), output_dtype)
    storage.fill_(-17)
    out = storage[2:4]
    assert gemm.try_bf16_gemm(x, weight, out=out, output_dtype=output_dtype) is out
    assert calls[-1][0] == "tgv_out" and calls[-1][3] is out
    assert storage[:2].eq(-17).all() and storage[4:].eq(-17).all()
    assert out.eq(16).all()


@pytest.mark.parametrize("direct_route,bias", [(True, False), (False, False), (False, True)])
def test_splitk_uses_mature_runners_sdk_table_and_current_call_contract(
    monkeypatch, direct_route, bias
):
    policy, calls = install_policy(monkeypatch, "splitk")
    tactic = object()

    def run_direct(x, transposed_weight, out, pdl, selected):
        calls.append(("direct", transposed_weight, pdl, selected, out))

    def make_tactic(*values):
        assert values == policy.sdk._BF16_SPLITK_TUNED_TACTICS[(2, 32, 16)]
        return tactic

    def run_split(x, transposed_weight, passed_bias, out, pdl, selected):
        calls.append(("split", transposed_weight, pdl, selected, out, passed_bias))

    direct = SimpleNamespace(
        prefer_direct_bf16_gemm_sm100=lambda *shape: direct_route,
        default_tactic=lambda *shape: tactic,
        run_direct_dense=run_direct,
    )
    split = SimpleNamespace(SplitKTactic=make_tactic, run_splitk_dense=run_split)
    monkeypatch.setattr(gemm, "_load_splitk", cache(lambda: (direct, split)))
    x, weight = fake((2, 16)), fake((32, 16))
    b = fake((32,)) if bias else None
    result = gemm.try_bf16_gemm(x, weight, b)
    assert result.shape == (2, 32)
    call = calls[-1]
    assert call[0] == ("direct" if direct_route else "split")
    assert call[1].shape == (16, 32) and call[1].stride() == (1, 16)
    assert call[1].data_ptr() == weight.data_ptr()
    assert call[2] is True and call[3] is tactic
    if bias:
        assert call[5] is b
    else:
        out = fake((2, 32))
        assert gemm.try_bf16_gemm(x, weight, out=out) is out and calls[-1][4] is out


@pytest.mark.parametrize(
    "invalid",
    [
        "dtype",
        "width",
        "stride",
        "nd_stride",
        "pointer",
        "grad",
        "out_shape",
        "out_dtype",
        "overlap",
        "bias",
    ],
)
def test_unsupported_metadata_declines_before_sdk_import(monkeypatch, invalid):
    x, weight, bias, out, dtype = fake((2, 16)), fake((32, 16)), None, None, None
    if invalid == "dtype":
        x = x.float()
    elif invalid == "width":
        weight = fake((32, 8))
    elif invalid == "stride":
        x = fake((2, 32))[:, ::2]
    elif invalid == "nd_stride":
        x = fake((4, 2, 16))[::2]
    elif invalid == "pointer":
        x = fake((33,))[1:].view(2, 16)
    elif invalid == "grad":
        x.requires_grad_()
    elif invalid == "out_shape":
        out = fake((1, 32))
    elif invalid == "out_dtype":
        out, dtype = fake((2, 32)), torch.float32
    elif invalid == "overlap":
        out = weight[:2]
    elif invalid == "bias":
        bias = fake((31,))
    monkeypatch.setattr(gemm, "_load_policy", lambda: pytest.fail("unsupported input imported SDK"))
    assert gemm.try_bf16_gemm(x, weight, bias, out=out, output_dtype=dtype) is None


def test_cpu_keeps_original_torch_without_importing_sdk(monkeypatch):
    monkeypatch.setattr(gemm, "_load_policy", lambda: pytest.fail("portable path imported SDK"))
    x, weight = torch.ones(2, 16, dtype=torch.bfloat16), torch.ones(32, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(linear.linear_bf16(x, weight), F.linear(x, weight), rtol=0, atol=0)


def test_retained_fp32_keeps_original_cast_for_mixed_and_wider_cpu_inputs():
    for input_dtype, weight_dtype in [
        (torch.float32, torch.bfloat16),
        (torch.float64, torch.float64),
    ]:
        x = torch.tensor([[1.0000001, 1.0000002]], dtype=input_dtype)
        weight = torch.tensor([[1.0000003, -1.0000004]], dtype=weight_dtype)
        actual = linear.linear_fp32(x, weight)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, F.linear(x.float(), weight.float()), rtol=0, atol=0)


def test_shared_dispatch_covers_ordinary_out_and_both_merged_precisions(monkeypatch):
    policy, calls = install_policy(monkeypatch)
    op = linear.Linear(16, 32, dtype=torch.bfloat16)
    op.weight = fake((32, 16))
    x = fake((2, 16))
    assert op.forward(x).dtype == torch.bfloat16
    out = fake((2, 32))
    assert op.forward(x, out=out) is out
    plan = linear.MergedLinear(op.weight, (16, 16), ("a", "b"))
    assert plan.forward(x, output_dtype=torch.bfloat16)["a"].dtype == torch.bfloat16
    # Merged FP32 retains its final allocation and uses the out API directly.
    empty = torch.empty
    monkeypatch.setattr(
        linear.torch,
        "empty",
        lambda shape, **kw: empty(shape, dtype=kw["dtype"]).as_subclass(CudaMetadata),
    )
    assert plan.forward(x)["a"].dtype == torch.float32
    assert [call[0] for call in calls] == ["tgv", "tgv_out", "tgv", "tgv_out"]
    assert policy.sdk is not None


@pytest.mark.parametrize("failure", ["import", "missing_abi", "other_arch", "invariant", "ready"])
def test_policy_loader_owns_only_pure_policy_and_declines_missing_dependencies(
    monkeypatch, failure
):
    sdk, tgv = source_oracle()
    sdk._enable_bf16_splitk_gemm = False  # Uninitialized globals are irrelevant.
    sdk._use_cutedsl_bf16_gemm = None
    sdk.Bf16GemmBackend = SimpleNamespace(CUTEDSL=object())
    sdk.should_enable_bf16_splitk_gemm = lambda backend: backend is sdk.Bf16GemmBackend.CUTEDSL
    sdk.get_exec = lambda: pytest.fail("adapter read global execution config")
    tgv.cutedsl_bf16_gemm = lambda *args: pytest.fail("policy test ran native kernel")
    tgv.cutedsl_bf16_gemm_out = tgv.cutedsl_bf16_gemm
    if failure == "missing_abi":
        del tgv.cutedsl_bf16_gemm_out
    modules = {
        "sglang.srt.runtime_context": SimpleNamespace(
            get_platform=lambda: SimpleNamespace(is_sm100=failure != "other_arch")
        ),
        "sglang.srt.layers.quantization.unquant": sdk,
        "minisgl.kernel.gemm.tgv": tgv,
        "sglang.srt.batch_invariant_ops": SimpleNamespace(
            is_batch_invariant_mode_enabled=lambda: failure == "invariant"
        ),
    }

    def load(name):
        if failure == "import":
            raise ImportError("optional dependency unavailable")
        return modules[name]

    monkeypatch.setattr(gemm.importlib, "import_module", load)
    policy = gemm._load_policy()
    if failure in ("invariant", "ready"):
        assert policy.sdk is sdk and policy.tgv is tgv and policy.splitk_enabled
        assert (
            policy.select(1, 7168, 1536, output_dtype=torch.bfloat16, bias=False, caller_out=False)
            == "tgv"
        )
        if failure == "invariant":
            assert gemm.try_bf16_gemm(fake((2, 16)), fake((32, 16))) is None
    else:
        assert policy is None


def test_missing_splitk_declines_before_final_allocation(monkeypatch):
    install_policy(monkeypatch, "splitk")
    monkeypatch.setattr(gemm, "_load_splitk", cache(lambda: None))
    monkeypatch.setattr(gemm, "_allocate", lambda *args: pytest.fail("declined route allocated"))
    assert gemm.try_bf16_gemm(fake((2, 16)), fake((32, 16))) is None


@pytest.mark.parametrize("out", [False, True])
def test_native_runtime_error_propagates_without_torch_retry(monkeypatch, out):
    policy, calls = install_policy(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("native failure must propagate")

    policy.tgv.cutedsl_bf16_gemm = fail
    policy.tgv.cutedsl_bf16_gemm_out = fail
    monkeypatch.setattr(linear.F, "linear", lambda *args, **kwargs: pytest.fail("retried Torch"))
    with pytest.raises(RuntimeError, match="native failure must propagate"):
        linear._linear(fake((2, 16)), fake((32, 16)), None, out=fake((2, 32)) if out else None)
    assert not calls


def test_cold_import_is_rejected_inside_graph_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        gemm.importlib, "import_module", lambda *args: pytest.fail("capture imported SDK")
    )
    with pytest.raises(RuntimeError, match="Warm native BF16 GEMM"):
        gemm.try_bf16_gemm(fake((2, 16)), fake((32, 16)))


@pytest.mark.parametrize("caller_out", [False, True])
def test_torch_fallback_preserves_fp32_functional_and_out_mm_overloads(monkeypatch, caller_out):
    monkeypatch.setattr(gemm, "_load_policy", cache(lambda: None))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    calls = []
    output = fake((2, 32), torch.float32)

    def mm(x, weight, **kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr(torch, "mm", mm)
    result = linear._linear(
        fake((2, 16)),
        fake((32, 16)),
        None,
        out=output if caller_out else None,
        output_dtype=torch.float32,
    )
    assert result.data_ptr() == output.data_ptr()
    assert calls == [dict(out_dtype=torch.float32, **({"out": output} if caller_out else {}))]


@pytest.mark.parametrize("missing", ["dependency", "abi"])
def test_splitk_loader_declines_missing_runner_before_dispatch(monkeypatch, missing):
    def load(name):
        if missing == "dependency":
            raise ImportError("optional runner unavailable")
        return SimpleNamespace()

    monkeypatch.setattr(gemm.importlib, "import_module", load)
    assert gemm._load_splitk() is None


@pytest.mark.parametrize("caller_out", [False, True])
def test_retained_fp32_torch_fallback_adds_bias_after_mm(monkeypatch, caller_out):
    _, native_calls = install_policy(monkeypatch)
    x, weight = fake((2, 16)), fake((32, 16))
    bias = torch.arange(32, dtype=torch.bfloat16).as_subclass(CudaMetadata)
    output = fake((2, 32), torch.float32) if caller_out else None
    mm = torch.mm
    calls = []

    def retained(x, transposed_weight, *, out_dtype, out=None):
        calls.append((out_dtype, out))
        value = mm(x.float(), transposed_weight.float())
        if out is not None:
            out.copy_(value)
            return out
        return value

    monkeypatch.setattr(torch, "mm", retained)
    actual = linear._linear(x, weight, bias, out=output, output_dtype=torch.float32)
    assert actual.dtype == torch.float32
    if caller_out:
        assert actual is output
    assert actual.eq(16 + bias.float()).all()
    assert calls == [(torch.float32, output)]
    assert not native_calls  # FP32 plus bias declines native before dispatch.
