"""CPU dispatch/ABI and generic-layer contracts, not native-kernel correctness."""

from types import SimpleNamespace

import minisgl.kernel as kernels
import minisgl.kernel.index as index
import minisgl.layers.embedding as embedding
import pytest
import torch
import torch.nn.functional as F
from test_collective_outputs import cuda_tensor


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU test attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    loader = index._native_vocab_embedding
    loader.cache_clear()
    yield
    loader.cache_clear()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_native_masked_lookup_preserves_exact_tail_shard_arguments(monkeypatch, dtype, index_dtype):
    # Existing CPU-backed tensor subclass supplies CUDA metadata only.
    weights = cuda_tensor(6, 8, dtype=dtype)[::2]
    indices = cuda_tensor(2, dtype=index_dtype)
    result, calls = object(), []

    def native(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(index, "_native_vocab_embedding", lambda: native)
    monkeypatch.setattr(index, "_jit_index_module", lambda *a, **kw: pytest.fail("legacy JIT"))
    assert index.indexing(weights, indices, vocab_range=(8, 2)) is result
    assert calls == [(indices, weights, 8, 10, 0, 0, 0)]


@pytest.mark.parametrize(
    "case",
    [
        "output",
        "unmasked",
        "CPU_weight",
        "CPU_indices",
        "weight_dtype",
        "index_dtype",
        "weight_columns",
        "indices_stride",
        "weight_rank",
        "index_rank",
        "range_length",
    ],
)
def test_other_index_semantics_keep_original_launcher(monkeypatch, case):
    weights, indices = cuda_tensor(4, 8), cuda_tensor(2, dtype=torch.int32)
    output, vocab_range = None, (4, 4)
    if case == "output":
        output = cuda_tensor(2, 8)
    elif case == "unmasked":
        vocab_range = None
    elif case == "CPU_weight":
        weights = torch.ones(4, 8)
    elif case == "CPU_indices":
        indices = torch.ones(2, dtype=torch.int32)
    elif case == "weight_dtype":
        weights = cuda_tensor(4, 8, dtype=torch.int32)
    elif case == "index_dtype":
        indices = cuda_tensor(2, dtype=torch.float32)
    elif case == "weight_columns":
        weights = cuda_tensor(8, 4).T
    elif case == "indices_stride":
        indices = cuda_tensor(4, dtype=torch.int32)[::2]
    elif case == "weight_rank":
        weights = cuda_tensor(2, 4, 8)
    elif case == "index_rank":
        indices = cuda_tensor(2, 1, dtype=torch.int32)
    elif case == "range_length":
        vocab_range = (4, 5)
    calls = []
    monkeypatch.setattr(index, "_native_vocab_embedding", lambda: pytest.fail("SDK import"))
    monkeypatch.setattr(
        index,
        "_jit_index_module",
        lambda *args, **kw: SimpleNamespace(launch=lambda *args: calls.append(args)),
    )
    actual = index.indexing(weights, indices, output=output, vocab_range=vocab_range)
    assert len(calls) == 1
    assert calls[0][0] is weights and calls[0][1] is indices
    assert calls[0][2] is actual and calls[0][3] == vocab_range
    if output is not None:
        assert actual is output


@pytest.mark.parametrize(
    "missing",
    ["sglang", "sglang.kernels", "sglang.kernels.ops.embeddings.vocab_parallel_embedding"],
)
def test_only_absent_native_module_is_optional(monkeypatch, missing):
    def absent(name):
        raise ModuleNotFoundError("optional native module absent", name=missing)

    monkeypatch.setattr(index.importlib, "import_module", absent)
    assert index._native_vocab_embedding() is None


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("internal dependency absent", name="triton"),
        ImportError("selected ABI mismatch"),
        RuntimeError("selected import failure"),
    ],
)
def test_real_dependency_and_ABI_errors_propagate(monkeypatch, error):
    def fail(name):
        raise error

    monkeypatch.setattr(index.importlib, "import_module", fail)
    with pytest.raises(type(error), match=str(error)):
        index._native_vocab_embedding()


def test_missing_function_and_selected_launch_error_do_not_fallback(monkeypatch):
    monkeypatch.setattr(index.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(AttributeError, match="vocab_parallel_embedding"):
        index._native_vocab_embedding()

    def fail(*args):
        raise RuntimeError("native launch failed")

    monkeypatch.setattr(index, "_native_vocab_embedding", lambda: fail)
    monkeypatch.setattr(index, "_jit_index_module", lambda *a, **kw: pytest.fail("fallback"))
    with pytest.raises(RuntimeError, match="native launch failed"):
        index.indexing(cuda_tensor(4, 8), cuda_tensor(2, dtype=torch.int32), vocab_range=(4, 4))


def test_absent_SDK_uses_original_masked_launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(index, "_native_vocab_embedding", lambda: None)
    monkeypatch.setattr(
        index,
        "_jit_index_module",
        lambda *a, **kw: SimpleNamespace(launch=lambda *a: calls.append(a)),
    )
    weights, indices = cuda_tensor(4, 8), cuda_tensor(2, dtype=torch.int64)
    result = index.indexing(weights, indices, vocab_range=(4, 3))
    assert calls == [(weights, indices, result, (4, 3))]


def test_generic_sharded_embedding_and_legacy_layer_share_indexing(monkeypatch):
    calls, reductions = [], []
    result = torch.ones(2, 8)
    tp = SimpleNamespace(rank=1, size=2, all_reduce=lambda x: reductions.append(x) or x)
    monkeypatch.setattr(embedding, "TensorParallel", lambda: tp)
    monkeypatch.setattr(embedding, "get_tp_info", lambda: tp)
    monkeypatch.setattr(embedding, "DistributedCommunicator", lambda: tp)

    def lookup(weights, indices, *, vocab_range):
        calls.append((weights, indices, vocab_range))
        return result

    monkeypatch.setattr(kernels, "indexing", lookup)
    ids = cuda_tensor(2, dtype=torch.int32)
    layer = embedding.Embedding(8, 8)
    assert layer.forward(ids) is result
    legacy = embedding.VocabParallelEmbedding(7, 8)
    assert legacy.forward(ids) is result
    assert calls[0][0] is layer.weight and calls[0][1] is ids and calls[0][2] == (4, 4)
    assert calls[1][0] is legacy.weight and calls[1][1] is ids and calls[1][2] == (4, 3)
    assert reductions == [result, result]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_CPU_shard_values_weight_load_and_host_ID_validation_unchanged(monkeypatch, dtype):
    full = torch.arange(8 * 3, dtype=dtype).reshape(8, 3)
    ids = torch.tensor([0, 3, 4, 7], dtype=torch.int32)
    outputs = []
    monkeypatch.setattr(kernels, "indexing", lambda *a, **kw: pytest.fail("CPU native lookup"))
    for rank in range(2):
        tp = SimpleNamespace(rank=rank, size=2, all_reduce=lambda x: x)
        monkeypatch.setattr(embedding, "TensorParallel", lambda: tp)
        layer = embedding.Embedding(8, 3)
        layer.weight = layer.weight.to(dtype)
        state = {"weight": full[rank * 4 : (rank + 1) * 4].clone()}
        layer.load_state_dict(state)
        assert not state
        outputs.append(layer.forward(ids))
        assert layer.state_dict()["weight"].dtype == dtype
        for invalid in (-1, 8):
            with pytest.raises(IndexError, match="outside the vocabulary"):
                layer.forward(torch.tensor([invalid]))
    torch.testing.assert_close(sum(outputs), F.embedding(ids.long(), full))
    assert torch.equal(ids, torch.tensor([0, 3, 4, 7], dtype=torch.int32))


@pytest.mark.parametrize("vocab,tp_size", [(8, 1), (7, 2)])
def test_unsharded_and_meta_paths_remain_torch_embedding(monkeypatch, vocab, tp_size):
    tp = SimpleNamespace(rank=0, size=tp_size, all_reduce=lambda x: pytest.fail("unsharded TP"))
    monkeypatch.setattr(embedding, "TensorParallel", lambda: tp)
    monkeypatch.setattr(kernels, "indexing", lambda *a, **kw: pytest.fail("unsharded native"))
    layer = embedding.Embedding(vocab, 3)
    layer.weight = torch.arange(vocab * 3, dtype=torch.float32).reshape(vocab, 3)
    ids = torch.tensor([[0, vocab - 1]])
    torch.testing.assert_close(layer.forward(ids), F.embedding(ids, layer.weight))
    with torch.device("meta"):
        meta = embedding.Embedding(vocab, 3)
        result = meta.forward(torch.empty(2, dtype=torch.int64))
    assert result.device.type == "meta" and result.shape == (2, 3)


def test_sharded_meta_path_keeps_original_shape_and_dtype(monkeypatch):
    tp = SimpleNamespace(rank=1, size=2, all_reduce=lambda x: x)
    monkeypatch.setattr(embedding, "TensorParallel", lambda: tp)
    monkeypatch.setattr(kernels, "indexing", lambda *a, **kw: pytest.fail("meta native lookup"))
    with torch.device("meta"):
        layer = embedding.Embedding(8, 3)
        output = layer.forward(torch.empty((2, 2), dtype=torch.int32))
    assert output.shape == (2, 2, 3) and output.device.type == "meta"
    assert output.dtype == layer.weight.dtype
