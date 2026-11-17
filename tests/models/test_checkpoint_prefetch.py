"""CPU-only checkpoint staging: official byte reader, no model or GPU kernel."""

import builtins
import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.models import checkpoint
from safetensors.torch import save_file


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint prefetch CPU test entered CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert not torch.cuda.is_initialized()
    assert not any(t.name.startswith("checkpoint-prefetch") for t in threading.enumerate())


@pytest.fixture
def sdk():
    return importlib.import_module("sglang.srt.model_loader.weight_utils")


def tiny_checkpoint(tmp_path, count=2):
    index = {}
    for i in range(count):
        path = tmp_path / f"shard-{i:02d}.safetensors"
        save_file({f"tensor{i}": torch.arange(64, dtype=torch.int32).reshape(4, 16)}, path)
        index[f"tensor{i}"] = path
    return index


def test_actual_official_reader_preserves_tensors_and_bounds_reads(tmp_path, sdk, monkeypatch):
    index = tiny_checkpoint(tmp_path)
    reads = []

    class Reader:
        def __init__(self, path, mode):
            assert Path(path) in index.values() and mode == "rb"
            self.file = builtins.open(path, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def read(self, size):
            data = self.file.read(size)
            reads.append((size, len(data)))
            return data

    monkeypatch.setattr(sdk, "open", Reader, raising=False)

    def collect(**kwargs):
        return {
            name: reader.get_tensor(name).clone()
            for name, reader in checkpoint.shard_readers(index, **kwargs)
        }

    plain = collect()
    staged = collect(prefetch=True, prefetch_num_threads=2, tp_info=DistributedInfo(0, 1))
    assert plain.keys() == staged.keys()
    assert all(torch.equal(plain[name], staged[name]) for name in plain)
    assert len(reads) == 4
    assert all(size == 16 << 20 and 0 <= returned <= size for size, returned in reads)


def test_disabled_path_does_not_import_prefetch_sdk(tmp_path, monkeypatch):
    index = tiny_checkpoint(tmp_path)
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert name != "sglang.srt.model_loader.weight_utils"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    assert len(list(checkpoint.shard_readers(index))) == 2


def test_sorted_unique_rank_assignment_and_more_ranks_than_shards(tmp_path, sdk, monkeypatch):
    files = list(tiny_checkpoint(tmp_path, count=6).values())
    assigned = {}
    for rank in range(8):
        seen = []
        monkeypatch.setattr(
            sdk, "_prefetch_checkpoint_file", lambda file, cancel: seen.append(file)
        )
        with checkpoint._prefetch_shards(files[::-1] + files[:2], 2, DistributedInfo(rank, 8)):
            pass
        assigned[rank] = sorted(seen)
        assert assigned[rank] == [str(p) for p in sorted(files)[rank::8]]
    assert sorted(p for paths in assigned.values() for p in paths) == sorted(map(str, files))


def test_expert_filtered_index_keeps_global_prefetch_assignment(tmp_path, sdk, monkeypatch):
    index = tiny_checkpoint(tmp_path, count=6)
    paths = list(index.values())
    seen = []
    monkeypatch.setattr(sdk, "_prefetch_checkpoint_file", lambda file, cancel: seen.append(file))
    rows = list(
        checkpoint.shard_readers(
            {"tensor0": paths[0]},
            prefetch=True,
            prefetch_num_threads=2,
            tp_info=DistributedInfo(1, 2),
            prefetch_files=paths,
        )
    )
    assert [name for name, _ in rows] == ["tensor0"]
    assert sorted(seen) == [str(p) for p in paths[1::2]]


def test_concurrent_readers_are_bounded_and_joined(tmp_path, sdk, monkeypatch):
    paths = list(tiny_checkpoint(tmp_path, count=7).values())
    lock, gate, entered = threading.Lock(), threading.Event(), threading.Event()
    counts = {"active": 0, "maximum": 0, "completed": 0}

    def read(path, cancel):
        with lock:
            counts["active"] += 1
            counts["maximum"] = max(counts["maximum"], counts["active"])
            if counts["active"] == 2:
                entered.set()
        try:
            assert gate.wait(3), "CPU fixture gate timed out"
        finally:
            with lock:
                counts["active"] -= 1
                counts["completed"] += 1

    monkeypatch.setattr(sdk, "_prefetch_checkpoint_file", read)
    try:
        with checkpoint._prefetch_shards(paths, 2, DistributedInfo(0, 1)):
            assert entered.wait(3)
            assert counts["maximum"] == 2
            gate.set()
    finally:
        gate.set()
    assert counts == {"active": 0, "maximum": 2, "completed": 7}


def test_generator_close_cancels_and_joins_workers(tmp_path, sdk, monkeypatch):
    index = tiny_checkpoint(tmp_path, count=5)
    entered, stopped = threading.Event(), threading.Event()

    def read(path, cancel):
        entered.set()
        assert cancel.wait(3), "CPU fixture cancellation timed out"
        stopped.set()

    monkeypatch.setattr(sdk, "_prefetch_checkpoint_file", read)
    stream = checkpoint.shard_readers(
        index, prefetch=True, prefetch_num_threads=2, tp_info=DistributedInfo(0, 1)
    )
    next(stream)
    assert entered.wait(3)
    stream.close()
    assert stopped.is_set()


def test_consumer_failure_preserves_primary_and_joins_workers(tmp_path, sdk, monkeypatch):
    index = tiny_checkpoint(tmp_path)
    entered, stopped = threading.Event(), threading.Event()
    primary = RuntimeError("reader fixture failure")

    def read(path, cancel):
        entered.set()
        assert cancel.wait(3)
        stopped.set()

    def fail_open(*args, **kwargs):
        assert entered.wait(3)
        raise primary

    monkeypatch.setattr(sdk, "_prefetch_checkpoint_file", read)
    monkeypatch.setattr(checkpoint.safetensors, "safe_open", fail_open)
    with pytest.raises(RuntimeError) as result:
        list(checkpoint.shard_readers(index, prefetch=True, tp_info=DistributedInfo(0, 1)))
    assert result.value is primary and stopped.is_set()


def test_prefetch_failure_surfaces_after_join(tmp_path, sdk, monkeypatch):
    paths = list(tiny_checkpoint(tmp_path).values())
    primary = OSError("prefetch fixture failure")

    def fail(path, cancel):
        raise primary

    monkeypatch.setattr(sdk, "_prefetch_checkpoint_file", fail)
    with pytest.raises(OSError) as result:
        with checkpoint._prefetch_shards(paths, 2, DistributedInfo(0, 1)):
            pass
    assert result.value is primary


@pytest.mark.parametrize(
    "threads,info", [(0, DistributedInfo(0, 1)), (-1, DistributedInfo(0, 1)), (1, None)]
)
def test_invalid_prefetch_options_fail_before_reader(tmp_path, threads, info):
    with pytest.raises(ValueError):
        with checkpoint._prefetch_shards([tmp_path / "unused"], threads, info):
            pytest.fail("Invalid prefetch options were accepted")


def test_same_official_flag_names_defaults_and_validation():
    from minisgl.server.args import parse_args

    base = ["--model-path", "unused", "--dtype", "bfloat16"]
    plain, _ = parse_args(base)
    assert not plain.weight_loader_prefetch_checkpoints
    assert plain.weight_loader_prefetch_num_threads == 4
    staged, _ = parse_args(
        base + ["--weight-loader-prefetch-checkpoints", "--weight-loader-prefetch-num-threads", "8"]
    )
    assert staged.weight_loader_prefetch_checkpoints
    assert staged.weight_loader_prefetch_num_threads == 8
    with pytest.raises(SystemExit):
        parse_args(base + ["--weight-loader-prefetch-num-threads", "0"])


def test_both_engine_weight_paths_forward_prefetch_options(monkeypatch):
    import minisgl.engine.engine as module

    expected = {"weight": torch.ones(2)}
    calls = []

    def load(path, device, **kwargs):
        calls.append((path, device, kwargs))
        yield "weight", torch.ones(2)

    monkeypatch.setattr(module, "load_weight", load)
    fake = SimpleNamespace(
        model=SimpleNamespace(state_dict=lambda: expected), device=torch.device("cpu")
    )
    config = SimpleNamespace(
        model_path="unused",
        use_dummy_weight=False,
        weight_loader_prefetch_checkpoints=True,
        weight_loader_prefetch_num_threads=8,
    )
    assert len(list(module.Engine._iter_weights(fake, config))) == 1
    assert set(module.Engine._load_weight_state_dict(fake, config)) == {"weight"}
    assert len(calls) == 2 and all(
        c[2] == {"prefetch": True, "prefetch_num_threads": 8} for c in calls
    )


def test_enabled_cross_shard_packed_scales_and_column_slices(tmp_path, sdk):
    import safetensors
    from minisgl.distributed.shard import ShardSpec
    from minisgl.layers.weight import read_mxfp4_pair
    from minisgl.models.kimi_weight import load_kimi_weight

    prefix = "model.layers.0.mlp.experts.0.w2"
    weight = torch.arange(64, dtype=torch.uint8).reshape(2, 32)
    scales = torch.tensor([[126, 127], [128, 129]], dtype=torch.uint8)
    weight_path, scale_path = tmp_path / "packed.safetensors", tmp_path / "scales.safetensors"
    save_file({prefix + ".weight_packed": weight}, weight_path)
    save_file({prefix + ".weight_scale": scales}, scale_path)
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    plain = dict(load_kimi_weight(str(tmp_path), "cpu", packed=True))
    staged = dict(
        load_kimi_weight(str(tmp_path), "cpu", packed=True, prefetch=True, prefetch_num_threads=2)
    )
    assert plain.keys() == staged.keys()
    assert all(torch.equal(plain[name], staged[name]) for name in plain)
    index = {prefix + ".weight_packed": weight_path, prefix + ".weight_scale": scale_path}
    for name, reader in checkpoint.shard_readers(
        index, prefetch=True, prefetch_num_threads=2, tp_info=DistributedInfo(0, 1)
    ):
        if name.endswith(".weight_packed"):
            with safetensors.safe_open(scale_path, framework="pt", device="cpu") as other:
                actual_weight, actual_scale = read_mxfp4_pair(
                    reader,
                    name,
                    other,
                    prefix + ".weight_scale",
                    ShardSpec(1, 32, 64),
                    keep_packed=True,
                )
            assert torch.equal(actual_weight, weight[:, 16:32])
            assert torch.equal(actual_scale, scales[:, 1:2])


@pytest.mark.parametrize("enabled", [False, True])
def test_prefetch_preserves_sorted_shard_and_tensor_iteration(tmp_path, sdk, enabled):
    first, last = tmp_path / "a.safetensors", tmp_path / "z.safetensors"
    save_file({"a.second": torch.ones(2), "a.first": torch.zeros(2)}, first)
    save_file({"z.second": torch.ones(2), "z.first": torch.zeros(2)}, last)
    index = {"z.second": last, "a.second": first, "z.first": last, "a.first": first}
    names = [
        name
        for name, _ in checkpoint.shard_readers(
            index, prefetch=enabled, tp_info=DistributedInfo(0, 1)
        )
    ]
    assert names == ["a.first", "a.second", "z.first", "z.second"]
