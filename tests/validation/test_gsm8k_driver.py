"""Control/lifecycle tests for the validation copy; no model mathematics."""
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = __import__("pathlib").Path(__file__).parent
spec = importlib.util.spec_from_file_location("quality_adapter", HERE / "gsm8k_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    from minisgl.distributed import DistributedInfo
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr(layout, "_DCP_SIZE", 1)
    def forbidden(*args, **kwargs):
        pytest.fail("CPU quality harness test entered CUDA")
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=True))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield
    assert not torch.cuda.is_initialized()


def args(tmp_path, *, engine="mini"):
    cases = tmp_path / "cases.json"
    cases.write_text("{}")
    return SimpleNamespace(
        model="unused", output=tmp_path, cases=cases, engine=engine, tp_size=1,
        dcp_size=1, ep_size=1, attention_backend="auto", distributed_backend="auto",
        distributed_provider=None, mxfp4_backend="flashinfer", moe_backend="auto",
        activation_precision="mxfp8", kv_cache_dtype="fp8_e4m3",
        recurrent_state_dtype="bfloat16", bf16_precision="default", max_context=64,
        batch_size=17, max_running_requests=48, cuda_graph=True, cuda_graph_max_bs=48,
        max_new_tokens=3, flashinfer_autotune_skip_ops=[],
        expected_reference_revision="94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
        sglang_attention_backend=None, sglang_prefill_attention_backend=None,
        sglang_decode_attention_backend=None,
    )


def engine_type(seen, *, event_failure=None):
    from minisgl.core import Context, Req, SamplingParams
    from minisgl.engine.graph import GraphRunner
    from minisgl.engine.sample import Sampler
    from minisgl.kvcache.config import HybridCacheConfig
    from minisgl.kvcache.hybrid_pool import HybridCache

    class ObservedSampler:
        def __init__(self, engine):
            self.engine, self.original, self.calls = engine, Sampler(torch.device("cpu"), 512), 0
        def prepare(self, batch):
            seen.setdefault("prepare_rows", []).append(batch.size)
            return self.original.prepare(batch)
        def sample(self, logits, sampling_args):
            self.calls += 1
            engine = self.engine
            return torch.tensor([
                0 if engine.step == 1 and req.table_idx < 10 else req.table_idx + 20 + engine.step
                for req in engine.current.reqs
            ], dtype=torch.int32)

    class Engine:
        def __init__(self, config):
            self.device = torch.device("cpu")
            self.useful = config.num_page_override * config.logical_page_size
            self.page_table = torch.arange(49 * 64, dtype=torch.int32).reshape(49, 64)
            self.page_table[48].fill_(self.useful)
            cache_config = HybridCacheConfig(1, (), (0,), 4, 2, 1, 2, 2)
            self.kv_cache = HybridCache(cache_config, config.num_page_override + 1,
                config.logical_page_size, 48, torch.float32, "cpu",
                state_slot_capacity=144, state_dtype=torch.float32)
            self.ctx = Context(config.logical_page_size)
            self.ctx.kv_cache = self.kv_cache
            self.ctx.page_table = self.page_table
            runner = object.__new__(GraphRunner)
            runner.max_graph_bs, runner.graph_bs_list = 24, [8, 24]
            runner.graph_map = {8: object(), 24: object()}
            runner.attn_backend = SimpleNamespace(supports_graph_padding=True)
            runner.dummy_req = Req(torch.tensor([0], dtype=torch.int32), 48, 0, 1, -1,
                                   SamplingParams(), None)
            self.graph_runner = runner
            self.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
            self.sampler = ObservedSampler(self)
            self.step, self.closed = 0, False
            seen["engine"] = self
        def forward_batch(self, batch, sampling_args):
            self.current = batch
            if batch.is_decode:
                assert batch.input_ids.numel() == batch.padded_size
                assert batch.positions.numel() == batch.padded_size
                assert batch.out_loc.numel() == batch.padded_size
                assert batch.out_loc[batch.size:].eq(self.useful).all()
                assert batch.positions[batch.size:].eq(0).all()
            seen.setdefault("batches", []).append((batch.size, batch.padded_size, batch.phase))
            with self.ctx.forward_batch(batch):
                assert sum(self.kv_cache.recurrent._active) == batch.size
                self.kv_cache.recurrent.batch_indices([req.table_idx for req in batch.reqs])
                assert all(owner != -1 for owner in self.kv_cache.recurrent._owners)
            for req in batch.reqs:
                req.complete_one()
            tokens = self.sampler.sample(torch.zeros(batch.size, 512), sampling_args)
            step, self.step = self.step, self.step + 1
            class Output:
                def __init__(self):
                    self.waited = False
                    self.copy_done_event = SimpleNamespace(synchronize=self.wait)
                def wait(self):
                    if event_failure is not None and step == 1:
                        raise event_failure
                    self.waited = True
                @property
                def next_tokens_cpu(self):
                    assert self.waited, "CPU IDs accessed before copy completion"
                    seen.setdefault("published_rows", []).append(len(tokens))
                    return tokens
            return Output()
        def shutdown(self):
            self.closed = True
    return Engine


def install_mini(monkeypatch, seen, event_failure=None):
    import minisgl.engine as engine
    import minisgl.utils as utils
    monkeypatch.setattr(engine, "Engine", engine_type(seen, event_failure=event_failure))
    monkeypatch.setattr(utils, "load_tokenizer", lambda model: SimpleNamespace(decode=lambda ids: "#### 1"))
    monkeypatch.setitem(sys.modules, "provenance", SimpleNamespace(execution_identity=lambda name: {"runtime": "CPU_FAKE_ONLY"}))


def test_real_requests_sparse_padding_publication_and_exactly_once(monkeypatch, tmp_path):
    from minisgl.core import Req
    seen, advances = {}, {}
    install_mini(monkeypatch, seen)
    original = Req.complete_one
    def complete(req):
        advances[req.uid] = advances.get(req.uid, 0) + 1
        return original(req)
    monkeypatch.setattr(Req, "complete_one", complete)
    cases = [{"id": i, "gold": "1", "input_ids": [3, 4]} for i in range(17)]
    suite.run_mini(args(tmp_path), cases, {0})
    assert seen["batches"] == [(17, 17, "prefill"), (17, 24, "decode"), (7, 8, "decode")]
    assert seen["prepare_rows"] == seen["published_rows"] == [17, 17, 7]
    engine = seen["engine"]
    assert engine.closed and engine.kv_cache.recurrent.is_idle
    assert engine.sampler.calls == 3 and "sample" not in vars(engine.sampler)
    assert advances == {i + 1: (2 if i < 10 else 3) for i in range(17)}
    rows = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert [len(row["token_ids"]) for row in rows] == [2] * 10 + [3] * 7
    assert len(rows) == 17 and all(row["correct"] for row in rows)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["quality_observer"]["delegates_original_sampler_once"]


def test_copy_event_failure_precedes_CPU_publication_and_preserves_stage(monkeypatch, tmp_path):
    seen, primary = {}, RuntimeError("copy completion failed")
    install_mini(monkeypatch, seen, event_failure=primary)
    cases = [{"id": i, "gold": "1", "input_ids": [3, 4]} for i in range(17)]
    with pytest.raises(RuntimeError) as error:
        suite.run_mini(args(tmp_path), cases, {0})
    assert error.value is primary and seen["published_rows"] == [17]
    assert seen["engine"].closed and "sample" not in vars(seen["engine"].sampler)
    failure = json.loads((tmp_path / "failure-rank0.json").read_text())
    assert failure["stage"] == "token_transfer"
    assert all(len(ids) == 1 for ids in failure["generated_ids"])


def test_finite_observer_delegates_once_and_restores_on_failure():
    class Sampler:
        calls = 0
        def sample(self, logits, sampling_args):
            self.calls += 1
            return torch.tensor([17])
    sampler = Sampler()
    engine = SimpleNamespace(sampler=sampler)
    with suite.finite_sampling(engine):
        assert sampler.sample(torch.zeros(1, 3), None).tolist() == [17]
        with pytest.raises(AssertionError, match="non-finite real logits"):
            sampler.sample(torch.tensor([[float("nan")]]), None)
    assert sampler.calls == 1 and "sample" not in vars(sampler)


def test_official_reference_revision_is_explicit_and_mismatch_precedes_engine(monkeypatch, tmp_path):
    seen = []
    finish = {"type": "stop", "matched": 0}
    class Engine:
        def __init__(self, **kwargs):
            seen.append(kwargs)
        def generate(self, **kwargs):
            return [{"output_ids": [1], "text": "#### 1",
                     "meta_info": {"finish_reason": dict(finish), "completion_tokens": 1}}]
        def shutdown(self):
            seen.append("closed")
    options = args(tmp_path, engine="sglang")
    options.batch_size = 1
    identity = {"revision": options.expected_reference_revision, "tracked_changes": []}
    monkeypatch.setitem(sys.modules, "provenance", SimpleNamespace(execution_identity=lambda name: dict(identity)))
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    suite.run_sglang(options, [{"id": 0, "gold": "1", "input_ids": [3]}], {0})
    assert seen[-1] == "closed" and seen[0]["mamba_ssm_dtype"] == "bfloat16"
    assert seen[0]["flashinfer_mxfp4_moe_precision"] == "default"
    row = json.loads((tmp_path / "responses.jsonl").read_text())
    assert row["native_meta_info"] == {"finish_reason": finish, "completion_tokens": 1}
    finish.update(type="abort", message="fixture cancellation")
    with pytest.raises(AssertionError, match="non-completion"):
        suite.run_sglang(options, [{"id": 1, "gold": "1", "input_ids": [3]}], {0})
    saved = json.loads((tmp_path / "native-terminal-error.json").read_text())
    assert saved["case_ids"] == [1] and saved["results"][0]["meta_info"]["finish_reason"] == finish
    assert json.loads((tmp_path / "failure-rank0.json").read_text())["stage"] == "native_result"
    assert len((tmp_path / "responses.jsonl").read_text().splitlines()) == 1
    identity["revision"] = "baa9befa381511cd4d56f83980951e5a45fffc6d"
    before = len(seen)
    with pytest.raises(RuntimeError, match="pinned clean source"):
        suite.run_sglang(options, [{"id": 0, "gold": "1", "input_ids": [3]}], {0})
    assert len(seen) == before
