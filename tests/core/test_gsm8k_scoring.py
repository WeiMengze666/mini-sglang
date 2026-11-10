import json
import runpy
from pathlib import Path

import pytest

answer_number = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))[
    "answer_number"
]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#### 1,000", "1E+3"),
        ("#### 1000.0", "1E+3"),
        ("#### -0.0", "0"),
        ("#### -12.5<|end_of_msg|>", "-12.5"),
        ("#### 1\nCorrection:\n#### 2", "2"),
        ("Working: 17 * 23 = 391", None),
        ("#### 1/2", None),
        ("#### 10-2", None),
        ("#### 5.0.2", None),
        ("#### 42.", "42"),
    ],
)
def test_final_explicit_answer_and_numeric_normalization(text, expected):
    assert answer_number(text) == expected


@pytest.mark.parametrize(
    "engine_kind,failure",
    [
        ("mini", None),
        ("mini", "model"),
        ("mini", "transfer"),
        ("mini", "cleanup"),
        ("sglang", None),
        ("sglang", "model"),
        ("sglang", "cleanup"),
    ],
)
@pytest.mark.parametrize("target", [False, True])
def test_quality_runner_passes_explicit_precision_to_the_real_engine_interface(
    monkeypatch, tmp_path, engine_kind, target, failure, capsys
):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import minisgl.engine as mini_engine
    import minisgl.utils as utils
    import torch

    suite = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))
    seen = {}
    batch_size, capacity = (8, 48) if target else (1, 1)
    primary, cleanup = RuntimeError("primary model failure"), RuntimeError("cleanup failure")
    original_cpu = torch.Tensor.cpu
    transfers = []

    def cpu(tensor, *args, **kwargs):
        transfers.append(1)
        if failure == "transfer" and len(transfers) == 2:
            raise primary
        return original_cpu(tensor, *args, **kwargs)

    if failure == "transfer":
        monkeypatch.setattr(torch.Tensor, "cpu", cpu)

    class Engine:
        def __init__(self, config=None, **kwargs):
            seen.update(config=config, kwargs=kwargs)
            self.device = torch.device("cpu")
            self.page_table = torch.zeros((capacity, 4), dtype=torch.int32)
            self.graph_runner = SimpleNamespace(
                pad_batch=lambda batch: None, can_use_cuda_graph=lambda batch: False
            )
            self.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
            self.ctx = SimpleNamespace(forward_batch=self.forward_batch)
            self.model = SimpleNamespace(forward=self.forward)
            self.kv_cache = SimpleNamespace(release_request=lambda slot, uid: None)

        def forward_batch(self, batch):
            seen["group_size"] = len(batch.reqs)
            return nullcontext()

        def forward(self):
            if failure == "model":
                raise primary
            return torch.tensor([[0.0, 1.0]]).repeat(seen["group_size"], 1)

        def generate(self, **kwargs):
            seen["group_size"] = len(kwargs["input_ids"])
            if failure == "model":
                raise primary
            return [
                {
                    "output_ids": [1],
                    "text": "#### 1",
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }
                for _ in kwargs["input_ids"]
            ]

        def shutdown(self):
            seen["closed"] = True
            if failure is not None:
                raise cleanup

    identity = {"revision": "baa9befa381511cd4d56f83980951e5a45fffc6d", "tracked_changes": []}
    monkeypatch.setitem(
        sys.modules, "provenance", SimpleNamespace(execution_identity=lambda name: dict(identity))
    )
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    monkeypatch.setattr(mini_engine, "Engine", Engine)
    monkeypatch.setattr(
        utils, "load_tokenizer", lambda path: SimpleNamespace(decode=lambda ids: "#### 1")
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8" if target else "1")
    # cuBLAS exposes this flag dynamically. Patch its Python owner instead of
    # asking pytest to restore an attribute absent from the proxy's __dict__.
    matmul = SimpleNamespace(allow_tf32=True)
    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul)
    cases = tmp_path / "cases.json"
    cases.write_text("{}")
    args = SimpleNamespace(
        model="unused",
        output=tmp_path,
        cases=cases,
        engine=engine_kind,
        tp_size=8 if target else 1,
        dcp_size=8 if target else 1,
        ep_size=1,
        attention_backend="auto",
        distributed_backend="mnnvl" if target and engine_kind == "mini" else "auto",
        distributed_provider=(
            "minisgl.distributed.sgl" if target and engine_kind == "mini" else None
        ),
        mxfp4_backend="flashinfer" if target else "torch",
        moe_backend="auto" if target else "marlin",
        activation_precision="mxfp8" if target else "bf16",
        kv_cache_dtype="fp8_e4m3" if target else "bfloat16",
        recurrent_state_dtype="bfloat16" if target else "float32",
        bf16_precision="default" if target else "strict",
        max_context=4,
        batch_size=batch_size,
        max_running_requests=capacity,
        cuda_graph_max_bs=48 if target else 0,
        max_new_tokens=3 if failure == "transfer" else 1,
        cuda_graph=target,
        flashinfer_autotune_skip_ops=[],
        sglang_attention_backend=None,
        sglang_prefill_attention_backend=None,
        sglang_decode_attention_backend=None,
    )

    def run():
        suite["run_" + engine_kind](
            args,
            [{"id": i, "gold": "1", "input_ids": [0]} for i in range(batch_size)],
            {0} if failure == "transfer" else {1},
        )

    if failure is None:
        run()
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["completed"] == report["correct"] == batch_size
        assert "not the canonical online serving benchmark" in report["workload_boundary"]
    else:
        with pytest.raises(RuntimeError) as raised:
            run()
        assert raised.value is (cleanup if failure == "cleanup" else primary)
        if failure in ("model", "transfer"):
            saved = json.loads((tmp_path / "failure-rank0.json").read_text())
            assert saved["case_ids"] == list(range(batch_size)) and not saved["complete"]
            assert "primary model failure" in saved["error"]
            assert "cleanup failure" in capsys.readouterr().err
            if failure == "transfer":
                assert saved["stage"] == "token_transfer"
                assert saved["generated_ids"] == [[1]] * batch_size
                request = saved["requests"][0]
                assert request["cached_len"] == 1 and request["device_len"] == 2
                assert request["input_ids"]["values"] == [0, 1]

    if engine_kind == "mini":
        assert matmul.allow_tf32 is False
        config = seen["config"]
        assert config.kv_cache_dtype == (torch.float8_e4m3fn if target else torch.bfloat16)
        assert config.recurrent_state_dtype == (torch.bfloat16 if target else torch.float32)
        assert config.mxfp4_activation_precision == args.activation_precision
        assert config.moe_backend == args.moe_backend
        assert config.distributed_backend == args.distributed_backend
        assert config.distributed_provider == args.distributed_provider
        assert config.max_running_req == capacity
        assert config.cuda_graph_bs == (None if target else [])
        assert config.cuda_graph_max_bs == args.cuda_graph_max_bs
        assert config.tp_info.size == args.tp_size
        assert config.dcp_size == args.dcp_size and config.ep_size == args.ep_size
    else:
        config = seen["kwargs"]
        assert config["kv_cache_dtype"] == args.kv_cache_dtype
        assert config["mamba_ssm_dtype"] == args.recurrent_state_dtype
        assert config["flashinfer_mxfp4_moe_precision"] == ("default" if target else "bf16")
        assert config["moe_runner_backend"] == args.moe_backend
        assert config["max_running_requests"] == capacity
        assert config["cuda_graph_max_bs_decode"] == args.cuda_graph_max_bs
        assert config["disable_cuda_graph"] is (not target)
        assert config["tp_size"] == args.tp_size and config["dcp_size"] == args.dcp_size
        assert config["ep_size"] == args.ep_size
        assert config["enable_deterministic_inference"] is (not target)
    assert seen["group_size"] == batch_size
    assert seen["closed"]
    assert not torch.cuda.is_initialized()


def test_failure_metadata_never_reads_device_payloads(monkeypatch, tmp_path):
    import hashlib
    from dataclasses import dataclass
    from types import SimpleNamespace

    import torch

    suite = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))
    device_view = torch.empty_strided((2, 3), (8, 1), device="meta")
    indices = torch.tensor([7], dtype=torch.int32)
    history = list(range(200))
    request = SimpleNamespace(
        uid=24,
        table_idx=3,
        cached_len=200,
        device_len=201,
        input_ids=torch.tensor([101, *history], dtype=torch.int32),
    )

    @dataclass
    class Metadata:
        state_indices: torch.Tensor
        workspace: torch.Tensor

    batch = SimpleNamespace(
        reqs=[request],
        phase="decode",
        input_ids=device_view,
        positions=torch.tensor([200]),
        out_loc=torch.arange(5000),
        attn_metadata=Metadata(indices, device_view),
    )
    cases = tmp_path / "cases.json"
    cases.write_text('{"immutable":true}')
    args = SimpleNamespace(output=tmp_path, cases=cases, engine="mini")

    def forbidden(*args, **kwargs):
        raise AssertionError("Failure capture must not transfer a tensor")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    monkeypatch.setenv("RANK", "3")
    target = suite["save_failure"](
        args,
        RuntimeError("device lost"),
        stage="token_transfer",
        group=[{"id": 42}],
        generated=[history],
        batch=batch,
        logits=device_view,
        graph=True,
    )
    saved = json.loads(target.read_text())
    assert target.name == "failure-rank3.json" and not target.with_suffix(".tmp").exists()
    assert saved["generated_ids"] == [history]
    assert saved["requests"][0]["table_idx"] == 3
    assert saved["requests"][0]["input_ids"]["values"] == [101, *history]
    assert saved["batch"]["attn_metadata"]["state_indices"]["values"] == [7]
    assert saved["batch"]["input_ids"]["shape"] == [2, 3]
    assert saved["batch"]["input_ids"]["stride"] == [8, 1]
    assert saved["batch"]["input_ids"]["device"] == "meta"
    assert "values" not in saved["batch"]["input_ids"]
    assert "values" not in saved["batch"]["out_loc"]
    assert saved["cases_sha256"] == hashlib.sha256(cases.read_bytes()).hexdigest()
    assert saved["stage"] == "token_transfer" and saved["graph_replay"]
    original = target.read_bytes()
    suite["save_failure"](args, RuntimeError("later cleanup"), stage="cleanup", group=[])
    assert target.read_bytes() == original


def test_failed_output_directory_cannot_be_reused(monkeypatch, tmp_path):
    import sys

    suite = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))
    evidence = tmp_path / "failure-rank3.json"
    evidence.write_text('{"original_failure":true}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k_suite",
            "--model",
            "unused",
            "--cases",
            "missing-cases.json",
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(FileExistsError, match="fresh output"):
        suite["main"]()
    assert evidence.read_text() == '{"original_failure":true}'


@pytest.mark.parametrize("engine", ["mini", "sglang"])
@pytest.mark.parametrize(
    "extra,capacity,graph_capacity",
    [
        ([], 8, 0),
        (["--cuda-graph"], 8, 8),
        (["--max-running-requests", "48"], 48, 0),
        (
            ["--max-running-requests", "48", "--cuda-graph", "--cuda-graph-max-bs", "48"],
            48,
            48,
        ),
        (
            ["--max-running-requests", "48", "--cuda-graph", "--cuda-graph-max-bs", "4"],
            48,
            4,
        ),
    ],
)
def test_quality_cli_separates_request_groups_from_engine_capacity(
    monkeypatch, tmp_path, engine, extra, capacity, graph_capacity
):
    import sys

    import torch

    suite = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))
    cases = tmp_path / "cases.json"
    saved = {
        "stop_token_ids": [1],
        "examples": [{"id": i, "gold": "1", "input_ids": [0]} for i in range(8)],
    }
    cases.write_text(json.dumps(saved))
    original = cases.read_bytes()
    seen = {}

    def run(args, group, stop_ids):
        seen.update(args=args, group=group, stop_ids=stop_ids)

    monkeypatch.setitem(suite["main"].__globals__, "run_" + engine, run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k_suite",
            "--model",
            "unused",
            "--cases",
            str(cases),
            "--output",
            str(tmp_path / "output"),
            "--engine",
            engine,
            *extra,
        ],
    )
    suite["main"]()
    args = seen["args"]
    assert args.batch_size == len(seen["group"]) == 8
    assert args.max_running_requests == capacity
    assert args.cuda_graph_max_bs == graph_capacity
    assert seen["stop_ids"] == {1}
    assert cases.read_bytes() == original
    assert args.moe_backend == "marlin" and args.activation_precision == "bf16"
    assert args.kv_cache_dtype == "bfloat16" and args.recurrent_state_dtype == "float32"
    assert args.bf16_precision == "strict"
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["auto", "torch", "pynccl", "mnnvl"])
def test_quality_cli_preserves_engine_provider_validation(monkeypatch, tmp_path, backend):
    import sys

    suite = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"stop_token_ids": [1], "examples": [{"input_ids": [0]}]}))
    seen = {}
    monkeypatch.setitem(
        suite["main"].__globals__,
        "run_mini",
        lambda args, group, stop_ids: seen.update(vars(args)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k_suite",
            "--model",
            "unused",
            "--cases",
            str(cases),
            "--output",
            str(tmp_path / "output"),
            "--distributed-backend",
            backend,
            "--distributed-provider",
            "example.provider",
        ],
    )
    suite["main"]()
    assert seen["distributed_backend"] == backend
    assert seen["distributed_provider"] == "example.provider"


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--max-running-requests", "0"], "at least --batch-size"),
        (["--max-running-requests", "-1"], "at least --batch-size"),
        (["--max-running-requests", "7"], "at least --batch-size"),
        (["--cuda-graph", "--cuda-graph-max-bs", "0"], "Graph capacity"),
        (["--cuda-graph", "--cuda-graph-max-bs", "-1"], "Graph capacity"),
        (["--cuda-graph", "--cuda-graph-max-bs", "49"], "Graph capacity"),
        (["--cuda-graph-max-bs", "48"], "requires --cuda-graph"),
        (["--distributed-backend", "mnnvl"], "requires --distributed-provider"),
        (["--engine", "sglang", "--distributed-backend", "torch"], "only to --engine mini"),
        (
            ["--engine", "sglang", "--distributed-provider", "example.provider"],
            "only to --engine mini",
        ),
    ],
)
def test_quality_cli_rejects_invalid_capacity_or_native_communication(
    monkeypatch, tmp_path, extra, message, capsys
):
    import sys

    import torch

    suite = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k_suite",
            "--model",
            "unused",
            "--cases",
            "missing-cases.json",
            "--output",
            str(output),
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as raised:
        suite["main"]()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()
    assert not torch.cuda.is_initialized()
