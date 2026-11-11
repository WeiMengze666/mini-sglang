"""Canonical server arguments and native shared-prefill budget semantics."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.scheduler import SchedulerConfig
from minisgl.scheduler.prefill import PrefillAdder
from minisgl.server.args import parse_args


def parse(*options):
    return parse_args(["--model-path", "unused", "--dtype", "bfloat16", *options])[0]


def test_canonical_arguments_are_independent_and_keep_storage_precision():
    config = parse(
        "--context-length", "1048576", "--mem-fraction-static", ".9",
        "--mamba-ssm-dtype", "bfloat16", "--mamba-full-memory-ratio", ".31",
        "--mamba-radix-cache-strategy", "extra_buffer", "--max-prefill-tokens", "128",
        "--chunked-prefill-size", "512", "--cuda-graph-max-bs-decode", "48",
        "--disable-overlap-schedule",
    )
    assert config.max_seq_len_override == 1048576 and config.memory_ratio == .9
    assert config.recurrent_state_dtype == torch.bfloat16 and config.state_memory_ratio == .31
    assert config.state_cache_strategy == "extra_buffer" and not config.enable_cache_overlap
    assert (config.max_prefill_tokens, config.chunked_prefill_size) == (128, 512)
    assert config.max_forward_len == 512 and config.cuda_graph_max_bs == 48
    assert not hasattr(config, "max_cached_states")


def test_default_budgets_are_16384_and_overlap_is_enabled():
    config = parse()
    assert (config.max_prefill_tokens, config.chunked_prefill_size) == (16384, 16384)
    assert config.enable_cache_overlap


@pytest.mark.parametrize("name,value", [
    ("--model", "unused"), ("--max-seq-len-override", "1024"),
    ("--memory-ratio", ".9"), ("--recurrent-state-dtype", "bfloat16"),
    ("--state-memory-ratio", ".31"), ("--max-extend-length", "128"),
    ("--max-prefill-length", "128"), ("--cuda-graph-max-bs", "48"),
    ("--graph", "48"), ("--max-cached-states", "12"),
    ("--attn", "sgl"), ("--tokenizer-count", "0"), ("--tensor-parallel-size", "8"),
    ("--enable-cache-overlap", None),
])
def test_removed_names_and_abbreviations_are_rejected(name, value):
    with pytest.raises(SystemExit):
        parse(name, *([] if value is None else [value]))


@pytest.mark.parametrize("options", [
    ["--max-prefill-tokens", "64", "--chunked-prefill-size", "512"],
    ["--chunked-prefill-size", "512", "--max-prefill-tokens", "64"],
])
def test_argument_order_cannot_alias_the_two_budgets(options):
    config = parse(*options)
    assert (config.max_prefill_tokens, config.chunked_prefill_size) == (64, 512)


@pytest.mark.parametrize("name", ["max_prefill_tokens", "chunked_prefill_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_budgets_require_positive_integer_values(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        SchedulerConfig("unused", DistributedInfo(0, 1), torch.bfloat16, **{name: value})


def test_engine_has_no_fixed_checkpoint_capacity_configuration():
    with pytest.raises(TypeError, match="max_cached_states"):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, max_cached_states=8)


def test_extra_buffer_auto_uses_existing_phase_portfolio_without_model_name(monkeypatch):
    monkeypatch.setattr("minisgl.engine.engine.logger.warning_rank0", lambda *args: None)
    config = SchedulerConfig(
        "unused", DistributedInfo(0, 8), torch.bfloat16,
        state_cache_strategy="extra_buffer", dcp_size=8, page_size=64,
        recurrent_state_dtype=torch.bfloat16, max_running_req=48,
    )
    config.__dict__["model_config"] = SimpleNamespace(
        has_recurrent_state=True, cache_config=HybridCacheConfig(
            2, (1,), (0,), 512, 64, 96, 128, 4
        ), num_qo_heads=96, is_moe=False,
    )
    _adjust_config(config)
    assert config.attention_backend == "fi,sgl"
    assert config.cache_type == "hybrid" and config.state_memory_ratio == .9
    assert config.state_slot_budget == 144


def native_budget_methods():
    spec = importlib.util.find_spec("sglang")
    if spec is None:
        pytest.skip("Optional native SDK source is not installed")
    path = Path(next(iter(spec.submodule_search_locations))) / "srt/managers/schedule_policy.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrefillAdder")
    names = {"ceil_paged_tokens", "_update_prefill_budget", "_select_prefill_admission"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {
        "CLIP_MAX_NEW_TOKENS": 4096,
        "_PrefillAdmission": lambda p, n, m, c: SimpleNamespace(
            prefix_len=p, extend_len=n, max_new_tokens=m, is_chunked=c
        ),
        "AddReqResult": SimpleNamespace(OTHER="other", NO_TOKEN="no_token"),
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("input_budget,chunk_budget,lengths", [
    (128, 512, [200]), (512, 128, [63, 65]), (512, 256, [1, 63, 65]), (128, 128, [65]),
])
def test_shared_page_billing_and_admission_match_native_methods(
    monkeypatch, input_budget, chunk_budget, lengths
):
    methods = native_budget_methods()
    native = SimpleNamespace(
        page_size=64, exact_chunk_fill=False, dllm_config=None,
        rem_input_tokens=input_budget, rem_chunk_tokens=chunk_budget,
        memory_budget=SimpleNamespace(reserve=lambda *args, **kwargs: None),
        per_req_token_overhead=64, rem_mamba_slots=None,
        log_hit_tokens=0, log_input_tokens=0, can_run_list=[],
        _check_prefill_tile_budget=lambda tokens: None,
    )
    for name in ("ceil_paged_tokens", "_update_prefill_budget", "_select_prefill_admission"):
        setattr(native, name, methods[name].__get__(native))
    native._check_prefill_budget = lambda *args, **kwargs: (True, native.rem_chunk_tokens)
    adder = PrefillAdder(
        input_budget, 0, None, SimpleNamespace(token_pool=torch.empty(1, 1024, dtype=torch.int32)),
        chunk_alignment=64, chunk_budget=chunk_budget, compute_page_size=64,
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda t: t)
    for uid, length in enumerate(lengths):
        if native.rem_input_tokens <= 0 or native.rem_chunk_tokens <= 0:
            assert adder.try_add_one(None) is None
            break
        req = SimpleNamespace(prefix_len=0, full_untruncated_fill_ids=range(length),
                              sampling_params=SimpleNamespace(max_new_tokens=1))
        expected = native._select_prefill_admission(
            req, total_tokens=length, host_hit_length=0, swa_host_hit_length=0,
            truncation_align_size=None,
        )
        if expected == "other":
            assert adder._chunk_size(SimpleNamespace(input_len=length), SimpleNamespace(), 0) == 0
            break
        pending = SimpleNamespace(
            input_len=length, input_ids=torch.arange(length, dtype=torch.int32),
            output_len=1, uid=uid, chunked_req=None, sampling_params=SamplingParams(max_tokens=1),
        )
        actual = adder._add_one_req(pending, SimpleNamespace(), 0, 0)
        assert actual.extend_len == expected.extend_len
        native._update_prefill_budget(0, expected.extend_len, 1, False)
        assert (adder.token_budget, adder.chunk_budget) == (
            native.rem_input_tokens, native.rem_chunk_tokens
        )
        native.can_run_list.append(req)
    # The native input budget is an admission stop gate, not a hard shape cap.
    if (input_budget, chunk_budget, lengths) == (128, 512, [200]):
        assert adder.token_budget == -128 and adder.chunk_budget == 256
