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


@pytest.mark.parametrize("engine_kind", ["mini", "sglang"])
@pytest.mark.parametrize("target", [False, True])
def test_quality_runner_passes_explicit_precision_to_the_real_engine_interface(
    monkeypatch, tmp_path, engine_kind, target
):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import minisgl.engine as mini_engine
    import minisgl.utils as utils
    import torch

    suite = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))
    seen = {}

    class Engine:
        def __init__(self, config=None, **kwargs):
            seen.update(config=config, kwargs=kwargs)
            self.device = torch.device("cpu")
            self.page_table = torch.zeros((1, 4), dtype=torch.int32)
            self.graph_runner = SimpleNamespace(
                pad_batch=lambda batch: None, can_use_cuda_graph=lambda batch: False
            )
            self.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
            self.ctx = SimpleNamespace(forward_batch=lambda batch: nullcontext())
            self.model = SimpleNamespace(forward=lambda: torch.tensor([[0.0, 1.0]]))
            self.kv_cache = SimpleNamespace(release_request=lambda slot, uid: None)

        def generate(self, **kwargs):
            return [
                {
                    "output_ids": [1],
                    "text": "#### 1",
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }
            ]

        def shutdown(self):
            seen["closed"] = True

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
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32
    )
    cases = tmp_path / "cases.json"
    cases.write_text("{}")
    args = SimpleNamespace(
        model="unused",
        output=tmp_path,
        cases=cases,
        engine=engine_kind,
        tp_size=1,
        dcp_size=1,
        ep_size=1,
        attention_backend="auto",
        mxfp4_backend="flashinfer" if target else "torch",
        moe_backend="auto" if target else "marlin",
        activation_precision="mxfp8" if target else "bf16",
        kv_cache_dtype="fp8_e4m3" if target else "bfloat16",
        recurrent_state_dtype="bfloat16" if target else "float32",
        bf16_precision="default" if target else "strict",
        max_context=4,
        batch_size=1,
        max_new_tokens=1,
        cuda_graph=False,
        flashinfer_autotune_skip_ops=[],
        sglang_attention_backend=None,
        sglang_prefill_attention_backend=None,
        sglang_decode_attention_backend=None,
    )
    suite["run_" + engine_kind](args, [{"id": 0, "gold": "1", "input_ids": [0]}], {1})
    if engine_kind == "mini":
        config = seen["config"]
        assert config.kv_cache_dtype == (torch.float8_e4m3fn if target else torch.bfloat16)
        assert config.recurrent_state_dtype == (torch.bfloat16 if target else torch.float32)
        assert config.mxfp4_activation_precision == args.activation_precision
        assert config.moe_backend == args.moe_backend
    else:
        config = seen["kwargs"]
        assert config["kv_cache_dtype"] == args.kv_cache_dtype
        assert config["mamba_ssm_dtype"] == args.recurrent_state_dtype
        assert config["flashinfer_mxfp4_moe_precision"] == ("default" if target else "bf16")
        assert config["moe_runner_backend"] == args.moe_backend
    assert seen["closed"]
    assert not torch.cuda.is_initialized()
