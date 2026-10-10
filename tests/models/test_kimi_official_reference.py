"""Optional CPU parity against unchanged definitions from Moonshot's HF source.

Fetch only the pinned 52 KiB source (never weights) with fetch_kimi_reference.py,
then set KIMI_OFFICIAL_REFERENCE_PATH to that file. No network access occurs in
these tests. The entire file's SHA256 is checked before parsing or executing it.

Only the named, unchanged AST definitions are executed. Imports, registration,
HF model wrappers, and KDA definitions are excluded; no fused-kernel replacement
or mini-sglang operation is injected into the reference. Postponed annotations
avoid importing unused typing dependencies. The reference config is a plain
namespace, its supported MLA backend is eager, and its own dynamic cache is used.

These are official operator and MLA-only decoder-stack checks, NOT full hybrid
K3 parity, GPU parity, checkpoint validation, or deterministic generation tests.
The full HF model forces FlashAttention2 and KDA requires FLA GPU kernels.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.kvcache.kimi_pool import KimiHybridCache
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_attention import kda_recurrence
from minisgl.models.kimi_attention_ops import KimiMLA
from minisgl.models.kimi_config import KimiTextConfig
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM, attention_residual
from minisgl.models.kimi_moe import KimiK3MoE, KimiLinear, KimiRMSNorm, KimiRouter

REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
SOURCE_SHA256 = "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"
SOURCE_URL = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{REVISION}/modeling_kimi_linear.py"
DEFINITIONS = {
    "SituAndMul",
    "_get_situ_activation_params",
    "KimiDynamicCache",
    "KimiRMSNorm",
    "KimiBlockSparseMLP",
    "KimiMLP",
    "repeat_kv",
    "eager_attention_forward",
    "KimiMLAAttention",
    "KimiMoEGate",
    "KimiSparseMoeBlock",
    "KimiDecoderLayer",
    "_apply_attn_res",
}
FLA_REVISION = "14bef3d9ec98ee359c9a9c7860795d91675dd091"
FLA_SHA256 = "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016"


@pytest.fixture(scope="module")
def official():
    path = os.environ.get("KIMI_OFFICIAL_REFERENCE_PATH")
    if not path:
        pytest.skip("Set KIMI_OFFICIAL_REFERENCE_PATH to the pinned official HF Python source")
    source = Path(path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256, "Official source hash mismatch"
    parsed = ast.parse(source, filename=str(path))
    definitions = [
        node
        for node in parsed.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in DEFINITIONS
    ]
    assert {node.name for node in definitions} == DEFINITIONS
    # Preserve every selected node unchanged, including function bodies/decorators.
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *definitions],
        type_ignores=[],
    )
    namespace = {"torch": torch, "nn": torch.nn, "F": F, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def config(**changes):
    values = {
        "hidden_size": 8,
        # HF MLA hardcodes its low-rank norms to 1e-6. Match that here to
        # isolate stack wiring; the released-config discrepancy is tested below.
        "rms_norm_eps": 1e-6,
        "vocab_size": 23,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 12,
        "max_position_embeddings": 32,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 3,
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
            "kda_layers": [],
            "full_attn_layers": [1, 2, 3, 4],
        },
        "attn_res_block_size": 2,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_expert_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 1.3,
        "moe_intermediate_size": 6,
        "routed_expert_hidden_size": 4,
        "latent_moe_use_norm": True,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
    }
    values.update(changes)
    return KimiTextConfig(**values)


def official_config(c):
    values = c.to_dict()
    values.update(_attn_implementation="eager", is_mla=True, is_kda_layer=lambda _: False)
    # Official HF represents disabled shared experts with None, not zero.
    values["num_shared_experts"] = c.num_shared_experts or None
    return SimpleNamespace(**values)


def initialize(op, seed=71):
    generator = torch.Generator().manual_seed(seed)
    for name, value in op.state_dict().items():
        value.copy_(torch.randn(value.shape, generator=generator).to(value) * 0.3)
        if "norm.weight" in name:
            value.add_(1.0)


def load_official(reference, actual):
    weights = {name: tensor.detach().clone() for name, tensor in actual.state_dict().items()}
    reference.load_state_dict(weights, strict=True)
    reference.eval()


def assert_close(actual, expected, dtype=torch.float32, *, stack=False):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.04 if dtype == torch.bfloat16 else (1e-4 if stack else 2e-5),
        atol=0.02 if dtype == torch.bfloat16 else (2e-5 if stack else 2e-6),
    )


def causal_mask(start, end, dtype):
    allowed = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]
    return torch.zeros((1, 1, end - start, end), dtype=dtype).masked_fill(~allowed, -torch.inf)


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("renormalize", [False, True])
def test_official_grouped_router(official, top_k, renormalize, request):
    c = config(num_experts_per_token=top_k, moe_renormalize=renormalize)
    actual = KimiRouter(c)
    initialize(actual)
    reference = official.KimiMoEGate(official_config(c))
    load_official(reference, actual)
    x = torch.randn(2, 3, c.hidden_size, generator=torch.Generator().manual_seed(19))
    ids, weights = reference(x)
    got_weights, got_ids = actual.forward(x.flatten(0, 1))
    torch.testing.assert_close(got_ids, ids)
    if top_k == 1 and renormalize:
        request.node.add_marker(
            pytest.mark.xfail(strict=True, reason="HF does not renormalize top-1; mini-sglang does")
        )
    assert_close(got_weights, weights)


@pytest.mark.parametrize("latent,shared", [(4, 1), (None, 1), (4, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_official_latent_moe(official, latent, shared, dtype):
    c = config(routed_expert_hidden_size=latent, num_shared_experts=shared)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        actual = KimiK3MoE(c)
    finally:
        torch.set_default_dtype(old)
    initialize(actual)
    reference = official.KimiSparseMoeBlock(official_config(c)).to(dtype)
    load_official(reference, actual)
    x = torch.randn(2, 3, c.hidden_size, generator=torch.Generator().manual_seed(13)).to(dtype)
    assert_close(actual.forward(x), reference(x), dtype)


@pytest.mark.parametrize("output_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_official_mla_prefill_and_cached_decode(official, output_gate, dtype):
    c = config(mla_use_output_gate=output_gate)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        actual = KimiMLA(c)
    finally:
        torch.set_default_dtype(old)
    initialize(actual)
    ref_config = official_config(c)
    reference = official.KimiMLAAttention(ref_config, 0).to(dtype)
    load_official(reference, actual)
    x = torch.randn(7, c.hidden_size, generator=torch.Generator().manual_seed(11)).to(dtype)
    expected = reference(x[None], attention_mask=causal_mask(0, 7, dtype))[0]
    got, _ = actual.forward(x, torch.arange(7))
    assert_close(got, expected, dtype)
    state = None
    ref_cache = official.KimiDynamicCache(ref_config)
    for start, end in [(0, 3), (3, 5), (5, 6), (6, 7)]:
        got, state = actual.forward(x[start:end], torch.arange(start, end), state)
        expected = reference(
            x[None, start:end],
            attention_mask=causal_mask(start, end, dtype),
            past_key_values=ref_cache,
        )[0]
        assert_close(got, expected, dtype)


@pytest.mark.parametrize("bank_size", [0, 1, 3])
def test_official_attention_residual(official, bank_size):
    norm, proj = KimiRMSNorm(8, 1e-5), KimiLinear(8, 1)
    initialize(norm)
    initialize(proj, seed=73)
    reference_norm = official.KimiRMSNorm(8, eps=1e-5)
    reference_proj = torch.nn.Linear(8, 1, bias=False)
    load_official(reference_norm, norm)
    load_official(reference_proj, proj)
    rows = torch.randn(3, bank_size + 1, 8, generator=torch.Generator().manual_seed(17))
    head, bank = rows[:, -1], list(rows[:, :-1].unbind(1))
    expected = official._apply_attn_res(head, rows[:, :-1], reference_proj, reference_norm)
    assert_close(attention_residual(head, bank, norm, proj), expected)


@pytest.mark.xfail(strict=True, reason="HF MLA low-rank eps=1e-6 vs mini-sglang config eps=1e-5")
def test_official_mla_released_config_epsilon_discrepancy(official):
    c = config(rms_norm_eps=1e-5)
    actual = KimiMLA(c)
    initialize(actual)
    reference = official.KimiMLAAttention(official_config(c), 0)
    load_official(reference, actual)
    x = torch.randn(7, c.hidden_size, generator=torch.Generator().manual_seed(11))
    expected = reference(x[None], attention_mask=causal_mask(0, 7, torch.float32))[0]
    got, _ = actual.forward(x, torch.arange(7))
    assert_close(got, expected)


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_official_mla_only_decoder_stack(official, block_size):
    """Compare each layer and snapshot bank across multiple AttnRes boundaries."""
    c = config(attn_res_block_size=block_size)
    model_config = ModelConfig.from_hf(c)
    actual = KimiK3LinearForCausalLM(model_config)
    initialize(actual)
    ref_config = official_config(c)
    references = []
    for index, layer in enumerate(actual.model.layers.op_list):
        reference = official.KimiDecoderLayer(ref_config, index)
        weights = {
            name.replace("mlp.", "block_sparse_moe.", 1)
            if name.startswith("mlp.") and hasattr(reference, "block_sparse_moe")
            else name: value.clone()
            for name, value in layer.state_dict().items()
        }
        reference.load_state_dict(weights, strict=True)
        references.append(reference.eval())
    cache = KimiHybridCache(c, 64, 1, 1, torch.float32, torch.device("cpu"))
    reference_cache = official.KimiDynamicCache(ref_config)
    for start, end in [(0, 3), (3, 5), (5, 6)]:
        tokens = torch.tensor([2, 4, 8, 3, 9, 1])[start:end]
        request = SimpleNamespace(
            uid=10, table_idx=0, cached_len=start, device_len=end, extend_len=end - start
        )
        batch = SimpleNamespace(
            reqs=[request],
            input_ids=tokens,
            positions=torch.arange(start, end),
            out_loc=torch.arange(start, end),
        )
        head = actual.model.embed_tokens.forward(tokens)
        expected = head[None].clone()
        bank = []
        ref_bank = expected.new_empty(end - start, 0, c.hidden_size)
        # Request ownership is normally begun by model.forward_batch.
        cache.begin_request(0, 10, start)
        for layer, reference in zip(actual.model.layers.op_list, references):
            head = layer.forward(head, bank, batch, cache, torch.arange(32)[None])
            expected, ref_bank = reference(
                expected,
                attention_mask=causal_mask(start, end, torch.float32),
                past_key_values=reference_cache,
                block_residual=ref_bank,
            )
            # Different score/reduction ordering is amplified by repeated latent
            # norms. Keep the tighter tolerance for standalone operators above.
            assert_close(head, expected[0], stack=True)
            assert_close(torch.stack(bank, dim=1), ref_bank, stack=True)
        cache.finish_request(0, 10, end)
        norm = official.KimiRMSNorm(c.hidden_size, c.rms_norm_eps)
        proj = torch.nn.Linear(c.hidden_size, 1, bias=False)
        load_official(norm, actual.model.output_attn_res_norm)
        load_official(proj, actual.model.output_attn_res_proj)
        assert_close(
            attention_residual(
                head, bank, actual.model.output_attn_res_norm, actual.model.output_attn_res_proj
            ),
            official._apply_attn_res(expected[0], ref_bank, proj, norm),
            stack=True,
        )


@pytest.fixture(scope="module")
def official_fla():
    path = os.environ.get("KIMI_FLA_REFERENCE_PATH")
    if not path:
        pytest.skip("Set KIMI_FLA_REFERENCE_PATH to the pinned official FLA naive.py source")
    source = Path(path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == FLA_SHA256, "FLA source hash mismatch"
    parsed = ast.parse(source, filename=str(path))
    names = {"naive_recurrent_kda", "naive_chunk_kda"}
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["recurrent", "chunk"])
def test_official_fla_kda(official_fla, initial_state, dtype, mode):
    """Official recurrence and independently formulated chunk solve on CPU.

    FLA's naive API expects already normalized Q/K and gates in log-space. The
    adapter performs only that documented input normalization; it never invokes
    a mini-sglang primitive to compute reference outputs. Gate formulas and
    convolution are outside this specific check's scope.
    """
    if mode == "chunk":
        official_fla["rearrange"] = pytest.importorskip("einops").rearrange
    generator = torch.Generator().manual_seed(47)
    q, k = [torch.randn(8, 2, 4, generator=generator).to(dtype) for _ in range(2)]
    v = torch.randn(8, 2, 3, generator=generator).to(dtype)
    g = -torch.rand(8, 2, 4, generator=generator)
    beta = torch.rand(8, 2, generator=generator)
    state = torch.randn(2, 4, 3, generator=generator) if initial_state else None
    # Keep normalized inputs FP32, matching the production API's internal norm
    # and FLA naive's documented FP32 recurrence arithmetic.
    normalized = [
        x.float() / torch.sqrt(x.float().square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)
    ]
    reference = official_fla[f"naive_{mode}_kda"]
    kwargs = {"chunk_size": 2} if mode == "chunk" else {}
    expected, expected_state = reference(
        normalized[0][None],
        normalized[1][None],
        v[None],
        g[None],
        beta[None],
        initial_state=None if state is None else state[None],
        output_final_state=True,
        **kwargs,
    )
    actual, actual_state = kda_recurrence(q, k, v, g, beta, state)
    assert_close(actual, expected[0], dtype)
    assert_close(actual_state, expected_state[0])
    carried = state
    outputs = []
    for start, end in [(0, 3), (3, 7), (7, 8)]:
        output, carried = kda_recurrence(
            q[start:end], k[start:end], v[start:end], g[start:end], beta[start:end], carried
        )
        outputs.append(output)
    assert_close(torch.cat(outputs), expected[0], dtype)
    assert_close(carried, expected_state[0])
