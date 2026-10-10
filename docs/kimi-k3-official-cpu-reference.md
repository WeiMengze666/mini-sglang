# Independent official-source CPU checks

These optional tests execute unchanged definitions from official Moonshot HF
and FLA sources, without importing SGLang, FLA kernels, or a GPU stack. They
establish **operator and MLA-only decoder-stack parity**, not full hybrid K3
model parity, released-weight parity, or serving correctness.

## Reproduce

From the repository root, using the CPU test environment described in
[kimi-k3.md](kimi-k3.md):

```sh
python tests/models/fetch_kimi_reference.py /tmp/kimi-official-reference
# Optional: required only for the four official FLA chunk-formulation cases.
python -m pip install einops==0.8.1
KIMI_OFFICIAL_REFERENCE_PATH=/tmp/kimi-official-reference/modeling_kimi_linear.py \
KIMI_FLA_REFERENCE_PATH=/tmp/kimi-official-reference/fla_kda_naive.py \
PYTHONPATH=python python -m pytest -o addopts='' \
  tests/models/test_kimi_official_reference.py -q
```

The fetch script downloads four pinned files, 62,006 bytes total: two source
files and their licenses. Each download is bounded to 128 KiB and hash-checked.
It never downloads weights, executes downloaded code, installs packages, or uses
HF credentials. Test runs never access the network. If an optional source path
is absent, its cases explicitly skip. A supplied invalid path/hash fails.

Validation on PyTorch 2.9.1+cpu, Python 3.12, einops 0.8.1:
**27 passed, 2 strict expected failures**. The expected failures document actual
upstream semantic disagreements below; they must not be reported as parity.
Without einops, four chunk cases skip; recurrent KDA still runs using only torch.

## Independence and boundaries

- Full-file SHA256 validation precedes AST extraction. Selected class/function
  nodes are unchanged. Module-level imports and registration are not executed.
  HF annotations are postponed to avoid unrelated import dependencies.
- HF eager MLA, its own expanded K/V dynamic cache, grouped routing, dense and
  latent/shared SiTU MoE, and attention residuals execute their official code.
- The four-layer MLA-only stack executes official `KimiDecoderLayer` objects,
  comparing every layer output, raw snapshot bank, and final attention-residual
  readout. It covers dense and MoE layers, block sizes 1/2/3, prefill/chunk/decode.
  This deliberately omits the HF full-model wrapper, embeddings/logits comparison,
  KDA layers, and its forced FlashAttention2 selection.
- The reference receives a simple configuration namespace, eager attention,
  explicit causal masks, and matched synthetic weights. `is_kda_layer` is false;
  no attention kernel is replaced. Zero shared experts maps to HF's `None`.
- FLA's unchanged recurrent and chunk algorithms independently check KDA outputs
  and final recurrent states with FP32/BF16 inputs, nonzero initial states, and
  split-stream carry. FLA's naive API requires already-normalized Q/K, log-space
  decay and sigmoid beta. The adapter normalizes Q/K with ordinary torch; no
  mini-sglang numerical function computes reference outputs. Gate computation,
  convolution, and fused gated norm are outside this FLA comparison.
- Operator FP32 tolerance is `rtol=2e-5, atol=2e-6`. Stack tolerance is
  `rtol=1e-4, atol=2e-5`: different score/reduction ordering is amplified by
  repeated latent norms (observed approximately 9e-6 absolute difference).
  BF16 comparisons use `rtol=0.04, atol=0.02`. The two semantic discrepancies
  retain tight FP32 operator tolerances rather than being hidden by widening them.

An independently runnable, unchanged hybrid HF model is still unavailable on
this CPU-only setup: the official module imports FLA kernels unconditionally,
KDA calls GPU chunk/recurrent kernels and fused modules, and the full model
constructor forces FlashAttention2. Replacing all those pieces with fresh
handwritten math would introduce another port, rather than validate the existing
port against unmodified execution. No claim of full hybrid parity is made.

## Additional hybrid math oracle (explicitly adapted)

`tests/models/test_kimi_hybrid_reference.py` adds eight FP32 cases using the same
hash-verified HF source plus its unchanged `KimiDeltaAttention` and decoder
classes. GPU-only short convolution, fused gated RMSNorm and raw-gate/QK
preprocessing are explicitly replaced by independent PyTorch formulas. KDA
updates execute the unchanged FLA naive recurrence. No mini numerical routine
computes reference outputs; the convolution uses `F.conv1d` rather than mini's
window/reduction implementation. Recurrent layout is transposed at the adapter
boundary, and convolution history storage follows the CPU adapter contract.

These tests compare every layer and AttnRes bank, final logits and three greedy
steps on KDA/MLA/KDA/MLA tiny networks with both gate ranks, bounded/unbounded
decay and residual block sizes 1/2. MLA epsilon is the mutually supported 1e-6,
not the released-config discrepancy. They execute in addition to the original
unchanged-kernel-boundary operator suite:

```sh
KIMI_OFFICIAL_REFERENCE_PATH=/path/to/modeling_kimi_linear.py \
KIMI_FLA_REFERENCE_PATH=/path/to/fla_kda_naive.py PYTHONPATH=python \
python -m pytest -o addopts='' tests/models/test_kimi_hybrid_reference.py -q
```

This closes a tiny hybrid wiring/math gap but remains an **adapted independent
mathematical reference**, not an unchanged full HF/SGLang GPU execution, packed
kernel validation, production cache-layout parity or released-weight accuracy.

## Observed upstream disagreements

The implementation follows pinned SGLang for these two behaviors:

1. **MLA low-rank RMSNorm epsilon.** Official HF
   [lines 368 and 383](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)
   omit epsilon, selecting `1e-6`. SGLang
   [DeepseekV2AttentionMLA, lines 2042 and 2128](https://github.com/sgl-project/sglang/blob/ff19159e162595f26e6d01eabcbbd4638b3bd228/python/sglang/srt/models/deepseek_v2.py#L2042)
   uses `config.rms_norm_eps`; the released K3 config has `1e-5`. The retained
   discrepancy case measures maximum absolute output difference **8.4341e-5**
   (maximum elementwise relative difference **8.4631e-4**) for the fixed synthetic
   input. Wiring parity cases explicitly set the mutually supported `1e-6`.
   This does not demonstrate released-config HF logit parity.
2. **Renormalized top-1 routing.** HF
   [line 753](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)
   renormalizes only when `top_k > 1`. SGLang
   [biased grouped top-k, lines 1663–1674](https://github.com/sgl-project/sglang/blob/ff19159e162595f26e6d01eabcbbd4638b3bd228/python/sglang/srt/layers/moe/topk.py#L1663)
   renormalizes whenever requested. At scaling factor 1.3, this test gives our
   weights 1.3 versus HF weights 0.4132–0.9746; maximum absolute gap **0.88684**.
   Released K3 uses top-16, so this is a reduced-config edge case.

Both discrepancy cases are strict expected failures: a future change that makes
either pass must trigger review rather than silently changing the test meaning.

## Source and license provenance

- Moonshot HF revision `f831ab66814297da540d832a5235f8e904f29d06`:
  `modeling_kimi_linear.py`, SHA256
  `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`.
  Upstream header identifies Apache-2.0 for DeepSeek-derived portions and the
  [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/LICENSE)
  for other portions. The fetch helper retains the full original source/header
  and license; no HF source is vendored in this repository.
- Official FLA revision `14bef3d9ec98ee359c9a9c7860795d91675dd091`:
  [fla/ops/kda/naive.py](https://github.com/fla-org/flash-linear-attention/blob/14bef3d9ec98ee359c9a9c7860795d91675dd091/fla/ops/kda/naive.py),
  SHA256 `60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016`.
  [MIT License](https://github.com/fla-org/flash-linear-attention/blob/14bef3d9ec98ee359c9a9c7860795d91675dd091/LICENSE).
  Full source and license are downloaded separately, not vendored.
