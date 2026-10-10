# Kimi-K3: text-only correctness baseline

This is an **experimental, eager, single-rank implementation**, not validated
production support for the released checkpoint. It deliberately stops before
performance work, TP/EP/DCP, fused kernels and real-model accuracy claims.
The released MXFP4 checkpoint is decoded into dense weights. Its complete dense
model must fit on one device; this version is **not a practical full K3 deployment
recipe**. There is no GPU on the development machine, and no model weights were
downloaded or run.

## Implemented

- Local data-only `kimi_k3`/nested `kimi_linear` config loading without executing
  checkpoint Python. K3 architecture is selected explicitly and MoE detection
  uses expert count, not a name substring.
- Local data-only TikToken tokenizer with segmented XTML text chat, literal-marker
  escaping, thinking options and generation stop IDs distinct from tokenizer EOS.
- KDA causal depthwise convolution, normalized Q/K delta recurrence with FP32
  state, safe lower-bound decay gate, full/low-rank output gate and gated RMSNorm.
- Gated MLA with a normalized latent/shared-key cache. K3's NoPE keeps the shared
  suffix in attention scores without applying RoPE. The readable attention path
  temporarily expands cached keys/values; it is not an optimized MLA kernel.
- Block attention residuals, SiTU and Stable LatentMoE with correction-biased
  grouped routing, shared experts and post-mixture latent normalization.
- Unfused checkpoint names, text-wrapper / `block_sparse_moe` mapping, KDA A_log
  normalization and streaming E2M1/E8M0 MXFP4 dequantization. Native dense dtypes
  are preserved until assignment to the declared model parameter dtype.
- Fixed-capacity request-owned convolution/recurrent state, paged latent MLA
  cache, fixed-state memory reservation, processed-length/ownership validation,
  abort/release/reset and slot reuse. Only real token rows are processed/written.

The existing scheduler/engine remain in charge of batching, token pages and
request lifetime. K3 uses a small model-specific pool and metadata backend;
existing MHA models retain their original cache/backend path. No production
SGLang execution framework was imported.

## Deliberate restrictions

K3 selects its eager backend and disables CUDA graphs, cross-request prefix
reuse and overlap scheduling with a startup warning. TP greater than one and
incompatible attention backends fail explicitly. Vision and prediction heads
are excluded; the server remains text-only. Tied embeddings, all-KDA schedules,
tool/image chat formatting, unsupported quantization formats and unsupported config shapes fail rather
than silently following another model path.

Request state is not a reusable token-prefix snapshot. A failed forward discards
state for the requests it began; restarting them must start from token zero.
It does not rewind a partly evaluated recurrent state. Slots cannot be resumed
at a mismatched processed length.

## CPU checks

From the repository root, using Python 3.12:

```sh
python -m venv .venv
.venv/bin/pip install 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install 'transformers==4.57.3' safetensors pytest ruff pyzmq msgpack einops 'tiktoken>=0.9,<1'
PYTHONPATH=python .venv/bin/python -m pytest -o addopts='' \
  tests/core/test_kimi_attention.py tests/core/test_kimi_model.py \
  tests/test_kimi_cache.py tests/models/test_kimi_moe.py tests/models/test_kimi_weight.py \
  tests/models/test_kimi_headers.py tests/models/test_kimi_tokenizer.py \
  tests/models/test_tokenizer_integration.py tests/core/test_cache_allocate.py
PYTHONPATH=python .venv/bin/python tests/misc/test_serialize.py
```

These checks do not install CUDA kernels or download model weights. Optional
[official-source CPU comparisons](kimi-k3-official-cpu-reference.md) and a
[bounded checkpoint-header audit](kimi-k3-checkpoint-audit.md) add independent
reference and actual-shape evidence. [Tokenizer checks](kimi-k3-tokenizer.md)
cover the data-only text-chat path. They test
independent scalar/matrix operator formulas, MXFP4 byte patterns and synthetic
shard loading; whole-model tests check streaming equivalence and isolation on a
tiny random network. They **do not compare full-model logits to an independently
executed SGLang model**.

For a future GPU smoke test, use a *local, suitably small K3-format checkpoint*
whose dense weights fit, the normal mini-sglang CUDA dependencies, TP=1 and a
small request limit. This is a proposed, not executed, command:

```sh
python -m minisgl --model /path/to/local/tiny-k3 --tp-size 1 \
  --max-running-requests 1 --max-seq-len-override 128 --max-extend-length 64
```

Do not use the full remote model ID as a lightweight smoke test: normal
weight loading can download the enormous checkpoint.

## Validation gates

| Gate | Status |
|---|---|
| Primitive KDA, convolution, decay, gated norm, MLA, SiTU, MoE and MXFP4 formulas | CPU tests pass |
| Prefill/chunk/decode equivalence, heterogeneous batch vs isolated, real-row padding, reset/reuse/failure | Tiny CPU model passes (FP32 and selected BF16 paths) |
| Local synthetic checkpoint through strict model loader | CPU test passes |
| Public checkpoint names/shapes/dtypes/packing | All 96 headers audited; zero errors, no tensor payload requested |
| Independent official CPU formulas / MLA-only stack | 27 pass, 2 explicit HF/SGLang semantic disagreements; not full hybrid parity |
| Local tokenizer and serving stop/chat integration | CPU tests pass; optional pinned official BPE-data check passes |
| Existing CPU cache allocation | 6 CPU regression tests pass |
| Existing serialization script | Direct script smoke check passes (not a collected pytest test) |
| GPU serving scheduler, cancellation races, sampling and CUDA memory behavior | Not run |
| Released-checkpoint loading, per-layer/logit parity, deterministic generation | Not run |
| TP, EP, DCP, prefix reuse, graphs, overlap and kernel performance | Deferred |

Before optimization: establish GPU parity on reduced configurations and a
feasible full-checkpoint execution plan; independently compare layer outputs,
logits and short deterministic generations with dtype-aware tolerances. Capacity
sharding may be required for the released model, but is not evidence of basic
correctness. Only after that gate should EP/DCP and performance work begin.

## Semantic sources

- Development target: [WeiMengze666/mini-sglang](https://github.com/WeiMengze666/mini-sglang), with [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) retained as upstream. Both main branches were verified at `9a91cfafe754aa85daee49998176275667eb58f2` on 2026-10-10.
- SGLang reference: `ff19159e162595f26e6d01eabcbbd4638b3bd228`,
  [K3 model](https://github.com/sgl-project/sglang/blob/ff19159e162595f26e6d01eabcbbd4638b3bd228/python/sglang/srt/models/kimi_k3.py),
  [attention residuals](https://github.com/sgl-project/sglang/blob/ff19159e162595f26e6d01eabcbbd4638b3bd228/python/sglang/srt/layers/attn_residual.py).
- [Released config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
  and public safetensors index inspected on 2026-10-10. References establish
  implementation semantics, not independently verified cookbook accuracy.

The implementation preserves the upstream correctness lessons: KDA head size
comes from `linear_attn_config`, not MLA's value head; padded rows cannot write
cache; future shared-expert sharding must match its actual reduction group.

### Metadata audit fingerprint

The public index contains 497,220 raw keys. Filtering vision/auxiliary scale
keys and normalizing text names gives 249,756 keys, exactly matching the full
model constructed on PyTorch's meta device (zero missing/unexpected keys).
The initial key-only audit validates names. The later [header audit](kimi-k3-checkpoint-audit.md) additionally validates actual shapes, dtypes, packing and offsets; neither validates tensor values or actual full-model loading.

- Config SHA256: `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`
- Index SHA256: `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`
- Index bytes: 59,764,096. The large public index and weights are not bundled.

To repeat this audit on locally supplied metadata (no weight reads):

```sh
PYTHONPATH=python .venv/bin/python tests/models/audit_kimi_index.py \
  /path/to/config.json /path/to/model.safetensors.index.json
```
