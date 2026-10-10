# Kimi-K3: text-only correctness baseline

This is an **experimental eager implementation**. It now retains MXFP4 packed
weights and supports basic tensor-parallel sharding for capacity. It is not yet
validated production support for the released checkpoint. GPU serving, real
distributed collectives and released-weight accuracy remain open gates; EP/DCP,
fused kernels and performance work have not begun. No model weights were
downloaded or run on the development machine.

Packed linears decode bounded row chunks during inference rather than retaining
the complete dense model. This is a slow, portable correctness/capacity path,
not a low-bit GEMM optimization. The entire per-rank packed model plus replicated
weights, request state, cache and temporary buffers must still fit in memory.

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
  normalization and E2M1/E8M0 MXFP4 storage. Checkpoint TP slices are read before
  materialization; packed weights and scales stay U8 and decode only bounded
  row chunks. Native dense dtypes are cast only on strict streamed assignment.
- Basic attention-head / FFN-intermediate tensor parallelism. Routed mixtures
  reduce before latent normalization; shared experts use the same TP group.
  Routers/latent projections and MLA caches stay replicated; vocabulary shards
  only when evenly divisible. KDA recurrent and convolution state are rank-local.
- Request validation sends terminal errors instead of silently dropping requests;
  zero-output requests finish without compute. Coalesced detokenizer batches keep
  per-request token order, and abort/terminal paths clean up request state.
- Fixed-capacity request-owned convolution/recurrent state, paged latent MLA
  cache, fixed-state memory reservation, processed-length/ownership validation,
  abort/release/reset and slot reuse. Only real token rows are processed/written.

The existing scheduler/engine remain in charge of batching, token pages and
request lifetime. K3 uses a small model-specific pool and metadata backend;
existing MHA models retain their original cache/backend path. No production
SGLang execution framework was imported.

## Deliberate restrictions

K3 selects its eager backend and disables CUDA graphs, cross-request prefix
reuse and overlap scheduling with a startup warning. Invalid TP divisibility, packed-column block alignment and incompatible
attention backends fail explicitly. Native collectives are used; custom PyNCCL
execution remains disabled for this correctness path. Vision and prediction heads
are excluded; the server remains text-only. Tied embeddings, all-KDA schedules,
tool/image chat formatting, unsupported quantization formats and unsupported config shapes fail rather
than silently following another model path.

Request state is not a reusable token-prefix snapshot. A failed forward discards
state for the requests it began; restarting them must start from token zero.
It does not rewind a partly evaluated recurrent state. Slots cannot be resumed
at a mismatched processed length.

See [capacity layout and loading evidence](kimi-k3-capacity.md) for the packed
format, sharding boundaries and memory-preflight limitations.

## CPU checks

From the repository root, using Python 3.12:

```sh
python -m venv .venv
.venv/bin/pip install 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install 'transformers==4.57.3' safetensors pytest ruff pyzmq msgpack einops fastapi uvicorn prompt_toolkit httpx 'tiktoken>=0.9,<1'
PYTHONPATH=python .venv/bin/python -m pytest -o addopts='' \
  tests/core/test_kimi_attention.py tests/core/test_kimi_model.py \
  tests/test_kimi_cache.py tests/models/test_kimi_moe.py tests/models/test_kimi_weight.py \
  tests/models/test_kimi_headers.py tests/models/test_kimi_tokenizer.py \
  tests/models/test_tokenizer_integration.py tests/models/test_kimi_packed.py \
  tests/models/test_kimi_parallel.py tests/models/test_kimi_loading_integration.py \
  tests/models/test_kimi_request_lifecycle.py tests/models/test_kimi_runtime_gate.py \
  tests/core/test_cache_allocate.py
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

[The runtime gate](kimi-k3-runtime-gate.md) generates deterministic local tiny
dense/packed checkpoints and runs prefill, chunking, decode, greedy generation,
reset/reuse and failure recovery. It also provides opt-in CUDA and real Engine
commands; neither CUDA nor Engine GPU smoke has been executed here.

The two-rank numerical tests use test-only in-memory collectives to validate
shards and reduction placement. Real Gloo initialization was attempted but the
cloud environment denied its sockets (`Operation not permitted`); it is not a
passed distributed gate. On a socket-enabled host run:

```sh
KIMI_RUN_GLOO_TESTS=1 PYTHONPATH=python .venv/bin/python -m pytest -o addopts='' \
  tests/models/test_kimi_parallel.py -q
```

Do not use the full remote model ID as a lightweight smoke test: even packed
weight loading can download roughly 1.56 TB. Engine preflight rejects a rank
whose declared weights/state/minimum cache exceed its memory budget, before
weight download. The estimate is a lower bound, not a peak-memory guarantee.

## Validation gates

| Gate | Status |
|---|---|
| Primitive KDA, convolution, decay, gated norm, MLA, SiTU, MoE and MXFP4 formulas | CPU tests pass |
| Prefill/chunk/decode equivalence, heterogeneous batch vs isolated, real-row padding, reset/reuse/failure | Tiny CPU model passes (FP32 and selected BF16 paths) |
| Local synthetic checkpoint through strict model loader | CPU test passes |
| Public checkpoint names/shapes/dtypes/packing | All 96 headers audited; zero errors, no tensor payload requested |
| Independent unchanged official CPU formulas / MLA-only stack | 27 pass, 2 explicit HF/SGLang semantic disagreements |
| Hybrid CPU math oracle with explicit kernel adapters | 8 cases pass: layer/bank/logit/greedy comparisons; not unchanged GPU-kernel parity |
| Packed-persistent synthetic runtime | FP32/BF16 dense/packed fixtures pass |
| Basic TP sharding and collective placement | Two-rank in-memory numerical simulation passes; real Gloo blocked by environment sockets |
| Scheduler/tokenizer/frontend lifecycle | CPU tests with real scheduling managers and fake compute pass |
| Local tokenizer and serving stop/chat integration | CPU tests pass; optional pinned official BPE-data check passes |
| Existing CPU cache allocation | 6 CPU regression tests pass |
| Existing serialization script | Direct script smoke check passes (not a collected pytest test) |
| GPU serving scheduler, cancellation races, sampling and CUDA memory behavior | Not run |
| Released-checkpoint loading, per-layer/logit parity, deterministic generation | Not run |
| Real TP runtime, EP, DCP, prefix reuse, graphs, overlap and kernel performance | Unvalidated / deferred |

Before optimization: establish GPU parity on reduced configurations and a
feasible full-checkpoint execution plan; independently compare layer outputs,
logits and short deterministic generations with dtype-aware tolerances. Capacity-oriented TP and packed storage now exist, but are not evidence of
full-model runtime correctness. Only after that gate should EP/DCP and performance work begin.

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
