# Tiny K3 runtime correctness gate

This gate creates a complete **local synthetic K3-format checkpoint**, loads it
through mini-sglang's model registry, streaming weight loader and strict assignment,
and executes the hybrid KDA/MLA model and request-owned cache. It downloads
nothing, never accepts a remote model ID, and does not execute checkpoint Python.

The generated dense and packed checkpoints represent the same deterministic
weights. This provides a reproducible integration fixture and a next-step CUDA
smoke command. It is **not released-checkpoint accuracy, independent full hybrid
K3 parity, performance, or a practical full-model deployment claim**.

## Run offline on CPU

From the repository root, use the existing environment with PyTorch, Transformers,
safetensors and pytest. In this checkout the environment is `../.venv-kimi`:

```sh
PYTHONPATH=python ../.venv-kimi/bin/python tests/models/run_kimi_runtime_gate.py \
  --fixture-dir /tmp/tiny-k3-runtime \
  --report /tmp/k3-runtime-fp32.json

PYTHONPATH=python ../.venv-kimi/bin/python tests/models/run_kimi_runtime_gate.py \
  --fixture-dir /tmp/tiny-k3-runtime --dtype bfloat16 \
  --report /tmp/k3-runtime-bf16.json

PYTHONPATH=python ../.venv-kimi/bin/python -m pytest -o addopts='' -q \
  tests/models/test_kimi_runtime_gate.py
```

Omit `--fixture-dir` to use a temporary directory that is removed afterward.
`--generate-only --fixture-dir /tmp/tiny-k3-runtime` creates fixtures without
running them. An existing generated directory is checksum-verified and reused;
a modified fixture, different seed or unrelated nonempty directory fails without
overwriting it. `--seed` is available for additional test cases. The fixtures have
no tokenizer: token IDs are used directly, and these are not HTTP-server-ready
model folders.

### Fixture contract

- 32 hidden channels, vocabulary 64, three hybrid layers (KDA/MLA/KDA), four
  attention heads, four routed experts with top-2 grouped routing, latent MoE,
  shared experts, and attention residuals
- 64-token maximum sequence; three bounded request/cache slots
- Full 32-column MXFP4 blocks with varied E8M0 exponents. Low/high nibbles are
  packed independently of the production decoder; their directly synthesized
  floating values form the dense twin
- Native dense embeddings/head/router/latent projections; eligible attention and
  FFN projections are packed and remain uint8 in the packed runtime model
- Packed weights and their scales live in separate safetensors shards with an
  index, exercising cross-shard loading, wrapper names and `block_sparse_moe`
  normalization
- A manifest records the seed, parameter/packed-tensor counts and file hashes

The checkpoint is intentionally tiny. It validates packed-persistent execution
and numerical agreement with its dense twin; it does not estimate full K3 peak
memory or measure the quality impact of quantizing pretrained weights.

### Checks and reporting

For **each** dense and packed checkpoint, the gate checks:

1. Whole-prompt prefill against multiple prefill chunks followed by single-token
   decode, including final convolution and FP32 recurrent state
2. A heterogeneous two-request batch against isolated requests, with deliberately
   invalid padding token IDs and cache locations that must remain unused
3. Explicit abort/release, reuse of the same request slot and a stale release from
   the old request
4. An injected failure after the decoder layers have updated recurrent state,
   followed by a clean restart at token zero
5. Cache reset and repeatability
6. Five greedy generated token IDs: cached decode, complete-prefix recomputation,
   and an independent repeat must agree
7. Packed-versus-dense full logits and generation

The report contains dtype-aware tolerances and maximum absolute errors, generated
token IDs, retained weight bytes and cache bytes. FP32 uses `rtol=3e-5,
atol=3e-6`; BF16 uses `rtol=0.04, atol=0.03`. Greedy token comparisons are exact.
These are same-implementation consistency checks; finite logits and matching
sequences are not an external model-quality oracle.

## Optional independent official-source checks

Use the **existing** pinned-source suite rather than a second reference adapter:

```sh
PYTHONPATH=python ../.venv-kimi/bin/python tests/models/run_kimi_runtime_gate.py \
  --fixture-dir /tmp/tiny-k3-runtime \
  --official-source /path/to/verified/modeling_kimi_linear.py \
  --fla-source /path/to/verified/fla_kda_naive.py \
  --report /tmp/k3-runtime-with-reference.json
```

Both files must already exist locally. The reference tests verify pinned SHA256
hashes before selecting unchanged official definitions. The gate does not fetch
sources, install dependencies, or replace reference operations with mini-sglang
operations. Its report retains the pytest output, including explicit expected
semantic disagreements. See [official CPU reference scope and source provenance](kimi-k3-official-cpu-reference.md).

This optional stage covers independent operators, FLA recurrence/chunk formulas
and an MLA-only decoder stack. It still does **not** execute a complete independent
hybrid K3 model or compare real-checkpoint generated text.

## CUDA and real Engine smoke: proposed, not yet run

On a Linux CUDA host with the project's ordinary CUDA-enabled dependencies,
run the exact same local fixture gate on GPU:

```sh
PYTHONPATH=python python tests/models/run_kimi_runtime_gate.py \
  --fixture-dir /tmp/tiny-k3-runtime --device cuda --dtype float32 \
  --report /tmp/k3-runtime-cuda-fp32.json

PYTHONPATH=python python tests/models/run_kimi_runtime_gate.py \
  --fixture-dir /tmp/tiny-k3-runtime --device cuda --dtype bfloat16 \
  --engine-smoke --report /tmp/k3-runtime-cuda-engine.json
```

`--device cuda` fails if CUDA is unavailable; it never silently substitutes CPU.
`--engine-smoke` additionally launches a **fresh process for each fixture** and
constructs the production `Engine` with TP=1, three requests, 192 token pages and
a 64-token sequence limit. It uses real `Batch`/`Req`, attention metadata, the
Engine's stream/model/cache, greedy `Sampler`, and completion event. It compares
five-token Engine generations to direct complete-prefix evaluation, aborts a
partially generated request, reuses its slot, and resets/repeats. Each child
shuts down its process group. Separate processes avoid reusing global Engine and
CUDA initialization state.

The Engine initializes its existing localhost process group on port 2333; run
this smoke in isolation with that port free. `CUDA_VISIBLE_DEVICES` can select the
one GPU; the Engine child uses visible device zero. `--engine-smoke` is not a CPU
Engine shim and it does not test the scheduler, tokenizer, HTTP streaming,
network disconnects, concurrent cancellation races or non-greedy sampling.

Do not replace the tiny local path with the full remote K3 model ID. Actual
released-checkpoint execution still requires its own verified capacity plan and
independent correctness/accuracy validation.

## Recorded development evidence

On the CPU-only development environment, 2026-10-10:

- FP32 and BF16 dense/packed integration gate: passed
- Harness regression suite: 12 passed
- Optional official-source stage through this CLI: 27 passed, 2 explicit expected
  semantic disagreements (top-1 router renormalization and MLA norm epsilon)
- FP32 packed versus dense logits: maximum absolute difference about `3.9e-7`
- Both layouts generated `[22, 5, 58, 44, 47]` for the default fixture
- FP32 retained weight tensors: dense 373,888 bytes; packed 90,172 bytes. These
  counts exclude activation/decode temporaries and allocator overhead
- CUDA execution and production Engine smoke: **not run**; commands above are
  the next executable gate, not passing GPU evidence

The collected regression tests check deterministic fixture construction, exact
packed decoding against the independently synthesized dense tensors, cross-shard
scales, packed storage retention, runtime lifecycle, explicit CUDA failure,
local-path-only loading, safe reuse and CLI/report behavior. Existing
[operator/cache tests](kimi-k3.md) continue to provide narrower unit coverage.

## Long-prefill and non-finite state acceptance

The opt-in tiny-model gate extends both synthetic checkpoint layouts to two
requests longer than 2,048 tokens, compares batched, isolated and chunked
prefill, and continues decoding. It checks finite logits and live KDA/MLA cache
values as well as numerical agreement. Run it separately when iterating on
short unit tests:

```sh
PYTHONPATH=python python tests/models/run_kimi_runtime_gate.py \
  --device cpu --dtype float32 --long-prefill --report /tmp/k3-long-fp32.json
PYTHONPATH=python python tests/models/run_kimi_runtime_gate.py \
  --device cpu --dtype bfloat16 --long-prefill --report /tmp/k3-long-bf16.json
```

The ordinary gate also injects NaN and Inf into one request's recurrent state
following a successful prefill. The harness must detect the bad state even when
previous logits remain finite, preserve an unrelated request's continuation,
and recover cleanly through explicit release/reuse and reset. These are
acceptance-harness checks, not production-wide scans or automatic runtime
recovery. Released MLA storage need not be zeroed; only live cache rows are
checked after they have been written.

Reports include the implementation Git revision, dirty status, actual source
hashes, pinned reference revisions and whether they ran, distinct prefill and
decode backend labels, installed library versions and device model. An absent
optional library is recorded as null. Reference revisions identify the optional
operator checks; they do not mean that the long-prefill test uses an independent
full-model oracle. Exported sources can have no Git revision, while source hashes
still identify their contents.

### Upstream backend caution

[sglang issue44758](https://github.com/sgl-project/sglang/issues/44758), inspected
2026-10-17, reports non-finite long-prefill results with explicitly selected
`nvidia_kda`, especially multi-sequence prefills, with persistent polluted state.
The report distinguishes that path from Triton and the newer FlashInfer default.
These are upstream reported findings, not a reproduction in this repository.
Our gate executes mini-sglang's sequential FP32 KDA recurrence and eager PyTorch
MLA; a batched call processes requests sequentially. Passing it does not validate
NVIDIA KDA, Triton or FlashInfer kernels, real concurrent GPU execution, or the
released full checkpoint. Record and validate actual prefill and decode backend
choices separately when those future hardware gates become available.

CPU acceptance evidence on 2026-10-17: the full opt-in CLI passed FP32 and BF16
for dense/packed fixtures with lengths 2051/2054 and three continued decode
steps. Maximum FP32 comparison error across the long gate was approximately
`3.04e-6`; BF16 comparisons were exact for these synthetic inputs. NaN/Inf fault
checks passed for both layouts/dtypes. The aggregate CPU suite reported
312 passed, 3 skipped and 2 expected semantic disagreements; the skips remain
real Gloo, optional tokenizer asset and optional header pytest coverage. No CUDA,
real serving concurrency, released-weight or independent full-model accuracy
gate was run by this change.
