# Kimi-K3 text support

Development is gated in two phases: establish a reusable, correct hybrid runtime,
then add distributed layouts and kernel optimizations. The reference runtime has
passed the gates below on 8×B300. Distributed and performance work follows these
gates; it has not yet established performance parity.

## Runtime structure

K3 owns its checkpoint configuration, weight-name/shard policies, and decoder
composition in `models/kimi_*.py`. The reusable mechanisms live with the rest of
the framework:

| Mechanism | Implementation |
| --- | --- |
| Causal convolution, delta recurrence, gated RMSNorm | `attention/kda.py` |
| Compressed latent attention and optional RoPE | `attention/mla.py` |
| Attention projection layouts | `attention/config.py` |
| KDA and gated MLA projections | `layers/kda.py`, `layers/mla.py` |
| SiTU, grouped routing, shared experts, LatentMoE | `layers/activation.py`, `layers/latent_moe.py` |
| Block attention residuals | `layers/attn_residual.py` |
| Batch metadata and real-row/cache addressing | `attention/recurrent.py` |
| Compressed MLA pages and request-owned recurrent state | `kvcache/mla_pool.py`, `kvcache/recurrent_pool.py` |
| Hybrid cache composition and layout | `kvcache/hybrid_pool.py`, `kvcache/config.py` |
| Sparse state checkpoints over the existing radix tree | `kvcache/hybrid_cache.py` |
| Opt-in compact byte arena | `kvcache/unified.py` |
| Streamed, aligned weight arenas | `layers/weight.py` |
| E2M1/E8M0 decoding and bounded-row linear execution | `quantization/mxfp4.py` |
| Local BPE vocabulary and segmented XTML text formatting | `tokenizer/tiktoken.py`, `tokenizer/xtml.py` |

The engine selects cache/attention behavior from a cache layout and recurrent-state
capability. The scheduler retains ownership of admission, batches, token pages,
request completion and cancellation. K3 does not have a separate scheduler or
its own token/page allocator. The serving model calls the selected attention
backend; explicit-context numerical tests use the same reference layer runner.

## Mutable state and prefix reuse

A radix hit is usable only up to a checkpoint whose KDA state describes exactly
that token prefix. KV alone cannot restore an overwritten recurrent state.

1. Admission locks the matched token prefix and copies its checkpoint into the
   request's private working slot. This is copy-on-write, including convolution
   history and every recurrent layer.
2. The complete model forward commits the processed length only after all
   layers succeed. A failed forward discards its partial working state.
3. Aligned prefill chunk boundaries and decode intervals create independent
   snapshots. Donation transfers a snapshot handle to the radix node without
   copying state again. A later request cannot mutate a donated snapshot.
4. A mid-edge divergence replays from the nearest checkpoint, stops at the
   aligned fork, and places a checkpoint there. Continuing page tables adopt
   the existing shared KV pages before duplicate pages can be reused.
5. Checkpoints have a global capacity, per-path cap and LRU selection that
   prefers retaining branch nodes. KV eviction drops the associated checkpoint.
   Under unified-memory pressure, optional snapshots give bytes back to working
   state and token pages.

All state copies, compaction and forward operations run on the serial forward
stream. CUDA graphs and overlap remain disabled for this reference backend.
The overlapped two-slot donation protocol is **not implemented or accepted**.

`--enable-unified-memory` replaces separately sized state/KV storage with one
byte arena. State grows from one end, MLA pages from the other. A free swaps the
last block of its kind into the hole and updates logical-to-physical handles.
Views are acquired after mutations; logical token page IDs and request slots
stay stable. Admission accounts for a private state slot before admitting KV.
The default remains separate pools while the opt-in path is validated.

## Released checkpoint and validation

The current reference path keeps packed MXFP4 bytes/scales resident and decodes
bounded output-row chunks. This is a capacity/correctness implementation, not a
fused low-bit GEMM. Dense weights, scores, norms, attention and routing still use
their explicit reference operations.

The released model has hundreds of thousands of tensors. Loading them into
individual GPU allocations exhausted allocation/mapping resources with roughly
123 GiB still free per B300. The engine now streams CPU checkpoint slices into
256-byte-aligned views of one GPU arena per dtype. Full TP8 loading then
succeeded, with about 190.97 GiB of weight data per rank.

Evidence collected on 2026-10-26:

| Gate | Result |
| --- | --- |
| Original selected CPU baseline | 103 passed, 2 skipped |
| Refactored CPU tests, real two-process Gloo, actual tokenizer data, pinned HF/FLA oracle | 333 passed, 2 skipped, 2 expected semantic disagreements |
| Unified allocation/compaction and restored-prefix model checks | Passed on CPU |
| Synthetic CUDA, packed weights, real Engine, greedy sampling, release/reuse/reset | Passed, gpuq job 6 |
| Full released checkpoint TP8 loading and eight-token generation | Executed, gpuq job 8; generated `你好。` followed by XTML closing markers |
| Full checkpoint BF16 chunked-prefill pointwise logits | **Failed**: max absolute difference 1.3691, 45.4% outside rtol=0.05/atol=0.1; tracked separately |
| Full checkpoint FP32, 93-layer full/chunked prefill | **Passed**, gpuq job 18: max absolute difference 1.2040e-5, relative L2 9.5633e-7, same greedy token |
| Independent SGLang GPU comparison | gpuq job 13: all 8 generated token IDs match mini job 8; first-token logprob -0.0063424 versus mini -0.0067352 |
| HTTP scheduler/prefix/cancellation tests | **Passed**, gpuq job 14: static and unified pools, aligned forks, concurrent clients, stream close and request reuse |
| DCP, EP, fused kernels, graphs, overlap, performance parity | Not implemented/accepted in this phase |

The short generation is not an accuracy benchmark. The FP32 full-model comparison
isolates state/chunk behavior from BF16 shape-sensitive rounding and downstream
route changes; the BF16 pointwise failure remains visible. Per-layer traces and
full-vocabulary diagnostics are saved before numerical assertions. No throughput
comparison is accepted. Logs and compact reports are under the remote experiment
directory (`released-fp32-trace`, `serving-lifecycle`, `short-reference-comparison.json`).

## Running the gates

GPU work on the shared B300 machine must run through `gpuq`. Keep each complete
experiment, including cleanup, inside one queued task. The development checkout
is `/home/axis/mengze/mini-sglang-k3/mini-sglang`; logs and source snapshots live
in its sibling `experiment` directory. No checkpoint payload is committed.

```sh
# Pinned independent source checks, CPU only.
python tests/models/fetch_kimi_reference.py /tmp/k3-reference
KIMI_OFFICIAL_REFERENCE_PATH=/tmp/k3-reference/modeling_kimi_linear.py \
KIMI_FLA_REFERENCE_PATH=/tmp/k3-reference/fla_kda_naive.py \
KIMI_RUN_GLOO_TESTS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
python -m pytest -o addopts='' tests/models tests/core/test_kimi_attention.py \
  tests/core/test_kimi_model.py tests/core/test_hybrid_components.py \
  tests/core/test_hybrid_prefix.py tests/core/test_unified_memory.py \
  tests/core/test_weight_arena.py tests/core/test_cache_allocate.py tests/test_kimi_cache.py

# Synthetic CUDA and actual Engine.
gpuq env PYTHONPATH=python OMP_NUM_THREADS=1 python \
  tests/models/run_kimi_runtime_gate.py --device cuda --dtype bfloat16 --engine-smoke

# Full released checkpoint, explicitly expensive; serial independent reference follows.
gpuq env PYTHONPATH=python OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=8 tests/validation/released_k3.py \
  --model /data/models/Kimi-K3 --output /path/to/run --trace

# Synthetic HTTP scheduler, prefix forks, cancellation and both pool modes.
gpuq env PYTHONPATH=python OMP_NUM_THREADS=1 python \
  tests/validation/serving_lifecycle.py --output /path/to/http-run
```

The independent SGLang harness uses SGLang's own environment, with a writable
`HF_MODULES_CACHE` if the shared default is read-only. It reads the saved
`prefill.pt` token IDs and returns generation logprobs for comparison. Reference
processes must finish and clean up before another engine occupies the GPUs.

## Source boundaries

- Development starts at `WeiMengze666/mini-sglang` branch
  `kimi-k3-eager-text`, commit `d477af124de2b64a4709654c6144ffcac92e2079`.
- Installed official SGLang checkout: `release/v0.5.22`,
  `baa9befa381511cd4d56f83980951e5a45fffc6d` (checked 2026-10-26).
- CPU formula oracles retain their hash-pinned sources:
  HF `f831ab66814297da540d832a5235f8e904f29d06` and
  FLA `14bef3d9ec98ee359c9a9c7860795d91675dd091`.
- Cache/parallelism design reference:
  [SGLang K3 day-0 support](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support).
  The next phase follows its interleaved MLA token-position sharding and
  LSE-weighted attention merge, while KDA remains sharded by head. It must pass
  independent correctness gates before DCP/EP and kernel performance are accepted.

The older [checkpoint audit](kimi-k3-checkpoint-audit.md),
[CPU oracle notes](kimi-k3-official-cpu-reference.md), and
[tokenizer audit](kimi-k3-tokenizer.md) retain evidence from their stated source
snapshots; they do not establish full released-model accuracy.
