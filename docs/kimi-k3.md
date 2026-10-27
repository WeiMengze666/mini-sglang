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
| Independent DCP and expert ownership layouts | `distributed/layout.py`, `distributed/partition.py` |
| Packed output/LSE exchange and stable merge | `distributed/context.py` |
| E2M1/E8M0 decoding, selection and fused linear execution | `quantization/mxfp4.py`, `quantization/compressed_tensors.py`, `quantization/mxfp4_kernel.py` |
| Safetensors index validation and shard iteration | `models/checkpoint.py` |
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

Evidence collected on 2026-10-26/27:

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
| Real four-rank NCCL DCP2, EP2, DCP2+EP2, DCP4+EP4 | Passed, gpuq job 27: full/chunked prefill, empty KV shards, prefix restore, static/unified pools |
| Released FP32 TP8+DCP2+EP8 versus TP8 reference | Passed, gpuq job 27: max absolute difference 1.4067e-5, relative L2 1.0731e-6; chunked max absolute difference 1.8120e-5 |
| Weight-only fused MXFP4 GEMV/GEMM | 16 CUDA tests passed, gpuq job 29; FP32/BF16/FP16, bias, tails, codebook and scale blocks |
| Absorbed MLA decode and natural-log LSE | 4 CUDA tests passed, gpuq job 31; head padding, empty KV shards, plan reuse |
| Native KDA FP32-state adapter | 6 CUDA tests passed, gpuq job 36; bounded prefill and FP32 decode, nonzero state and continuation |
| Absorbed MLA causal/interleaved prefill | 8 CUDA tests passed, gpuq job 45, including empty causal rows and plan reuse |
| Full released FP32 + fused MXFP4 + DCP2/EP8 | Passed, gpuq job 42: max absolute difference 1.3351e-5 vs FP32 reference; full/chunked max 1.1444e-5 |
| Full released BF16 + native kernels + unified pool | gpuq job 42: all 8 greedy IDs match; warm decode about 0.255 s/token; full/chunked pointwise logits still fail (max 1.25) |
| Packed TP4/DCP2/EP4 HTTP lifecycle | Passed on retry, gpuq job 40, static and unified pools; previous first-request timeout was not reproduced |
| Existing stacked MoE EP1/EP2/EP4 | Passed, gpuq job 65: four-card NCCL, global routing, expert tensor splits and zero remote contributions |
| Combined native kernel regression | 30 passed, gpuq job 65: MXFP4, FP32-state KDA and absorbed MLA |
| Hybrid parallel regression after generic checkpoint refactor | 4 passed, gpuq job 65, including four-card NCCL and both pool modes |
| Generic DCP/EP/checkpoint CPU regression | 345 passed, 36 skipped, 2 expected semantic disagreements; standalone MHA DCP Gloo and ordinary MoE checkpoint shards included |
| Graphs, overlap and performance parity | Not accepted |

The short generation is not an accuracy benchmark. The FP32 full-model comparison
isolates state/chunk behavior from BF16 shape-sensitive rounding and downstream
route changes; the BF16 pointwise failure remains visible. Per-layer traces and
full-vocabulary diagnostics are saved before numerical assertions. The timings below are single-request latency observations. No throughput
comparison is accepted. Logs and compact reports are under the remote experiment
directory (`released-fp32-trace`, `serving-lifecycle`, `short-reference-comparison.json`).

## Distributed layouts and optional kernels

`--dcp-size D` divides both TP size and token page size. Within each contiguous
DCP group, MLA query/KV expansion heads are replicated and token position `p`
belongs to rank `p % D`. Each physical token page stores only `page_size / D`
compressed rows. The scheduler and radix tree keep their ordinary logical pages.
Each shard computes a partial output and natural-log LSE. One packed all-to-all
returns each destination's normal TP head slice, followed by a stable LSE merge.
All-empty partitions contribute zero. KDA continues to shard heads over full TP.

`--ep-size E` assigns contiguous global expert IDs to `rank % E`; only `TP / E`
ranks split an expert's intermediate channels. Unowned checkpoint payloads are
filtered before reading. Indexed `OPList` retains global checkpoint names while
storing only local experts. The current scheduler replicates tokens across TP,
so owned experts contribute to one full-TP routed-mixture reduction before latent
normalization. This supports expert ownership without a token all-to-all or a
separate scheduler. Router, shared experts and latent projections retain their
usual TP contracts. Only active expert IDs are visited during eager execution.

`--mxfp4-backend triton` fuses weight decoding with GEMV/GEMM; the default `torch`
backend retains the bounded-row reference. The fused path keeps checkpoint bytes
and scales resident and never materializes a decoded weight matrix. FP32 GEMM
uses Triton's `tf32x3` input precision; arithmetic equivalence is checked separately
from BF16 model accuracy.

`--attention-backend recurrent-fi` enables indexed FP32-state FlashInfer bounded
KDA prefill, a small native Triton FP32-state decode, and absorbed FlashInfer MLA
decode where the dimensions are supported. Other shapes retain the portable
operations. The installed public FI small-batch decode/unbounded-prefill routes
do not preserve this runtime's FP32 state contract; the adapter does not downcast
state to make those routes run. Multi-query MLA groups interleaved queries into
ordinary causal subsequences; unsupported shapes retain the reference path. Compatible FlashInfer and Triton installations are required for
these explicit kernel options. The `native` install extra declares the tested
FlashInfer/Triton API generation; the B300 run used Torch 2.14.1+cu130,
FlashInfer 0.7.1 and Triton 3.8.0. CPU/reference paths do not import the native
kernels. CUDA graphs and overlap remain disabled.

## Reusing distributed mechanisms

`Partition` defines contiguous logical ranges and checkpoint slices. Both the
ordinary stacked MoE loader and the K3 native-name loader use `ParallelLayout`;
model adapters map their own parameter names to these partitions. Packed readers
translate logical input slices into byte and scale-block slices before loading.

EP applies to the existing `MoELayer`/fused backend and the optional-latent expert
composition. Global routing and probability normalization happen before local
expert selection. In the replicated-token scheduler, each rank evaluates only
owned experts, then the ordinary TP reduction combines contributions. A reserved
local routing ID denotes a remote expert and must contribute exact zero. Both
expert execution and final mixing mask remote/padding buckets, including the
alignment kernel's negative expert IDs. This layout does not require token dispatch; adding a distributed-token scheduler
would require an explicit dispatch/combine implementation.

DCP communication accepts only partial outputs and natural-log LSE; it has no
model or MLA dependency. A standalone ordinary multi-head attention test exercises
its real four-rank Gloo exchange, including empty causal shards. The integrated
paged storage currently serves MLA. Another attention implementation can reuse
the exchange while providing its own local attention and KV ownership mapping.

## Current latency evidence

On the same 8×B300, TP8/DCP2/EP8, a 27-token prompt and 16 output tokens, official
SGLang `baa9bef` measured about 0.1165 s per decode token with graphs/overlap off,
and about 0.00915 s with both on (three warmed runs, gpuq job 45). These are
estimates from paired synchronous `Engine.generate(1)` and `generate(16)` calls,
including scheduler and logprob work. Mini's earlier approximately 0.255 s/token
covers model forward, greedy selection and token transfer, with a shorter output.
The measurement boundaries differ; this establishes a substantial remaining gap,
not a parity result. Broader mini warm measurements and profiling are collected
by `tests/validation/checkpoint_suite.py` before selecting further optimizations.

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

A bounded eight-card server configuration with the native backends is:

```sh
gpuq env OMP_NUM_THREADS=1 python -m minisgl --model /data/models/Kimi-K3 \
  --tp-size 8 --dcp-size 2 --ep-size 8 --dtype bfloat16 --page-size 64 \
  --max-running-requests 4 --max-cached-states 8 --num-pages 2048 \
  --max-seq-len-override 4096 --max-extend-length 1024 \
  --attention-backend recurrent-fi --mxfp4-backend triton --enable-unified-memory
```

The queue job must own the server for its entire lifetime. Adjust request/state
and page capacity together: the unified arena pays for private working state and
optional snapshots as well as compressed token pages. The example deliberately
bounds the serving capacity; it is not a throughput-tuned configuration.

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
  The runtime follows its interleaved MLA token-position sharding and
  LSE-weighted attention merge, while KDA remains sharded by head. Distributed
  correctness and kernel performance retain separate acceptance gates.

The older [checkpoint audit](kimi-k3-checkpoint-audit.md),
[CPU oracle notes](kimi-k3-official-cpu-reference.md), and
[tokenizer audit](kimi-k3-tokenizer.md) retain evidence from their stated source
snapshots; they do not establish full released-model accuracy.
