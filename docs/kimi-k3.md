# Kimi-K3 text support

K3 is assembled from reusable KDA, compressed MLA, attention-residual and latent
MoE operations. Admission, prefix reuse, token pages, distributed layouts and
request completion remain framework mechanisms. Code quality and numerical
correctness precede further performance work.

## Current validation

The committed runtime snapshot is `phase8-committed-v1-source`, commit `8d28bd6`,
manifest SHA256 `9e2cb98b3d1c1a5eb131eadf63a1b4f96ecf8cdae4b3bfb85c7a6c5ecc43a373`.
Its complete CPU/Gloo regression passed in `phase8-full-cpu-v9.log` from a clean
Git worktree. It replaces the model-specific config payload with `hf_config`
without changing operator math. Earlier GPU rows belong to its parent runtime, including
`phase8-format-contract-v1-source`; the config refactor still needs GPU startup
confirmation. Validation followed the environment rebuild on 2026-10-28. Queue IDs
below belong to boot `8dcb0ae2-5b59-4693-85f4-f2d9e8ecbd8e`; older queue IDs are
recorded separately in [the validation ledger](validation/kimi-k3-b300.json).

| Gate | Current evidence |
| --- | --- |
| CPU and Gloo regression | 444 passed, 183 skipped, two expected reference semantic disagreements |
| Native/distributed components | 173 passed, job 18; includes ordinary and packed single-expert routing regressions |
| Hybrid eager versus graph | Job 10: four request-slot batches, static/unified pools, reordered requests and compaction; logits and convolution/recurrent state bitwise equal |
| Ordinary model compatibility | Job 22: Qwen3 TP4 and Qwen3-MoE TP4/EP4; four batches each, including padded batch 3; zero logit error |
| Released FP32 TP8/DCP2/EP8 | Job 12: max error 1.2875e-5 against the all-PyTorch TP8 reference; full/chunked error 1.2159e-5; unchanged 1e-4 tolerance passed |
| Released BF16 full/chunked prefill | **Failed**: max error 1.3515625, relative L2 0.09507; same first greedy token |
| Native synthetic HTTP lifecycle | Job 25: serial/overlap × static/unified passed; TP4/DCP2/EP4, native dimensions and graphs enabled |
| Complete mini GSM8K | Job 25: 1282/1319 correct (97.19%), one invalid answer counted wrong, zero length truncation |
| Complete paired GSM8K | Official initialization failed in job 25; resumed as job 33, all 1319 cases retained; comparison pending |
| Performance parity with full SGLang | Not established |

The generic runtime is committed as `d315aa6`, configuration cleanup as
`fd6747b`, and reproducible evaluation tools as `0faeee4`. Configuration cleanup
also produces byte-identical synthetic checkpoint files to the GPU-tested
fixture. Its actual GPU startup remains a separate confirmation.

The earlier Engine mismatch was caused by remote expert routes using E where
the native alignment kernel accepts -1 or [0,E). Both packed execution and the
generic expert partition now use -1. An independent dense-expert regression
fails on the old code and passes after the fix, without changed tolerances or
native library patches. Stream and collective substitutions had not fixed it.

Job 22 stopped after the successful ordinary-model checks because an experimental
cuBLAS precision setting required another backend. Job 25's supported boolean
setting passed preflight. All four combinations of native/portable attention and
default/FP32 partial reductions still failed the original BF16 pointwise gate:
max errors 1.3516/1.2266/1.0703/1.3848 respectively. The first greedy token matched
in every case. Switching attention or disabling reduced-precision reductions
does not eliminate the discrepancy; neither is adopted as a numerical fix.
The saved 93-layer traces and full-vocabulary analysis remain diagnostic evidence,
with task accuracy and performance reported independently.

The complete mini task run generated 141,845 tokens in 3,767.77 evaluation
seconds, excluding loading. This includes prefill, decode, CPU transfer and
state reset, so it is not a fixed-batch decode microbenchmark. Every case ID,
gold answer, response flag and input hash was checked before archiving the full
run. It establishes a one-sided task score, not agreement with SGLang.

Official initialization then failed while JIT-compiling TRTLLM MoE: the same
`SfLayoutDecl.h` was included through two paths and produced duplicate
definitions. No official task result or later profile was produced. The separate
reference environment now includes matching `flashinfer-jit-cache==0.7.1+cu130`
and its declared architecture providers. Its SM103a TRTLLM module loads on CPU;
all nine core package versions and RECORD hashes are unchanged. SGLang source,
FlashInfer core and mini's environment were not patched. Job 33 resumes the
official arm before real-checkpoint HTTP confirmation and the original profile.

## Framework structure

K3 owns its configuration, checkpoint names/shard policies, and decoder
composition in `models/kimi_*.py`. Reusable mechanisms are organized by operation:

| Mechanism | Module |
| --- | --- |
| Causal convolution, delta recurrence, gated RMSNorm | `attention/kda.py` |
| Compressed MLA, optional RoPE, paged metadata | `attention/mla.py` |
| Native absorbed MLA plans | `attention/mla_fi.py` |
| KDA and gated MLA projection layers | `layers/kda.py`, `layers/mla.py` |
| Grouped routing, shared experts, optional latent mixture | `layers/latent_moe.py` |
| SiTU and block attention residuals | `layers/activation.py`, `layers/attn_residual.py` |
| Hybrid batch metadata and real-row execution | `attention/recurrent.py` |
| MLA pages and request-owned recurrent state | `kvcache/mla_pool.py`, `kvcache/recurrent_pool.py` |
| Cache composition, sparse checkpoints, compact byte arena | `kvcache/hybrid_pool.py`, `hybrid_cache.py`, `unified.py` |
| Logical DCP/EP partitions and checkpoint slices | `distributed/layout.py`, `distributed/partition.py` |
| Packed output/LSE exchange and stable merge | `distributed/context.py` |
| E2M1/E8M0 storage, linear and grouped expert execution | `quantization/mxfp4*.py` |
| Streamed aligned weights and derived metadata | `layers/weight.py`, `BaseOP.after_load` |
| Local BPE and segmented XTML text formatting | `tokenizer/tiktoken.py`, `tokenizer/xtml.py` |

`ModelConfig.hf_config` retains each model's normalized HF configuration. Generic
configuration has no `kimi` field or `is_kimi` flag; model recognition stays in
the configuration and checkpoint adapters.

XTML currently supports text messages and exact token decoding, retaining channel
and closing markers. Tools, multimodal inputs and a structured reasoning/response
parser are not implemented. Non-thinking prompts use
`tokenizer.apply_chat_template(..., thinking=False)`.

## Mutable state and prefix reuse

KV is append-only; a recurrent state overwrites its history. A prefix hit is
therefore usable only up to a checkpoint describing exactly that token prefix.
The cache follows the state moves and sparse checkpoint design in the
[SGLang K3 article](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support/):

1. Admission locks a matched prefix and copies its checkpoint into a private
   working slot, including convolution history and FP32 recurrent state.
2. `Context.forward_batch` brackets the complete invocation, including graph
   replay. State lengths commit only after all layers succeed; failure discards
   the partial working state. Ownership uses `(table_idx, uid)` so stale results
   cannot free a reused slot.
3. Aligned prefill boundaries and fixed decode intervals create snapshots.
   Donation transfers a snapshot handle to the radix node without another copy.
4. Mid-edge divergence replays from the nearest checkpoint to the aligned fork.
   Continuing requests adopt shared KV page IDs before duplicates are freed.
5. Global capacity, a per-path cap and LRU bound checkpoint storage, preferring
   branch points. Only pending restores pin shared state; private readers need
   no tree checkpoint after their restore copy is enqueued.

With `--enable-cache-overlap`, the scheduler preserves each completed forward's
lengths and snapshots eligible states before the next forward mutates them.
Processing the previous CPU reply donates these staged handles. Missing optional
snapshot capacity reduces reuse; an advanced state is never labelled with an old
prefix length. Staging is lazy and bounded by one extra state per running request.
Cancellation can find a final in-flight forward, and late replies are filtered by
request ID. Speculative private KV tails are reclaimed on completion.

Copies, compaction and mutable-state updates use the producer stream. Backend and
pool capabilities select stream policy and graph padding; there are no model
family checks in the scheduler. The ordinary TP control path uses reliable
per-receiver PUSH/PULL wakeups and Gloo count coordination, including long idle
periods. Startup detects any owned worker's early exit and cleans up its peers.

`--enable-unified-memory` replaces separately sized MLA/state pools with one byte
arena. State and KV grow from opposite ends. Freeing a middle block moves the
last block of the same kind into its hole and updates indirection. Logical pages
and request slots remain stable; views are resolved after mutations. Admission
budgets private state before KV. Optional checkpoints yield memory under pressure.

## Paged decode and graphs

Native MLA decode uses one physical-page plan for a complete batch and shares it
across compatible layers. Both static and unified storage expose strided views;
no token-history gather is needed. Native stores mask unowned DCP rows without
dynamic boolean indexing.

Each native plan owns its workspace: prefill, eager decode and each captured
batch size have independent lifetimes. Prefill replans current causal query
groups; an event protects asynchronous host staging storage before reuse. Native
planning and execution explicitly bind to the current PyTorch stream. Each plan
reserves 128 MiB plus planning buffers, so graph size selection also bounds memory.

Graphs are explicit: `--cuda-graph-max-bs 8` or
`--cuda-graph-bs 1 2 3 4 5 6 7 8`. Capture is limited by admitted capacity.
Recurrent attention captures exact sizes and uses eager execution for other
batches, since padding would mutate another request's state. Startup temporarily
borrows idle slots, releases them after capture, and replay resolves current
state rows/pages after compaction. Ordinary attention keeps padded graph support.

`--attention-backend recurrent-fi` selects bounded FlashInfer KDA prefill,
indexed FP32-state native decode and absorbed MLA. Current native graph geometry
requires BF16 KDA128 and MLA512/64. Portable shapes retain eager operations;
unsupported native state contracts are not made to work by downcasting state.
The KDA adapter selects the public FP32-state CuTe backend and copies beta only
when a slice is not 16-byte aligned. Device expert execution validates capture
support; a host expert loop cannot run inside a graph.

## Generic distributed layouts

DCP groups live inside TP. `--dcp-size D` divides TP and page size; position `p`
belongs to `p % D`. Each MLA page holds only `page_size / D` compressed rows.
Query/expansion heads are replicated inside the group. One packed all-to-all
carries partial outputs and natural-log LSE to each destination's normal TP head
slice, followed by a stable merge. Empty shards contribute zero; invalid active
values remain visible. KDA stays head-sharded over full TP.

The exchange accepts only output/LSE tensors and has no model or MLA dependency.
A real four-rank ordinary MHA test exercises it, including empty causal shards.
Integrated paged storage currently serves MLA; another attention backend can
reuse the exchange with its own local attention and ownership mapping.

`--ep-size E` assigns contiguous expert IDs to `rank % E`; `TP / E` ranks split
each expert's intermediate channels. Global routing and normalization precede
local selection. The existing scheduler replicates token rows across TP, so each
rank computes owned experts and one TP reduction combines contributions before
latent normalization. This supports ordinary MoE and optional latent MoE without
token dispatch. A distributed-token scheduler would need explicit dispatch/combine.
Unowned weights are filtered before payload reads, and indexed `OPList` preserves
global checkpoint names. Remote expert ID -1 contributes exact zero.

Packed experts hold six pointers per expert into existing packed weights/scales;
they do not stack or duplicate checkpoint weights. Grouped kernels cover
activation and down/mixing. Shared/routed contributions share one FP32 reduction.
`after_load` rebuilds pointer metadata after successful loads and reloads.

## Accuracy and performance boundaries

The independent FP32 reference is source `a4ee224`, all-PyTorch TP8/DCP1/EP1.
Its full/chunked max error is 1.3590e-5. This separates runtime/chunk correctness
from BF16 rounding and subsequent route changes; it is not task accuracy.
The two CPU expected disagreements concern HF top-1 renormalization and low-rank
MLA RMS epsilon. Mini uses the configured epsilon, matching pinned SGLang.

`gsm8k_suite.py` consumes all 1,319 examples from pinned dataset revision
`3101c7d5072418e28b9008a6636bde82a006892c`. Both engines use identical saved token
IDs, zero-shot `thinking=False`, greedy batch 8, 1,024 new tokens, context 4,096,
and the same generation/EOS stop IDs. It scores the last explicit `#### <number>`
and retains invalid/truncated responses. `compare_accuracy.py` rejects partial
runs, duplicate IDs, changed gold/case hashes or mismatched generation protocols.

The 11-prompt full-vocabulary/generation suite is separate from GSM8K. Warm
performance uses 64 raw output tokens, three repeats and batch 1/2/4/8; its boundary
includes model forward, greedy selection and token transfer, excluding loading
and first-use compilation. Raw and stopped token agreement are reported separately.

Historical job 166 measured 6.64/13.17/26.38/52.77 tokens/s for paged eager decode
at batch 1/2/4/8, versus 13.44/22.44/43.48/83.49 with graphs. All 11 diagnostic
prompts had identical eager/graph generations. These older snapshot measurements
are retained in the ledger; they establish neither current accuracy nor parity
with SGLang. Further optimization requires current correctness, a profile and
unchanged paired validation.

## Reproduction

Use compatible native dependencies (`pip install -e '.[native,dev]'`). The rebuilt
B300 environment uses Python 3.11.18, Torch 2.14.1+cu130, Triton 3.8.0,
FlashInfer 0.7.1, TVM FFI 0.1.11, CuTe DSL 4.8.0 and sglang-kernel 0.4.9.
The last distribution retains the public `sgl_kernel` import namespace.
Mini uses Transformers 4.57.3; the independent official environment is separate.

Run GPU work through `gpuq`, with one complete experiment and owned-process
cleanup per queue task. The remote checkout is
`/home/axis/mengze/mini-sglang-k3/mini-sglang`; frozen sources/logs live in sibling
`experiment`. Attested inputs are immutable. No model payload is committed.

```sh
# Pin and verify independent formula sources, then run CPU/Gloo checks.
python tests/models/fetch_kimi_reference.py /tmp/k3-reference
CUDA_VISIBLE_DEVICES= MINISGL_RUN_CUDA_TESTS=0 \
KIMI_OFFICIAL_REFERENCE_PATH=/tmp/k3-reference/modeling_kimi_linear.py \
KIMI_FLA_REFERENCE_PATH=/tmp/k3-reference/fla_kda_naive.py \
KIMI_RUN_GLOO_TESTS=1 MINISGL_RUN_GLOO_TESTS=1 \
KIMI_TOKENIZER_DATA=/data/models/Kimi-K3 OMP_NUM_THREADS=1 PYTHONPATH=python \
python -m pytest -o addopts='' tests/core tests/models tests/test_kimi_cache.py

# Ordinary dense and MoE graph regression, using a fresh synthetic fixture.
CUDA_VISIBLE_DEVICES= PYTHONPATH=python python tests/models/standard_graph_gate.py \
  --fixture /tmp/standard-graphs --create-fixture
gpuq env PYTHONPATH=python OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 tests/models/standard_graph_gate.py \
  --fixture /tmp/standard-graphs --output /path/to/dense.json
# Repeat with --moe and another output path for ordinary EP4.

# Full released FP32 runtime gate; explicitly expensive.
gpuq env PYTHONPATH=python OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=8 tests/validation/released_k3.py \
  --model /data/models/Kimi-K3 --output /path/to/run --dtype float32 --trace \
  --dcp-size 2 --ep-size 8 --mxfp4-backend triton --attention-backend recurrent-fi
```

A bounded native serving configuration is:

```sh
gpuq env OMP_NUM_THREADS=1 python -m minisgl --model /data/models/Kimi-K3 \
  --tp-size 8 --dcp-size 2 --ep-size 8 --dtype bfloat16 --page-size 64 \
  --max-running-requests 4 --max-cached-states 8 --num-pages 2048 \
  --max-seq-len-override 4096 --max-extend-length 1024 \
  --attention-backend recurrent-fi --mxfp4-backend triton --enable-unified-memory
```

The queue task must own the server throughout its lifetime. Size working state,
optional snapshots and KV together; the example bounds capacity, not throughput.

## Source provenance

Development starts at `WeiMengze666/mini-sglang`, branch `kimi-k3-eager-text`,
commit `d477af124de2b64a4709654c6144ffcac92e2079`. Official SGLang is clean pinned
`release/v0.5.22`, commit `baa9befa381511cd4d56f83980951e5a45fffc6d`, verified again
on the rebuilt server. CPU formula sources are HF
`f831ab66814297da540d832a5235f8e904f29d06` and FLA
`14bef3d9ec98ee359c9a9c7860795d91675dd091`.

The [checkpoint audit](kimi-k3-checkpoint-audit.md),
[CPU oracle notes](kimi-k3-official-cpu-reference.md), and
[tokenizer audit](kimi-k3-tokenizer.md) retain their stated snapshot boundaries.
The ledger preserves earlier diagnostics and failures rather than treating them
as current acceptance.
