# Kimi-K3 text support

K3 is assembled from reusable KDA, compressed MLA, attention-residual and latent
MoE operations. Admission, prefix reuse, token pages, distributed layouts and
request completion remain framework mechanisms. Code quality and numerical
correctness precede further performance work.

## Current validation

The attention composition at `5b810ac` explicitly combines recurrent and MLA
backends. KDA keeps its FlashInfer default; the MLA executor determines its own
paged-cache capability. Commit `46ffbab` adds optional generic stable integer
route grouping without changing the default. Its shipping-tree CPU/Gloo
regression passed 493 checks, with 187 skips and two existing expected semantic
disagreements. Its four GPU Graph tests and real local Marlin replay passed in
job 189. No experimental precision kernels have been adopted. Released-weight
GPU evidence below belongs to the recorded earlier snapshots.

The accepted numerical baseline is `phase8-committed-v1-source`, commit `8d28bd6`,
manifest SHA256 `9e2cb98b3d1c1a5eb131eadf63a1b4f96ecf8cdae4b3bfb85c7a6c5ecc43a373`.
Its complete CPU/Gloo regression passed in `phase8-full-cpu-v9.log` from a clean
Git worktree. It replaces the model-specific config payload with `hf_config`
without changing operator math. Earlier GPU rows belong to its parent runtime,
including `phase8-format-contract-v1-source`. Job 36 confirms the refactor on
released weights with graph capture, all 11 prompts and warm batch 1/2/4/8.
Validation followed the environment rebuild on 2026-10-28. Queue IDs
below belong to boot `8dcb0ae2-5b59-4693-85f4-f2d9e8ecbd8e`; older queue IDs are
recorded separately in [the validation ledger](validation/kimi-k3-b300.json).

| Gate | Current evidence |
| --- | --- |
| CPU and Gloo regression | 493 passed, 187 skipped, two expected reference semantic disagreements; generic stable grouping increment at `46ffbab` |
| Native/distributed components | 173 passed, job 18; includes ordinary and packed single-expert routing regressions |
| Hybrid eager versus graph | Job 10: four request-slot batches, static/unified pools, reordered requests and compaction; logits and convolution/recurrent state bitwise equal |
| Ordinary model compatibility | Job 22: Qwen3 TP4 and Qwen3-MoE TP4/EP4; four batches each, including padded batch 3; zero logit error |
| Released FP32 TP8/DCP2/EP8 | Job 12: max error 1.2875e-5 against the all-PyTorch TP8 reference; full/chunked error 1.2159e-5; unchanged 1e-4 tolerance passed |
| Released BF16 full/chunked prefill | **Failed**: max error 1.3515625, relative L2 0.09507; same first greedy token |
| Native synthetic HTTP lifecycle | Job 25: serial/overlap × static/unified passed; TP4/DCP2/EP4, native dimensions and graphs enabled |
| Complete mini GSM8K | Job 25: 1282/1319 correct (97.19%), one invalid answer counted wrong, zero length truncation |
| Current generic-config Engine | Job 36: 11 full/chunked prompts, 64-token generations and warm batch 1/2/4/8 completed; all first tokens match across full/chunked inputs |
| Complete paired GSM8K | Job 36: mini 1282/1319 (97.19%), official 1279/1319 (96.97%); identical saved cases and generation protocol |
| Released-checkpoint HTTP | Job 41: serial/overlap × static/unified all passed on the committed runtime; prefix reuse, fork, concurrency, cancellation, slot reuse and streaming budgets verified |
| MXFP4 matrix optimization trial | Job 41: rejected; 1280/1319 versus the required baseline 1282/1319, despite 12.75% geometric-mean decode speedup; no runtime change adopted |
| Performance parity with full SGLang | Not established |

The generic runtime is committed as `d315aa6`, configuration cleanup as
`fd6747b`, and reproducible evaluation tools as `0faeee4`. Configuration cleanup
also produces byte-identical synthetic checkpoint files to the GPU-tested
fixture. Job 36 confirms startup and execution of the committed generic payload.

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
run. Job 36 completed the official arm: 1279/1319 correct, one invalid response
and no truncation. Mini scored three more examples on this fixed protocol;
1309 valid final answers and 745 complete raw token sequences matched. These
counts establish neither strict token alignment nor statistical superiority.

Official initialization then failed while JIT-compiling TRTLLM MoE: the same
`SfLayoutDecl.h` was included through two paths and produced duplicate
definitions. Job 25 produced no official task result or later profile. The separate
reference environment now includes matching `flashinfer-jit-cache==0.7.1+cu130`
and its declared architecture providers. Its SM103a TRTLLM module loads on CPU;
all nine core package versions and RECORD hashes are unchanged. SGLang source,
FlashInfer core and mini's environment were not patched. Job 33 crossed the
previous compilation failure, then made no progress for 549 seconds during
TRTLLM MoE tuning. All GPUs were idle; an owned rank-zero
faulthandler trace locates its wait in the autotuner's distributed timing
`all_reduce`. The precise root cause remains unestablished. Core dumps were
disabled before manually aborting the owned process; SGLang cleaned its peers.

Job 36 completed both diagnostic suites, profiling and the full official task
arm. The public `flashinfer_autotune_skip_ops` option skips
only `flashinfer::trtllm_fp4_block_scale_moe`, which uses the official heuristic
fallback. This reference policy is recorded explicitly and cannot establish
default-tuned SGLang performance. Neither source nor precision was patched.

Job 36 expired its 180-second HTTP startup budget during first graph capture
after a 128-second weight load; no request assertion was reached. Job 41 passed
all four serial/overlap × static/unified released-checkpoint modes with a
600-second cold-start wait. The 60-second request timeout, first-use 180-second
timeout, all 8-token assertions and saved 468-token workload were unchanged.
Prefix repeat/fork, four concurrent clients, cancellation, slot reuse, zero and
invalid budgets, and streaming budgets 1/2/5 passed. All four modes generated
`甲乙丙丁戊己庚辛壬` for the shared 8-token check.

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
| Optional canonical integer expert grouping | `moe/routing.py`, `moe_align_block_size(..., stable=True)` |
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

The 11-prompt full-vocabulary/generation suite is separate from GSM8K. Warm mini
performance uses 64 raw output tokens, three repeats and batch 1/2/4/8; its
boundary includes model forward, greedy selection and token transfer,
excluding loading and first-use compilation. The optional official
`--benchmark-batch-sizes 1 2 4 8` gate records the same fixed-length workload
through synchronous `Engine.generate`, including prefill, scheduler and IPC,
without logprobs. These timing boundaries differ and are reported separately.
Raw and stopped token agreement are also reported separately.

Job 36 measured the following warm baselines. The columns have different timing
boundaries; they do not establish a directly comparable decode speed ratio.

| Batch | Mini decode tokens/s | Official whole-generation tokens/s |
| --- | ---: | ---: |
| 1 | 13.45 | 83.53 |
| 2 | 24.75 | 166.63 |
| 4 | 47.99 | 333.38 |
| 8 | 91.68 | 616.60 |

Mini duplicate requests and the singleton reference matched through the stop
in every batch. Across the two engines, all 11 first tokens matched and eight
generations matched through the stop. Python, reasoning and explanation wording
differed; raw 64-token sequences matched on none of the prompts.

These official default runs also use a different MoE activation format: the
pinned SM100 runner quantizes activations to MXFP8, while mini retains BF16.
The official public `--flashinfer-mxfp4-moe-precision bf16` option supplies
BF16 inputs with no activation scale. That setting was not used for the
recorded default timings or GSM8K reference; same-precision raw-logit and
performance measurements remain required.

The trace separates prefill from graph replay. MoE gate/up and down account for
26.53% of baseline decode kernel time. Job 41 tests one shared MXFP4 matrix helper
used by dense and grouped expert layers; it adds no model-specific branch. Its
71 module checks passed at unchanged tolerances. Same-job batch 1/2/4/8 decode
measured 15.22/27.73/54.09/103.82 tokens/s, improving 13.06/11.93/12.74/13.27%
(12.75% geometric mean). All 11 first tokens matched the baseline; eight
continuations matched through stop. BF16 bitwise equivalence is not claimed.

The complete candidate task run finished with 1280/1319 correct (97.04%), no
invalid answers and no truncation. Relative to the production baseline, four
examples changed from correct to wrong (73/322/951/1288) and two from wrong to
correct (425/675). The predeclared minimum was 1282 correct, so the trial was
rejected. This is a failed acceptance rule, not evidence of statistical
degradation. The shared matrix helper and its test changes were not merged.
Production throughput remains the baseline reported above.

Candidate HTTP and the additional strict FP32 checkpoint gate were not run,
because complete task acceptance failed first. No follow-up GPU task was
submitted. The isolated candidate passed 444 CPU/Gloo checks (188 skips and two
existing expected disagreements) and 71 module checks, which do not override
the full-task gate. Its FP32/FP16 fallback keeps the mathematical expression,
but CPU compilation produced different instruction layouts; no binary
equivalence or additional whole-checkpoint FP32 result is claimed. The saved
full responses, comparisons, source hashes and rejection record preserve the
experiment for further analysis without changing the accepted runtime.

Historical job 166 measured 6.64/13.17/26.38/52.77 tokens/s for paged eager decode
at batch 1/2/4/8, versus 13.44/22.44/43.48/83.49 with graphs. All 11 diagnostic
prompts had identical eager/graph generations. These older snapshot measurements
are retained in the ledger; they establish neither current accuracy nor parity
with SGLang. Further optimization requires current correctness, a profile and
unchanged paired validation.

## Current BF16 investigation

The isolated fused KDA decode candidate combines convolution, SiLU and the
FP32 delta update without mutating borrowed prefix state. GPU job 149 passed
six independent FP64, batch-isolation and graph-replay cases, together with
24 normalization cases (31 tests including the CPU fallback). This component
result does not establish consistency with the existing prefill path.

Job 153 failed the complete-model check: its first complete-model full/chunk check
had 35,607 of 163,840 logits outside the unchanged BF16 tolerance, with maximum
absolute difference 0.912109375. Its independent FP32 checkpoint check passed,
but graph, model performance and GSM8K stages were skipped. The fused decode
API therefore remains isolated and is not added to the production engine.
The next primary candidate uses the batch's declared phase to select the
algorithm. The failed integration chose decode whenever every request contained
one token, which also covers a singleton prefill tail. Prefill now retains
`layer.update`; only an explicit decode batch may select `layer.decode`.
Unknown-phase standalone callers retain the portable update. This changes no
arithmetic kernel. Six phase/capability regressions passed, and executing the
actual old function body reproduced failures for singleton prefill and missing
phase. The complete CPU regression passed 484 tests, with 341 skips and two
expected semantic disagreements. Job 160 then passed 43 GPU components, all
eleven complete-model full/chunk comparisons with exact full-vocabulary equality,
the independent FP32 gate and four dense/MoE/static/unified graph gates. Complete
11-by-16 raw captures are preserved. These results establish candidate basis
checks; independent SGLang BF16, whole-model speed and task-quality gates remain
open, and production runtime is unchanged.

A separate mathematical control shares convolution arithmetic and explicit
FP32 rounding across phases. It passed 478 CPU regressions, 23 cross-compilations
and two twelve-case host-interpreter checks, including an independent FP64
oracle with quantized FP32 buffers. Those host results do not establish GPU
rounding, model alignment or speed. This larger control remains isolated.


Job 135 found an unresolved reference-calibration issue. The same 27 input
tokens in the pinned official runner, with one versus sixteen warmup steps,
changed 161,192 of 163,840 first-position logits. Maximum error was 1.30859375;
79,396 entries exceeded the original `rtol=0.05, atol=0.1`, while the greedy token
was unchanged. Historical cross-engine differences remain recorded below, but
they do not isolate mini's numerical error until the reference controls pass.

All three mini arms in that job stopped before model loading: diagnostic
imports initialized CUDA before the Engine selected its device. The startup
assertion remains intact. Repaired diagnostics defer vendor imports and passed
15 CPU checks, including CUDA-visible import guards, stored-callable snapshots,
fresh-state initialization, independent warmup length and negative controls.
Job 143 completed the repaired collectors and generic-source integration:
334 GPU components passed, all eleven complete-model full/chunk comparisons
were exact, and the independent FP32 checkpoint gate passed. Initial recurrent
state was finite and zero on all eight official ranks, including the legacy
warmup protocol. Eighteen common first-layer boundaries also matched bitwise
across official warmup controls. These observations do not support a first-layer
state-leak explanation. Its complete official sweep was rejected because the
controller requested only one input length; the single-case traces remain valid.

Job 146 repaired that input schedule and captured two independently loaded
official runs with identical source, configuration, sixteen-step warmup and all
eleven prompts. Their complete 176-position self-comparison still failed:
9,235,864 raw-logit violations, with 169 of 176 greedy tokens matching. The first
six captured readouts matched bitwise on every rank; layer 3's attention readout
first changed 255 of 193,536 entries, with maximum difference 0.001953125.
That boundary precedes the first MLA layer's computation and does not identify
whether the cause lies in the preceding feed-forward computation, residual
aggregation or communication. Cross-engine counts remain diagnostic until
this reference control is calibrated.

The original four-warp generic persistent-expert candidate passed 44 independent
GPU tests and sixteen bitwise baseline/chunk/graph cases in job 149. Its BF16
component decode geometric-mean speedup was 1.269, with minimum BF16 speedup
0.991 across the tested decode/prefill cases. Eight-warp variants regressed
other shapes and were rejected. None of these measurements is a whole-model
speedup or a production adoption.

The evidence archives through jobs 143 and 149 contain 906 and 590 locally
verified files. The latter inventory explicitly retains sixteen large complete
layer traces remotely and includes original first-four-layer excerpts with
provenance. Complete raw-logit sweeps and all failures are preserved. The first
deterministic-reference attempt in job 153 failed before model loading because
its DCP decode backend was unsupported. Job 157 reached actual `resolve_once()` and configured deterministic NCCL, but
its expected prefill-backend assertion failed before model loading. Job 160
explicitly sets both supported phase backends and saves configuration values
before checking them. It combines complete repeated references, a normal-run
observer-transparency control and the phase-aware candidate in one GPU task.
Argument parsing alone is not configuration validation. No new runtime,
independent BF16 or whole-model performance acceptance is recorded here.

Job 163 completed effective-view configuration and two independent complete
official runs, including all-layer zero-state checks. The unchanged raw gate
still rejected official self-repetition: 9,704,318 pointwise violations and
169/176 greedy matches. All eight ranks first differed in the incoming tensor
at `003.attention_readout.input`: 440/193,536 changed entries, maximum
0.00048828125. The eight earlier captured tensors on each rank matched bitwise.
The following readout changed 576 entries with maximum 0.00390625. This narrows
the unobserved region to the preceding FFN, its communication and residual
writeback; it does not identify a specific faulty operator or implicate MLA
computation that has not yet executed.

The phase-aware candidate's complete comparison against this reference failed
with 9,785,819 pointwise violations and 166/176 greedy matches. Independent BF16
alignment is not accepted, and the reference instability prevents attribution
of those cross-engine differences to mini alone. The normal-run observer
control, model performance and complete GSM8K stages were skipped. No runtime
candidate was adopted. A one-prompt FFN/router/residual snapshot diagnostic has
passed CPU import, identity, snapshot and official-CLI checks. The user has
authorized continued server use for the remaining localization and final
acceptance after this completed round.

Job 172 repeated the single 27-token prompt with sixteen raw positions. Both
runs and all initial-state checks completed, but raw repetition still failed:
1,269,834 violations, maximum 11.75; all sixteen greedy tokens matched. On all
eight ranks, layer 2's FFN input and pending residual matched bitwise, while its
return changed 107/193,536 BF16 elements (maximum 0.00048828125). Captured links
from readout to FFN, FFN to finish and finish to the next readout were bitwise.
This places the first observed difference inside the FFN before finish. The
independent gate hook was bypassed by the fused front; no router stability is
inferred. Job 174 will observe the actual fused producers and reduction path.
Its stacked import, tuple identity, out-buffer snapshots and official CLI passed
CPU checks; uninitialized output buffers are omitted before their producers.

Job 174 observed the actual fused branch. Captured routed inputs, top-k IDs,
mixing weights and shared outputs matched bitwise on every rank. A few local
routed outputs already differed before communication: six elements in layer 1
on rank 7, and one/two/two elements in layer 2 on ranks 2/3/6. The first local
changes need not survive BF16 collective rounding; after layer 2's sum, one
latent element changed and the FFN return changed 191 elements. This places the
remaining region inside the routed executor without identifying a particular
Marlin GEMM, activation or mixture kernel. It does not establish a communication
bug. Raw repetition still failed with 1,353,818 violations and 15/16 greedy
matches; maximum raw error was 13.25.

A frozen local replay contains the real rank-3/layer-2 activations, routes and
112 native local experts (1,965,979,589 bytes), avoiding another full-model load.
Actual CPU import/signature and material checks passed. Job 178 stopped during
JIT setup because its submission omitted the existing `ninja` directory from
`PATH`; it produced no numerical report. The environment correction reuses the
same frozen replay and data, with the failure retained separately.

Job 180 completed that unchanged replay with the corrected build environment.
Repeated local outputs were not bitwise stable over 32 calls (maximum difference
0.0000152587890625), and at least one call matched captured official endpoint B
bitwise. Both observed GEMMs actually used `use_atomic_add=False` and
`use_fp32_reduce=True`. After excluding unused and remote routing entries, all
77 local routes remained the same set while their order changed. This is a
reproduced local instability. Job 182 fixed two captured plans: each repeated
32 times bitwise, but their outputs matched each other, so that control did not
establish a causal order effect. Job 183 instead froze two plans with different
captured endpoints. Each repeated 32 times bitwise and reproduced its endpoint
exactly; changing the plan changed 149 output elements (maximum
0.0000152587890625). The result held with native alignment work and allocations
retained before using the fixed plan. This establishes an order effect for this
real local workload, with unchanged GEMMs, precision, inputs and expert bytes.
It does not establish full-model repeatability or cross-engine alignment.

The user approved a deterministic official reference with fixed route order,
unchanged official GEMMs and precision, subject to complete calibration. Stock
reference instability remains recorded separately. Raw-logit tolerances
(`rtol=0.05`, `atol=0.1`) and exact greedy matching remain unchanged. A generic
optional stable grouping candidate, built on the phase-aware persistent source,
passed 524 CPU/Gloo tests (345 skips, two expected failures). Its independent
bucket oracle covers integer dtypes, empty and noncontiguous inputs, remote
routes and fully initialized padding. Job 189 then passed all four CUDA Graph
checks (12.65 seconds) and the real local Marlin replay: 32 stable-route calls
were bitwise equal and matched the independent integer-plan oracle, while 32
native-route calls remained unstable. Complete reference calibration remains
pending. Job 190 stopped before model loading because its configuration stage
hid CUDA, which public effective resolution requires for device discovery.
That failure is retained. Corrected job 191 completed all three full sweeps and
passed all initial-state checks, but reference calibration failed: self-repeat
had 9,592,964 raw violations with 171/176 greedy matches; observer transparency
had 9,641,738 violations with 169/176 greedy matches. Every rank's earliest
changed captured input was the layer-3 attention input (146/193,536 elements,
maximum 0.0009765625). The separate CPU audit rejects this reference, including
the changed captured depth boundaries. A subsequent source and actual-spawn CPU
control found a capture-wrapper defect: `runpy(..., run_name="__main__")` changed
the worker main path to the inner unpatched capture script. The parent-installed
routing override was lost. Job 191 therefore does not test full-model stable
routing. The corrected wrapper keeps the outer spawn entry, installs in every
worker, and requires actual planner-call counts plus GEMM metadata pointer
aliases on all ranks. CPU spawn controls reproduced the old negative case and
passed the new positive case, including Python's real main-path preparation.
Job 197 then passed complete calibration with that corrected entry. All three
11-prompt, 16-position sweeps matched bitwise across the full 163,840-token
vocabulary: zero raw error and 176/176 greedy matches for both self-repeat and
observer transparency. All eight ranks' captured layer boundaries matched
bitwise, all initial states were zero, and all 24 arm/rank coverage records
passed independent route-plan and actual GEMM-pointer checks (92 measured
planner calls each). Official GEMMs, activation precision and sources remain
unchanged. This canonical reference is now calibrated under the user-approved
contract; stock instability is retained separately.

Against this calibrated reference, source7 still fails the original raw gate:
9,456,592 violations, 170/176 greedy matches and maximum error 11.2734375.
Strict Mini BF16 alignment is not established. The independent combined
candidate passed 87 component checks in 25.21 seconds, full/chunked execution
for all 11 prompts, the independent FP32 reference (maximum error about
`1.37e-5`, original `1e-4` gate), and all four dense/MoE/hybrid graph paths.
Its complete raw capture is bitwise equal to source7, so persistent expert work
scheduling introduces no raw-logit difference in these 176 positions.

Matched native decode throughput for batches 1/2/4/8 was
15.91/31.58/62.80/122.33 tokens/s, versus
13.47/24.78/48.01/91.68 for the baseline: geometric mean gain 27.3%.
Each engine's timed output matched its ordinary execution token controls.
Matched full SGLang BF16/Marlin achieved
70.38/134.61/239.75/404.00 tokens/s; the candidate reaches only
22.6–30.3% of it. This passes the candidate improvement gate but remains far
from the requested full-SGLang performance target. Complete 1,319-question
GSM8K evaluation is running. One truncated response has already been observed,
so the candidate cannot pass the original zero-truncation gate; the complete
run is retained for diagnosis. Partial accuracy is not task-quality acceptance.
The optimized runtime remains isolated pending strict alignment and complete
quality gates.

The optional generic stable integer grouping utility also passed 493 CPU/Gloo
tests on the shipping tree (187 skips, two expected failures; 31.70 seconds).
It is available through `moe_align_block_size(..., stable=True)`; the default
route grouping remains unchanged. This utility is independent of the unshipped
optimized runtime candidate. Strict BF16 acceptance remains pending.

A separate performance composition adds only the already component-tested
four-warp persistent expert work grid to the phase-aware source. It passed
484 CPU/Gloo checks, with 341 skips and two expected disagreements; its full
manifest is `e6ef62bbbd47679c58a97c5705f7f13c14f3f5975feb30123b2c1d7eb26b6fef`.
Its seven actual validation entry points passed CPU startup checks. Queued job
195 was removed before starting and replaced by job 197, one serial allocation:
spawn-corrected complete reference calibration, independent CPU audit, then the
unchanged frozen candidate controller. It tests components, full/chunked model,
independent FP32 reference, bitwise source7 control, four graph paths and paired
native throughput. Each reference arm must record all eight workers' actual
planner calls and GEMM metadata aliases; parent installation alone is
insufficient. The corrected reference is calibrated. Timing and full-task
quality remain separate from strict cross-engine acceptance; the optimized
runtime has not been adopted.

Frozen operator traces isolate batch-dependent rounding in dense projections,
absorbed MLA projections and routed-expert RMSNorm. Generic fixed-layout kernels
and a sequential FP32 recurrence eliminated full/chunk logit differences for one
27-token, 93-layer run split into 13+14 tokens (job 86). This bounded result does
not establish strict agreement with SGLang.

The expanded test then failed unchanged pointwise tolerance on seven of eleven
prompts, including singleton tails and long prefills. Paired decode also slowed
in every batch. The candidate was rejected and its complete GSM8K run stopped;
partial task results are not accuracy acceptance. Exact-output MXFP4 bit-decoding
and larger expert tiles also failed paired performance tests and were rejected.

Job 92 added a generic fixed-layout causal MLA prefill kernel. Its 196 GPU
component checks and four dense/MoE/hybrid graph gates passed. The complete
93-layer BF16 model again matched exactly for 13+14 tokens, and FP32 retained
the original tolerance. Splitting the same prompt into 13+13+1 tokens still
failed, with a maximum logit difference of 0.46875. This candidate was not adopted.
An additional CPU comparison of its saved full-vocabulary FP32 output against
the independent all-Torch reference passed at the original `1e-4` tolerance:
maximum error `1.2874603271484375e-5`, with identical greedy tokens.

All eight ranks of the subsequent operator trace (job 96) first diverged at
layer 0's attention output all-reduce: identical recorded inputs produced a
maximum BF16 difference of 0.00048828125 on the singleton tail. The preceding
two 13-token chunks agreed. This localizes that failure to the collective;
the subsequent controls below distinguish low-precision and FP32 reductions.

Jobs 103 and 109 independently reproduced shape-dependent NCCL accumulation:
six of twelve BF16 comparisons changed, while FP32 accumulation followed by
BF16 rounding matched in all twelve. FP32 NCCL sums themselves still differed
by up to 1.1920928955078125e-7. Promoting only BF16 reductions therefore did
not fix the complete model. Its first remaining operator difference occurred
after the FP32 routed/shared-expert collective on all eight ranks.

A generic optional FlashInfer FP32 one-shot collective, composed with the
fixed-layout arithmetic candidate, passed all eleven released-model full versus
chunked prefill cases in job 109: every full-vocabulary logit matched exactly,
and every first greedy token matched. This includes the 13+13+1 singleton tail
and prompts through 1,950 tokens. The actual communicator factory, CUDA Graph
replay and teardown also passed; 24 primitive full/chunk comparisons matched.
These results belong to frozen manifest
8482dad719376d00b211deddc7b4f3099af06dcb67cb5b21b69a20c9022fa27f.
They do not yet establish independent BF16 alignment, complete task quality or
whole-model performance, and the candidate has not been adopted.

The attempted official FlashInfer BF16 reference produced no capture: its
workers reported an unavailable MXFP4 × BF16 SiTU kernel, although the benchmark
parent returned zero. Artifact inventory checks now reject this false success.
Job 113 subsequently captured all eleven prompts at sixteen teacher-forced
positions using the official Marlin BF16 backend. The unchanged raw-logit gate
failed: 170 of 176 greedy tokens matched, with 9,424,600 pointwise violations.
Its complete matched-boundary native benchmark measured
70.45/134.41/245.44/418.46 tokens/s at batch 1/2/4/8. The fixed-layout mini
candidate measured 10.38/20.47/40.43/78.68, compared with the production baseline's
13.46/24.77/48.02/91.71 under the same mini timer. Exact ordinary/native token
parity passed for all twelve 64-token runs. Both alignment and performance gates
therefore rejected this candidate. Its task evaluation was stopped at eight
completed cases; no complete GSM8K or HTTP acceptance is claimed.

The candidate retained FP32 correctness: maximum full-vocabulary error
1.1086463928222656e-5 against the independent all-Torch reference at the original
1e-4 tolerance. Its 213 component checks and four dense/MoE/hybrid graph gates
also passed. Existing official default-precision timings remain separate MXFP8
activation measurements.

Job 120's observational stage capture establishes the first independent BF16
semantic difference. Layer 0 has identical inputs, weights and epsilon, but
51,303 of 193,536 normalized values differ, with maximum error 0.015625. Mini
matches the FP64 formula with rounding before weight multiplication exactly;
SGLang matches rounding after multiplication exactly. Correcting later stages
still requires whole-model evidence; this observation is not an alignment pass.
Job 123 completed both eleven-prompt, sixteen-step normalization ablations with
graphs disabled. Rounding after multiplication matched 169/176 greedy positions;
additionally rounding each local expert mixture matched 173/176. Both retained
over nine million raw-logit tolerance violations, so neither was adopted.

Job 131 confirmed that the corrected layer-0 attention input is bitwise equal
across implementations; its observational runs also reproduced the earlier
complete capture's first logits exactly. The next difference is the layer-0
FFN input. A generic normalized residual readout reduced early-layer errors,
but both complete eleven-prompt, sixteen-position controls still failed:
171/176 and 168/176 greedy matches, respectively, with over nine million raw
pointwise violations. Its 64 FP64-oracle, exact chunk and graph component cases
passed. The frozen controller summary incorrectly labels these as 128 cases;
the component inventory contains 64.

The public FlashInfer TinyGEMM adapter passed the original 29 projection checks
and 28 graph timings. Some decode geometries improved about 1.2–3x; a long-prefill
geometry fell to 0.50x. These are component timings, with no whole-model speedup
or accuracy acceptance. A separate generic API integration passed 477 CPU/Gloo
checks, with 335 skips and two existing expected disagreements; it remains an
isolated candidate and has not been GPU-accepted or adopted.

A generic pointer-table projection group was also tested in job 118. It reuses
existing weight allocations and linear arithmetic, adding 24 persistent bytes
for three projections. Its complete eleven-prompt, sixteen-step raw logits
matched the prior candidate bitwise, and full/chunked logits remained exact.
Paired whole-model decode improved only 2.6%, below the adoption gate. The
prototype remains outside the production runtime. A separate batched library
GEMM trial failed twelve original accuracy/chunk-invariance checks and was
rejected without relaxing tolerances.

Job 86 additionally ran the pinned official native `one_batch.py` benchmark
with the same saved 27-token input, 64 raw output tokens and three repeats.
Median batch 1/2/4/8 decode throughput was 101.84/199.63/395.19/718.34 tokens/s.
Its timer covers metadata, forward, GPU greedy sampling and synchronization;
mini also prepares CPU requests and transfers sampled tokens to the CPU.
The official run retains the declared single-op autotuning skip. It is neither
default-tuned SGLang nor proof of identical measurement boundaries. Strict BF16
alignment and performance parity remain open.

Use `checkpoint_suite.py --strict-chunked-prefill` to assert full-vocabulary
consistency and exact first greedy-token equality for every prompt before
performance measurements. The released
checkpoint gate also accepts `--chunk-size` and prepares attention metadata for
each chunk, including singleton tails. Both retain the original tolerances.

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
