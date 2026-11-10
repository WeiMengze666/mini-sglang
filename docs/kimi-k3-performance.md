# Kimi-K3 serving performance target

## Replacement-machine baseline (2026-11-11 CST)

The [fresh canonical pair](validation/kimi-k3-replacement-baseline-20261110.json)
completes the original 393-entry, concurrency-48, 900-second AIPerf scenario
on the replacement eight-B300 server, through GPUq job23. Both fresh servers
finish with valid, complete AIPerf exports and zero request errors.
Mini source is `de99c88`; official source remains `baa9bef`.

| AIPerf metric | Mini | Official SGLang |
| --- | ---: | ---: |
| Output tokens/s | 220.63 | 364.20 |
| TTFT P50 / P95, ms | 2300.28 / 9439.95 | 1163.41 / 5114.00 |
| ITL P50 / P95, ms | 63.34 / 110.22 | 29.73 / 52.63 |
| Request latency P95, ms | 161732.53 | 83396.91 |
| Completed / request errors / deadline cancellations | 328 / 0 / 6 | 493 / 0 / 3 |
| Cache-read fraction of model prompt tokens | 88.73% | 93.72% |
| Mean effective concurrency | 15.65 | 12.43 |

Mini reaches **60.6%** of this official throughput point. The target remains
unmet. Hardware/software changed: do not interpret these values against old
395.42/276.90 results as a controlled speed regression. Both new runs preserve
the original command, precision settings and resolved workload; warmup and
model loading are outside the exported measurement. AIPerf receives online
access permission for local-tokenizer resolution in both runs; cached public
dataset bytes are unchanged. The failed first attempt is retained separately.

The scenario progresses through different requests as completion speeds differ.
The ratio is a serving test point, not a uniform kernel slowdown or an SLO curve.
Cache policies still differ; matching the 0.31 budget ratio does not establish
matching capacities or checkpoint reuse. Native prompt accounting omits three
assistant-stub tokens; the cache fraction above includes the documented offset.
Deadline cancellations remain separate from request errors. All rates and
percentiles retain AIPerf measurement boundaries rather than dividing by900.

The separate complete GSM8K baseline is Mini1281/1319 versus official1284/1319;
Mini has two invalid/truncated responses, official has none. This is not final
quality acceptance. The opt-in native FlashInfer KDA prefill candidate has
576 exact component controls across eight ranks, but its complete task run
is still in progress; neither its quality nor its serving performance is
claimed here. No candidate backend or checkpoint policy is enabled by this
documentation update.

## Historical checked scope (2026-11-05 CST)

The latest qualified runtime `51420e6` completes the requested canonical profile
at **276.90 output tokens/s**, 70.0% of the retained official 395.42 tokens/s point. The
[residual serving record](validation/kimi-k3-residual-serving.json) preserves its
full metrics and separate 1280/1319 task result. The serving target remains unmet.

Measured baseline runtime `9ec11fe` includes generic native-MNNVL payload canonicalization, qualified
by 947 CPU/Gloo passes and 256 exact eight-GPU mixed-view results. The retained
[whole-model native quality record](validation/kimi-k3-native-quality.json)
separately scopes the full GSM8K result to model source `385d797`: 1281/1319,
one invalid/truncated answer, versus native 1284/1319. The ABI update has component
and new-source whole-model B48 controls, not a separately rerun full-task score.

Default native AR algorithms produce eager/graph differences at batch 37–48.
Both directions of algorithm matching restore exact logits and convolution/KDA
state on eight ranks. Independent native SDK replay reproduces retained multicast
outputs exactly; real-input numerical checks use separately stated path envelopes.
The initial uniform FP32/single-output-rounding check remains a failed hypothesis.
These controls support numerical attribution to the native reduction path without
asserting an exact hardware reduction tree or changing serving precision.

The [first complete canonical Mini baseline](validation/kimi-k3-canonical-mini-baseline.json)
uses all 393 dataset entries, concurrency 48 and the requested 900-second profile,
with native SDK default heuristics. It completed 260 requests with zero request
errors and five deadline cancellations: 146.59 output tokens/s, TTFT P50 1612.12 ms
and ITL P50 74.82 ms. Statistics retain AIPerf's own measurement boundary.
The [completed sequential canonical pair](validation/kimi-k3-canonical-pair.json)
retains both servers' exact arguments, package versions, precision contract,
source revisions, request errors and deadline cancellations. It shows a substantial
remaining performance gap; near-SGLang performance acceptance remains false.
This single closed-loop scenario is not a per-kernel slowdown or an SLO curve.

| Metric | Mini | Official SGLang |
| --- | ---: | ---: |
| Output tokens/s | 146.59 | 395.42 |
| TTFT P50 (ms) | 1612.12 | 554.35 |
| TTFT P95 (ms) | 81327.42 | 5143.76 |
| ITL P50 (ms) | 74.82 | 25.67 |
| ITL P95 (ms) | 290.53 | 47.31 |
| Completed requests | 260.00 | 526.00 |

Historical stages retain their original source and evidence boundaries.


The primary performance target is the user-specified SGLang serving profile in
[`kimi-k3-sglang.json`](../benchmark/online/profiles/kimi-k3-sglang.json).
The profile preserves the exact requested argument vector. It is a target,
not a completed performance or precision-parity result.

```bash
sglang serve \
  --trust-remote-code \
  --model-path /data/models/Kimi-K3 \
  --served-model-name moonshotai/Kimi-K3 \
  --tp-size 8 \
  --dcp-size 8 \
  --page-size 64 \
  --context-length 1048576 \
  --mem-fraction-static 0.90 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-full-memory-ratio 0.31 \
  --max-running-requests 48 \
  --max-prefill-tokens 32768 \
  --chunked-prefill-size 32768 \
  --cuda-graph-max-bs-decode 48 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-metrics \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 38080
```

## Qualified indexed KDA

The [indexed KDA record](validation/native-indexed-kda.json) qualifies direct
working-pool updates through the existing KDA kernel registry. The adapter uses
the official convolution and recurrence kernels, preserves public K,V state and
immutable checkpoint ownership, and adds no model-specific scheduling branch.
288 eight-GPU component checks pass, including exact BF16 output/state comparison
with the functional path and normal/unified-pool graph replay.

Matched isolated real-model decode decreases from 25.52 to 20.19 ms/step at batch8,
and from 39.95 to 29.48 ms/step at batch48 (1K prompts). These are reduced-cache
component measurements. The [complete indexed-source quality record](validation/native-indexed-kda-quality.json)
reports 1283/1319 correct, zero invalid answers and zero truncations, versus native
1284/1319. All eight ranks execute batches1–48 with finite outputs and released
owners; only batches37–48 remain nonexact under default native AR choices, matching
the previously attributed boundary. The selected-runtime canonical result is
recorded below.
The [prefill frontier change](validation/prefill-page-frontiers.json) has host
scheduler/cache controls and the full-capacity GPU lifecycle qualification below.

## Indexed runtime canonical serving

The [completed selected-runtime profile](validation/kimi-k3-indexed-serving.json)
uses the exact 393-entry, concurrency-48, 900-second AIPerf workload and the
qualified indexed/aligned runtime `e87b651`. Dense dispatch retains its previous
qualified path; the automatic GEMM candidate is excluded.

| Metric | Initial Mini | Selected Mini | Official reference |
| --- | ---: | ---: | ---: |
| Output tokens/s | 146.59 | 265.61 | 395.42 |
| TTFT P50, ms | 1612.12 | 1408.82 | 554.35 |
| TTFT P95, ms | 81327.42 | 9042.55 | 5143.76 |
| ITL P50, ms | 74.82 | 45.34 | 25.67 |
| ITL P95, ms | 290.53 | 80.99 | 47.31 |
| Cache-read fraction | 70.09% | 90.39% | 93.35% |
| Completed requests | 260 | 366 | 526 |
| HTTP/request errors | 0 | 0 | 0 |
| Deadline cancellations | 5 | 4 | 2 |

Selected throughput improves by 81.2% over the initial Mini baseline and reaches
67.2% of the retained official point. The serving target remains unmet. These
are sequential profiles, not a randomized ABBA comparison or an SLO-goodput
curve. Effective concurrency and trajectory progress differ under this closed
workload; the ratio does not imply a uniform kernel slowdown. Input-derived
reference metrics include the documented three-token prompt-stub adjustment.

The same selected runtime scored 1283/1319 with no invalid answers or truncations.
The subsequent residual-bank runtime has separate qualification below; its
results must not be attributed to this completed serving profile.

## Aligned prefill and full-capacity cache checks

[Eight-GPU cache qualification](validation/native-indexed-kda-cache.json) uses
the selected indexed runtime with the canonical memory split, 312 cached states
and 38152 logical KV pages. Retained mixed long prompts restore 551936 tokens;
restored tensors match their checkpoint exactly, and the checkpoint remains
unchanged. Cache pressure with 408 prefixes, 48-request decode, four cancellation
stages and slot reuse pass on every rank. These are lifecycle checks, not an
independent model-quality oracle or serving throughput result.

The host control explains one baseline failure: a partial shared prefill budget
left every later chunk off the page boundary, so KV keys existed without a
restorable recurrent state. Aligning intermediate absolute frontiers fixes this
case without an eviction-policy change. The final unaligned tail still lacks an
interior checkpoint: this fixture matches KV through 584192 but restores state
at 551936, leaving extra replay. Native cache-policy parity remains unfinished.

The untraced canonical AIPerf result is recorded above. The retained initial
failed GPU fixture used the wrong final-request predicate; its corrected run
and host control are separately recorded.

## Owned residual bank

The [residual-bank record](validation/native-residual-bank.json) selects generic,
forward-local storage for frozen block sums. The model supplies its bank capacity;
the residual layer passes the valid row count to the existing native kernel.
This removes repeated history stacking without a new numerical kernel, precision
option or model-specific communication branch. CPU and larger-bank fallbacks remain.

1047 host checks pass. Expanded eight-GPU controls pass 544 checks, including
528 native-kernel cases and 16 larger-bank fallbacks. On actual model tensors,
17952 same-input residual reads match the previous stack path bitwise and preserve
all inputs. All eight ranks execute decode batches 1-48; matching the official AR
algorithm restores exact B48 eager/graph logits and convolution/KDA states.

The complete task result is 1280/1319, with two invalid answers and one truncation,
versus previous 1283/1319 and native 1284/1319. The previous and bank versions give
the same valid answer on 1311 cases. The local-operation and AR controls support the
numerical qualification; they do not identify the precise cause of every changed
greedy response. All responses and the lower score remain retained.

Within-process ABBA prefill of 32768 new tokens after 93184 cached tokens decreases
from 1942.98 to 1822.01 ms (6.2%); peak allocated memory does not decrease. Separate
reduced-cache decode measurements decrease 1.6-2.4%, with 178 fewer graph kernels.
These are diagnostic model measurements. A canonical serving measurement of this
runtime is now recorded separately below; the 265.61 tokens/s point above remains
source e87.

## Selected residual-bank serving

The [completed bank-runtime profile](validation/kimi-k3-residual-serving.json)
keeps 393 entries, concurrency 48, 900 seconds and all original precision/resource
arguments. Runtime `51420e6` includes the owned bank; the FP32-output projection
and packed-prefill candidates are excluded.

| Metric | Indexed runtime | Residual bank | Official reference |
| --- | ---: | ---: | ---: |
| Output tokens/s | 265.61 | 276.90 | 395.42 |
| TTFT P50, ms | 1408.82 | 1318.71 | 554.35 |
| TTFT P95, ms | 9042.55 | 9784.74 | 5143.76 |
| ITL P50, ms | 45.34 | 44.86 | 25.67 |
| ITL P95, ms | 80.99 | 80.85 | 47.31 |
| Cache-read fraction | 90.39% | 90.32% | 93.35% |
| Completed requests | 366 | 373 | 526 |
| HTTP/request errors | 0 | 0 | 0 |
| Deadline cancellations | 4 | 4 | 2 |

Throughput rises 4.2% versus the indexed runtime and 88.9% versus the initial Mini
baseline, reaching 70.0% of the retained official point. TTFT P95 increases;
not every metric improves. These sequential profiles preserve AIPerf's own
measurement boundary; they are not a randomized serving ABBA or an SLO-goodput
curve. Peak GPU memory was not sampled by this no-telemetry profile. Mini still
has 48 working/312 cached states and 38152 logical KV pages; native has 372 shared
slots and 39560 pages, under different ownership policies.

## BF16 GEMM candidate

The [generic GEMM candidate](validation/native-bf16-gemm.json) in commit
`47e4961` delegated supported shapes to the official CuTe DSL/TGV SDK. Its
component controls passed, but the full frozen task set scored 1274/1319,
with three invalid answers and two truncations. The previously qualified
indexed runtime scored 1283/1319 with neither invalid answers nor truncations.

The default retains the previously qualified dense path. The candidate's
small isolated timing changes do not justify its worse task result. Source,
tests and all responses remain available for diagnosis; this decision does
not establish an arithmetic bug in the SDK or general model-quality loss.

## Precision and parallel layout

A model-level BF16 label is insufficient to establish parity. Record the actual
resident tensor formats and execution path for both servers:

| Component | Required evidence |
| --- | --- |
| Weights and MoE | Packed checkpoint format; actual runner; GEMM1/GEMM2 activation formats, quantization groups/scales, nonlinear intermediate dtype, and routing-weight rounding. |
| MLA KV | FP8 E4M3 storage, actual quantization/scaling layout, page size 64, and the active attention backend. |
| KDA | BF16 persistent SSM state as requested; independently record accumulator and convolution-state dtypes and state write rounding. FP32 accumulation does not imply FP32 persistent storage. |
| Parallel groups | TP8/DCP8 and the actual EP, expert-TP, and dense/shared MLP groups. |
| Cache | extra_buffer checkpoint/rollback semantics, state/KV memory split 0.31, prefix reuse, and actual pool capacities under the 0.90 static memory fraction. |

The reference command leaves EP and MoE precision/backend selection implicit.
In the inspected SGLang commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`, EP defaults to 1, while the
packed K3 override selects `flashinfer_mxfp4` on SM100/SM103/SM107.
The runtime server-info export from job 863 confirms these selections.
The selected runner's default branch prepares MXFP8 activations; see
`sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:1088-1098`.
The runner's `default` precision must be resolved to actual activation and
scale tensors. EP8 and BF16/Marlin results must carry their own configuration
labels.

The calibrated BF16/Marlin TP8/DCP2/EP8 reference remains a numerical diagnostic
gate. The serving target above requires a separate correctness gate using its
actual precision choices before performance acceptance. Changes to the
reference command or environment must be reported explicitly.

Mini exposes `--recurrent-state-dtype float32|bfloat16` independently of the
model/convolution dtype. The target uses BF16 persistent state. The native KDA
adapter passes that storage dtype to the mature kernels and preserves native
state write rounding. CPU reference adapters are separate diagnostic paths.
[Target components](validation/kimi-k3-target-components.json) pass 320 checks
across eight GPUs, including MXFP8 expert inputs/BF16 output, FP8 MLA storage,
DCP8, BF16/FP32 KDA state, ownership and graph replay. These checks qualify their
listed operator contracts, rather than a full serving workload.

## Measurement boundary

Run both servers on the same hardware through gpuq. The replacement-machine
work uses the restored queue; preserve process ownership and serialization. Keep development and CPU validation remote. Record the model/tokenizer revision, both source commits and
dirty diff hashes, CUDA/NCCL/Torch/FlashInfer versions, resolved startup
configuration, and fallback warnings.

Match request/token limits, 32K prefill chunks, decode graph batch coverage
through 48, and the memory budget. Report the same prompt/output lengths,
concurrency or arrival policy, seeds, and warmup. Measure prefix-cache misses
and hits separately, including KDA checkpoint reuse. Use the same reasoning
and tool parser behavior at the serving boundary.

Report output-token throughput, TTFT, ITL/TPOT, request latency, peak memory,
and cache metrics. An isolated kernel, an eager model loop, or a different
precision/layout is useful diagnostic evidence but does not establish the
requested serving performance ratio.

## Frozen baseline and canonical workload

The BF16 quality baseline is commit `eb3bd7b`: paired full GSM8K scores are
1281/1319 for Mini and 1282/1319 for SGLang, with no invalid/truncated answers.
Its TP8/DCP2/EP8, BF16 KV and FP32 KDA profile is explicitly separate from the
requested serving profile. Native S0 and initial Mini R0 measurements are complete; final precision-path
and serving acceptance remain pending.

The profile JSON now preserves the exact requested AIPerf command:
`inferencex-agentx-mvp`, `semianalysis-cc-traces-weka-062126`, 393 dataset entries,
concurrency 48, 900 seconds, seed 42, streaming server token counts, first-turn
prefix cache busting and all requested timing/export/metrics options. Entries
are replay trajectories, not a promised count of completed inference requests.
Keep the original command and resolved scenario/dataset metadata in each run.

Before comparing performance, verify precision/state and the HTTP chat, streaming
usage and metrics interfaces. A missing capability or failed run is reported
explicitly. Do not substitute another workload, shorten the duration, truncate
inputs or change precision to obtain a performance number. Freeze a runnable
baseline first; then evaluate each generic backend/cache/scheduling change with
quality/state regression checks and the same AIPerf workload.

The native reference startup in job 858 exceeded the launcher readiness limit
while initializing/autotuning; AIPerf never started and this is not a performance
result. Job 863 retries the identical server and benchmark argument vectors,
allowing 7200 seconds for readiness and 10800 seconds for benchmark orchestration.
The actual AIPerf measurement remains 900 seconds. Both attempts retain their
commands/logs; the first attempt left no owned server processes. The selected
393-entry public dataset is cached at revision
`23f152f6f0f9399a85901b89a6458def0ef16729`.

Mini now reuses the checkpoint tokenizer and native reasoning/tool parsers through
generic adapters. CPU checks cover actual K3 thinking modes, tool conversations,
streaming/non-streaming composition and another supported model format. This
qualifies the protocol adapter on CPU; it does not establish GPU serving or
target precision acceptance.

AIPerf 0.13 uses `max_completion_tokens` by default. Mini accepts both completion
budget spellings and forwards the requested limit to scheduling. CPU validation
checked 386 actual captured AgentX payloads and their native-parser construction;
all preserved their requested output budgets. These are interface checks,
separate from model and serving-performance acceptance.

## Native S0 result

[Versioned result](validation/kimi-k3-aiperf-reference.json) freezes job 863,
SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`, AIPerf 0.13.0, the canonical
argument vectors, model/tokenizer metadata hashes and raw export location.
This is one reference run; the initial Mini R0 result below retains its explicit
precision and implementation gaps.

| AIPerf metric | Native S0 |
| --- | ---: |
| Output throughput | 378.53 tokens/s |
| Request throughput | 0.5447 requests/s |
| TTFT P50 / P95 | 590.35 / 4482.19 ms |
| ITL P50 / P95 | 26.76 / 51.08 ms |
| Request latency P50 / P95 | 9.51 / 79.14 s |
| Completed / HTTP errors / deadline cancellations | 512 / 0 / 3 |
| Prompt cache reads / total prompt tokens | 52,267,520 / 56,177,301 |
| Prompt cache-read fraction | 93.04% |
| Effective concurrency, mean | 11.86 requests |

The scenario completed 51 warmup requests without errors before its nominal
900-second measurement. It stopped three in-flight credits at the configured
grace deadline, giving a 940-second phase boundary. Values above are AIPerf's
unmodified exports; do not replace their denominator with 900 seconds. The
393 trajectories and concurrency cap 48 do not imply 393 completed requests or
48 continuously active requests. The original logs and cancellation counts are
retained. The owned server process audit found no remaining processes.

Runtime resolution confirms TP8/DCP8/EP1, `flashinfer_mxfp4` with precision
`default`, FP8 E4M3 KV, BF16 SSM, TRTLLM MLA prefill and CuTe MLA decode. Unified
memory is disabled in this requested command. The static state/KV budgeting
ratio 0.31 allocates `0.31 / (1 + 0.31)` of the remaining budget to state, with
padding/workspace terms included by the native allocator; it is not 31% of total
GPU memory. Mini's initial fixed checkpoint budget remains an explicit R0 gap.

The native small-batch route/quant fusion also packs routing weights to BF16;
its unfused runner accepts FP32 weights. Mini R0 currently uses that unfused
runner contract. This rounding boundary is recorded separately from MXFP8 expert
activations. A later generic prepared-input backend must adapt the native fused
route/quant operator before claiming full precision-path/performance parity.

Mini's cache-read usage is reported as `usage.prompt_tokens_details.cached_tokens`.
It counts the prefix skipped at admission and remains constant across this
request's own prefill chunks. Completed cache-read token totals are also exposed
in frontend metrics. This is token reuse accounting, not device-memory telemetry.

The native API deliberately omits K3's three-token assistant generation stub
from reported/billed prompt usage, while the model consumes those tokens.
Mini reports actual input token IDs through its generic usage interface. A CPU
replay of all 512 completed S0 AgentX requests confirms complete input-ID equality with
the native OpenAI encoder and reproduces its billing deduction. Preserve raw
AIPerf exports and normalize input-derived metrics to actual model tokens when
comparing; output throughput and latency need no such adjustment. No prompt
framing is removed to make usage counts match.

## Mini R0 correctness and measurement status

The first runnable target-format Mini R0 is frozen at
`07d94b5de00b883be81a45f6c6cc7c3d92d8de45`. Its bounded weight arenas load the
complete TP8/EP1 checkpoint without the prior tensor-by-tensor allocation
failure. [Target quality evidence](validation/kimi-k3-target-quality.json)
records the full 1319-case GSM8K run and real-checkpoint cache lifecycle gate.

| GSM8K, same cases and target formats | Mini R0 | Native |
| --- | ---: | ---: |
| Correct | 1281 / 1319 | 1284 / 1319 |
| Accuracy | 97.1190% | 97.3465% |
| Invalid answer format | 1 | 0 |
| Truncated | 0 | 0 |

There are 1313 identical valid extracted answers and 658 identical generated
ID sequences. Exact greedy identity is diagnostic. The Mini formatting failure
(case 754) calculates 89 correctly but omits the required `####` delimiter;
the scorer remains unchanged. The differing cases have not all received
first-divergence attribution, so this report does not claim that every
variation is a proven rounding effect or that comprehensive quality acceptance
is complete. Evaluation-loop wall time is not a serving-performance result.

All eight ranks pass the real-checkpoint scheduler/cache lifecycle gate:
checkpoint contents, independent restored storage, prefix branching, eviction,
serial/overlapped execution, decode graph replay, and cancellation at pending,
chunked, decoding and in-flight stages followed by slot reuse. Each rank records
49 checkpoints, four restores and 34 evictions. Tensor assertions synchronize;
this is content/ownership validation, not asynchronous timing-stress evidence.
The native-kernel audit observes the official KDA/convolution/MLA-cache kernels
and FlashInfer expert/MLA calls. Mini's only observed FFI module is the PyNCCL
communication shim. Python call counts include graph capture, not graph replays.

GPUq job 877 completed this frozen R0 with the exact requested AIPerf command.
Its profile explicitly retains initial implementation gaps: eight state-cache
slots instead of native ratio-based capacity, no built-in native MNNVL transport,
unfused FP32 routing weights, and serial default scheduling. Those are measured
baseline limitations, not final parity settings. Use these measurements as the initial baseline; a precision-matched serving
acceptance still requires the corrected path and remaining policy work.

The first R0 attempt (job 876) stopped before server readiness and before AIPerf:
a probe received its own `GET /v1/models HTTP/1.1` as an invalid status line.
A CPU-only Linux TCP self-connect reproduces that response on an unused
loopback port; the requested port 38080 lies in the host ephemeral range.
Original socket endpoints were not captured, so self-connect is a supported
explanation rather than a directly observed original connection. The launcher
now retries HTTP protocol exceptions alongside connection failures, records
probe errors, and still detects server exit/readiness timeout. Four real local
HTTP/process integration checks pass, including malformed responses and failed
benchmark cleanup. Job 877 changes only the launcher; model source, server argv,
precision and the 900-second AIPerf argv remain identical. Job 876 leaves no
owned processes and contributes no performance number.

## Prepared graph-coverage candidate

R0 startup captures batches `[1, 2, 4, 8, 16, 24, 32, 40, 48]`. Its hybrid
backend does not support padding, so other decode batch sizes execute eagerly.
The first candidate makes automatic graph selection use every size through the
requested maximum when a backend cannot pad. Backends that support padding keep
sparse buckets; an explicit `--cuda-graph-bs` list still takes precedence as a
memory/startup-time choice. This policy uses the existing backend capability
without naming a model or attention algorithm. It does not change numerical
kernels, precision, scheduling or cache policy.

The candidate passes 23 focused CPU checks, including graph-runner wiring,
request-capacity limits, explicit bucket selection and launcher cleanup. Its GPU
and serving effect remain unmeasured. R0 continues from its frozen source and
provides the comparison baseline before candidate performance acceptance.

## Additional prefill precision gap

A further audit of native `baa9bef` finds that `TRTLLMMLABackend` uses the KV
format for its ragged prefill Q/K/V as well: FP8 E4M3 inputs and a BF16 output
for this checkpoint's unit scales. Mini R0 stores FP8 KV and quantizes decode
queries, but passes BF16 expanded Q/K/V to the prefill kernel. The earlier
execution-contract comparison did not describe this boundary. Its GSM8K
results and component checks remain valid for their recorded paths; they do
not establish full precision-path equality. R0 measurements must retain this
explicit limitation and must not be published as a precision-matched ratio.

The correction passes cache precision into the generic FlashInfer MLA adapter
and uses normal Torch casts at the SDK boundary, as the native backend does.
Producer output stays BF16. There is no new numerical kernel or model-name
condition. Focused CPU coverage reports 89 passed and one skipped. Six opt-in
GPU cases are prepared to compare against the actual native backend prefill
method, recording Q/K/V/output dtype, shape and stride, with exact same-ABI
output and producer-stream checks. The native-API control now passes all 48 cases across eight ranks, and the
complete prefill component gate passes 304 checks with no skips. See
[the precision evidence](validation/kimi-k3-prefill-precision.json). Renewed
whole-model quality stopped after680 completed cases in job878 with an unknown
CUDA error. The subsequent all-device fresh-context smoke fails initialization:
NVML cannot access GPU7 and marks the other seven devices with recovery action
`Reboot`. The cause of the original failure is unconfirmed; full model and real
1–48 replay acceptance remain incomplete. See [the interruption evidence](validation/kimi-k3-gpu-interruption.json).
The graph-only R1 gate remains prepared but unsubmitted.

## Initial Mini R0 serving result

[Versioned R0 evidence](validation/kimi-k3-aiperf-mini-r0.json) records job 877,
source `07d94b5`, the same canonical AIPerf argv/resolved workload as S0, package
versions, raw export hashes and the completed run. Both runs complete all 51
warmup requests, then the nominal 900-second profiling phase with its configured
grace boundary. No input truncation, shorter duration or changed concurrency is
used. This is the runnable initial baseline, with the precision and policy gaps
above; it is not a final precision-matched comparison.

| Unmodified AIPerf metric | Mini R0 | Native S0 |
| --- | ---: | ---: |
| Output throughput | 22.84 tokens/s | 378.53 tokens/s |
| Request throughput | 0.0936 requests/s | 0.5447 requests/s |
| TTFT P50 / P95 | 10.02 / 233.02 s | 0.59 / 4.48 s |
| ITL P50 / P95 | 428.34 / 957.69 ms | 26.76 / 51.08 ms |
| Request latency P50 / P95 | 119.84 / 458.24 s | 9.51 / 79.14 s |
| Completed / HTTP errors / deadline cancellations | 88 / 0 / 20 | 512 / 0 / 3 |
| Prompt cache reads / total reported prompt tokens | 942,592 / 8,543,041 | 52,267,520 / 56,177,301 |
| Reported prompt cache-read fraction | 11.03% | 93.04% |
| Effective concurrency, mean | 14.84 requests | 11.86 requests |

Both exports are complete and mark the submission valid. Deadline cancellations
are recorded separately from HTTP errors, and the incomplete requests are
subject to AIPerf's normal accounting. Preserve its exported throughput and phase
denominators. Native prompt billing omits three input framing tokens per
request; the evidence separately normalizes input-derived totals to model tokens.
This tiny accounting difference does not explain the cache-reuse gap. Neither
run collects peak request GPU memory under the requested telemetry settings;
post-initialization free-memory logs are not peak-memory measurements.

The baseline falls far short of the target. Low state-backed prefix reuse and
missing graph coverage are measured/confirmed gaps; their individual performance
contributions are not yet isolated. First qualify the prefill precision correction
and graph policy, measure that candidate, then evaluate generic ratio-based
state/KV capacity and checkpoint ownership/policy improvements. The post-run
audit finds no owned R0 server processes remaining.

## Prepared generic cache-budget candidate

R0 has eight checkpoint slots and an implicit four-checkpoint limit per radix
path. The R3 candidate adds optional `--state-memory-ratio 0.31` in place of
`--max-cached-states 8`; both CLI options are mutually exclusive. State receives
`ratio / (1 + ratio)` of the post-weight budget, including working and overlap
scratch slots. Checkpoint count is derived from the selected state/convolution
dtypes, then agreed across ranks. Each rank subtracts its own state bytes before
the remaining KV budget is reduced, preserving asymmetric-rank memory safety.
Shared-arena sizing uses its actual aligned capacity and explicit page limit.
No model name, attention algorithm switch or numerical kernel is introduced.

The default checkpoint policy retains prefixes up to the global state budget
with the existing branch-aware LRU, allowing future forks before the latest four
snapshots. Explicit per-path limits remain supported. Snapshot contents, copy
ownership, precision and state updates are unchanged.

[CPU evidence](validation/kimi-k3-cache-budget.json) records 867 passed and 666
skipped with manual CUDA scripts excluded. Tests include earlier-prefix restore,
static/shared storage and unequal rank state footprints. R3 has no GPU or
performance acceptance yet. Device recovery and the pending R2 correctness and
serving measurements come first; native lazy extra-buffer ownership and mature
communication integration still remain to be completed.

## Failure context for resumed qualification

The GSM8K validation runner now writes `failure-rankN.json` before cleanup when
evaluation raises an exception. It records the current case IDs and immutable cases hash, completed
generated-token histories, logical request/state positions, batch phase, graph
selection, and tensor dtype/shape/stride. Small CPU indices retain their values;
device tensors retain metadata. Recording does not read a device payload or
synchronize an already-failed CUDA context. The primary error remains the raised
failure even if shutdown also reports an error. A timestamp allows correlation
with driver logs. This changes validation evidence, not inference or scoring.

The 32 focused CPU checks include a failure on the second token transfer, complete
history preservation, distinct request-table and state-index metadata, noncontiguous tensor views,
cleanup failing after the original error, and refusal to overwrite failed-run
evidence. The fresh health recheck in GPUq
job880 still reports a missing device and `Reboot` recovery actions; its NVML gate
stops before CUDA initialization. No corrected whole-model or serving acceptance
is established by these diagnostics.

## Routing ABI precision and CPU-only continuation

The FlashInfer expert adapter now preserves BF16 or FP32 route weights rather
than forcing every input to FP32. This is an SDK boundary rule shared by models;
K3's current router still emits FP32, so this change alone does not switch its
routing precision or establish fused-path parity. Unsupported weight dtypes fail
before the SDK call.

The [routing audit](validation/kimi-k3-routing-precision.json) executes the pinned
native runner's routing helper and FlashInfer's input splitter on CPU. STANDARD
routes preserve FP32 or BF16 weights; packed routes encode BF16 score bits in the
low half of each int32 entry. The SDK C++ launcher chooses its expert-weight dtype
from the unpacked input and borrows its pointer directly. Thus API-format and
precision choices must be recorded together. Native small-batch route/quant
fusion uses packed scores, but its STANDARD fallback can retain FP32. S0 did not
capture per-call fusion coverage; a statement that all native K3 route weights
are BF16 would be unsupported. Host bit-layout checks do not validate GPU
finalization or explain any task-answer differences.

Seven CPU-backed adapter ABI fixtures passed. The updated CPU suite passed 886
checks and skipped 678; manual `tests/kernel` scripts were excluded and queued
CUDA tests were disabled. The existing native expert component test now covers
BF16 and FP32 route weights with both activation dtypes and row layouts; these
additional GPU cases remain unrun. The standalone audit can be repeated with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python:/path/to/sglang/python \
  python tests/validation/moe_routing_contract.py --output /fresh/path/routing.json
```

After permission to use the remaining seven devices, GPUq job 886 selected each
by its NVML UUID, excluding missing GPU7. All seven fresh processes failed CUDA
initialization and NVML still reported `Reboot`. This rules out a usable seven-card
execution path at that observation; the raw evidence is in the
[device interruption report](validation/kimi-k3-gpu-interruption.json). CPU work
continues, while TP8/DCP8 quality and AIPerf acceptance require device recovery.

At the device interruption, native MNNVL bootstrap was still incomplete. The
pinned custom-AR v2 cold P2P check consults SGLang's global world despite
accepting a process group. The [native provider](distributed-backends.md) now
runs the official peer probe before group creation, then constructs the native
communicator with Mini-owned groups. Eight-device eager/graph communication
qualification passed; whole-model quality and serving performance remain pending.

## Device recovery qualification

The [recovery record](validation/kimi-k3-device-recovery.json) confirms all eight
B300 devices are visible again with NVML recovery action `None`. Each rank passed
BF16 and FP32 NCCL graph replay and device-to-host checks, 30 repeats per dtype,
and exited cleanly. The first smoke run left a captured graph alive while
attempting process-group destruction and did not exit after its arithmetic
checks. Explicitly resetting graphs before group destruction resolved the smoke
script's teardown. No native internal stack was captured, and this does not
attribute the earlier device fault to the inference implementation.

The unchanged R2 prefill component suite passed again on all ranks: 304 tests,
zero failures, errors or skips. The subsequent whole-model and graph/state
checks completed as described below. These recovery and component results do not replace
whole-model quality or serving acceptance. The user temporarily authorized direct
GPU execution while GPUq is unavailable; each stage has an owned-process ledger,
launch manifest and retained exit status. No direct run is recorded as a GPUq job.

## Comparing retained AIPerf exports

`benchmark/online/summarize_aiperf.py` reads completed, submission-valid exports
and rejects mismatched benchmark arguments or resolved workload settings. Only
the resolved artifact directory is excluded from workload comparison. Exported
metric objects and their units are preserved: the helper does not recalculate
throughput using the nominal 900-second duration. Profiling deadline cancellations
and HTTP errors are separate, and unavailable phase counts or package metadata
remain unknown. Workload equality alone does not establish precision, quality,
cache-policy or hardware parity.

For K3, pass `--reference-prompt-token-offset 3` only when comparing the verified
native billing convention described above. The generic default offset is zero;
normalization changes derived input accounting, never raw token counts or output
throughput. For example, after the candidate is qualified:

```bash
python benchmark/online/summarize_aiperf.py \
  --reference /path/to/reference-run --candidate /path/to/candidate-run \
  --reference-prompt-token-offset 3 --output /fresh/path/comparison.json
```

Seven CPU fixtures cover incomplete/invalid runs, unchanged tool boundaries,
cancellation counts, token accounting, workload differences, missing metadata and
duplicate exports. The helper also reproduced the original S0/R0 throughput and
completion/cancellation counts from their unmodified exports. This reporting
validation is separate from the pending new serving measurements.

## Scope of task contract comparison

The task comparator now reports `declared_execution_contract_matches` and leaves
`kernel_precision_path_verified` unknown. Matching declared model/MoE/KV/state
dtypes, parallel layout and page size does not prove that every kernel input,
accumulator, rounding boundary or fused path matches. Earlier immutable reports
used `precision_contract_verified` for this narrower declared-field check; retain
them as recorded, with that scope. The scorer, task protocol and accuracy values
are unchanged. Forty related CPU checks passed after this reporting clarification.


## R2 recovery qualification and fresh official baseline

The complete [R2 record](validation/kimi-k3-r2-recovery.json) uses frozen source
`743a4f53e79d9c6cf844bed7b02aecf68e965e6c`. It scores 1279/1319 versus the retained
canonical-precision native task result of 1284/1319, with one invalid fractional
answer and zero truncations. Thirteen valid-answer comparisons differ; the
comparison reports declared dtype/layout equality separately from unverified
kernel precision-path equality. No grader or failed record was changed.

With 48 distinct 30-token contexts, all eight ranks pass every batch size 1–48:
three teacher-forced eager steps and prefill plus two graph replays produce
bitwise-equal logits and persistent convolution/KDA states. State slots are
released after each batch. This establishes Mini graph/eager consistency for
that workload, not independent native model-quality agreement. The bounded and
full task runs also differ on case 423; their preceding batches/tuning context
differ, so repeatability and rounding attribution remain open.

The restored-device [official refresh](validation/kimi-k3-reference-refresh.json)
uses the exact user-specified serve/AIPerf arguments and SDK versions. It
completes the canonical 900-second workload with 524 completed requests, zero
HTTP errors and three deadline cancellations. AIPerf reports 389.47 output
tokens/s, TTFT P50/P95 589.22/4747.35 ms, ITL P50/P95 25.59/50.36 ms and 93.37%
prompt-cache reads under its original accounting. Historical S0 remains intact.

The first restored-device R2 Mini serving attempt failed during its first
warmup with a CUDA OOM in cached-prefix DCP all-gather. The GPU workers exited
while the HTTP frontend waited; the owned tree was stopped and raw evidence
retained. No profiling result was produced. Frozen R2 fixes R0's MLA
prefill precision and graph-size coverage while retaining eight cached states,
unfused FP32 routing, PyNCCL and serial scheduling. It is an intermediate
performance measurement. Final quality, fused-path precision, cache-policy and
serving acceptance remain open. The native communication provider is opt-in and
is not included in this frozen R2 measurement.


## Native MoE preparation candidate

The [component-qualified candidate](validation/native-moe-preparation.json)
keeps raw router scores and model-level grouping/scoring choices in
`moe/routing.py`. `BaseMoeBackend.forward_experts` lets a local expert executor
attempt combined routing and activation preparation before materializing
ordinary routes. The FlashInfer executor adapts the official fused router/MXFP8
kernel and forwards its packed BF16 routing weights, FP8 activations and UE8M0
scales directly to the mature expert kernel. Coverage comes from the SDK and
model-independent semantic checks. Prepared tensors have explicit ownership;
Mini introduces no global handoff.

Uncovered inputs preserve standalone routing, quantization and supplied routing
precision. Eight GPU ranks pass 352 component checks, including native runner
comparisons, fused/unfused preparation, packed-weight rounding, strided inputs
and changing scores/activations across graph replays. Full CPU regression passes
918 tests with 695 skips. This candidate has no full-model quality or serving
acceptance yet and was not used in the frozen R2 serving attempt.

## Long-prefill allocation failure and initialization fix

The restored R2 serving failure was reproduced with 65,537 synthetic input IDs,
32K chunks, all graph sizes 1–48 and the original dtype/memory settings. After
the first chunk, rank 0 had about 5.5 MiB driver-free memory while Torch retained
about 20 GiB of unused reserved storage. The next chunk's first cached-prefix
DCP all-gather failed allocating NCCL resources. The
[controlled record](validation/native-collective-bootstrap.json) retains all
rank outcomes and allocator/shape/stride metadata.

Initializing DCP collectives before weights and cache-budget measurement lets
all eight ranks complete the same three chunks and clean up. No hot-path cache
purge, precision change, chunk-size reduction or capacity override was used.
The production fix prepares Torch/NCCL resources during communication startup;
the builtin native provider also prepares its NCCL fallback. Updated full-model,
cache-lifecycle and serving validation remain pending. Worker exits now wake
pending frontend requests and make readiness/new requests return 503, avoiding
the HTTP wait observed in the failed serving attempt.

## Production long-prefix and automatic state-budget check

The [production qualification](validation/kimi-k3-long-prefix-budget.json) at
`1100408fa88b41de0cc593c309a15a319b24c85c` passes on all eight GPUs. Under the
0.90 memory fraction and 0.31 state/KV ratio, each rank resolves 333 cached-state
slots and 40,368 logical pages. The synthetic 163,841-token request uses 32K
chunks and retains five checkpoints along one radix path. A repeat restores the
163,840-token checkpoint; a fork restores the 131,072-token checkpoint into
distinct private working storage. Copies are exact. Cold/repeat final logits
and four greedy tokens match exactly; all outputs are finite and resources are
released. This uses the production collective warmup and mature fused MoE
preparation, with default PyNCCL and scoped DCP communication.

These are functional cache/state checks. Global-capacity pressure, eviction and
abort at the automatic capacity, whole-model native-MNNVL, updated task quality
and the canonical serving measurement remain open. Archived qualification trees
are pinned by their launch manifests and source hashes; an enclosing checkout's
HEAD does not identify an archived package.

## Updated fused-routing whole-model result

The [R4 result](validation/kimi-k3-r4-quality.json) completes all 1,319 task
examples at frozen model source `1100408fa88b41de0cc593c309a15a319b24c85c`:
1281 correct, zero invalid and zero truncated. The retained native target result
has 1284 correct. There are ten differing valid final answers; token identity is
a diagnostic. All eight ranks again pass the 48 distinct-context graph/eager
logits and state comparisons exactly. Runtime audit observes native fused
route/quant calls, BF16 state storage and no Mini numerical kernel calls.
PyNCCL communication and the portable DCP combine still distinguish this R4
configuration from the native communication candidate. Final native alignment
and serving performance remain unqualified.

The next [DCP combine adapter](validation/native-dcp-combine.json) reuses the
mature native LSE-combine kernel with natural-log statistics. It zeros undefined
empty-shard values and preserves corrupt active-row diagnostics. Selection uses
shape/dtype coverage and SDK availability, independently of model identity.
CPU/reference paths remain available. Eight GPUs pass 120 controls against
unpartitioned attention, empty/corrupt inputs and changing strided graph inputs;
the complete CPU/Gloo suite passes 938 tests with 706 skips. Whole-model native
communication, cache-capacity pressure and serving checks are still pending.

### Native communication and full-cache lifecycle

The `385d797` candidate passed whole-model TP8/DCP8 pressure with the opt-in
native MNNVL provider, native DCP LSE combine, FP8 KV, BF16 declared KDA state
and MXFP8 MoE activation. The 0.31 ratio budget yielded 312 cached state slots
and 38,152 logical pages per rank. All eight ranks filled the cache with 408
distinct prefixes, evicted cold checkpoints, restored hot and rebuilt states
exactly to separate working storage, decoded a live batch of 48, and drained
queued/prefilling/decoding/inflight abort-and-reuse cases. The native SGLang
framework World stayed uninitialized; no Mini numerical kernel was observed.

[The state-pressure record](validation/kimi-k3-native-state-pressure.json)
retains the earlier fixture failure and its independent corrected replay.
This functional result does not establish task quality, serving performance,
or native shared-pool memory-policy equivalence. Native-communication full
GSM8K evaluation and the exact canonical AIPerf comparison are the next gates.


## Replacement-server recovery (2026-11-10)

The [replacement-server record](validation/kimi-k3-machine-recovery-20261110.json)
separates recovery from final acceptance: 1077 CPU/Gloo tests passed (798 skipped),
1368 native component tests passed across eight devices, and native multicast
communication passed eager/graph lifecycle checks. The machine and driver changed;
retained serving ratios do not establish performance on this host. Fresh full-model
quality runs completed at **1281/1319 Mini versus 1284/1319 official** (delta
-0.227445 percentage points), with two Mini invalid/truncated answers versus zero
official; 1311 valid answers and 667 token sequences match. Declared BF16 model and
conv/SSM, MXFP8 MoE, FP8 KV and TP8/DCP8/EP1 contracts match, but kernel precision
paths remain unproven. Offline evaluation durations of 917.094 s and 618.574 s are
not serving metrics. The record's `latest` field preserves hashes and adverse
cases: isolated gate13 passed 70 component tests and 72 exact native communication
cases per rank, including graphs/cleanup; BF16-representable inputs do not prove
hidden FP32 precision. Gate15's FP32-sensitive communication passed, but its
prefill stage failed when the fixed SDK internally reshaped `dt_bias` before an
all-T1 vector ABI; the new fallback candidate remains unqualified. Job17's official
server reached ready with RustUnifiedTreeCore, but AIPerf failed during tokenizer
configuration: its offline branch treated the local path as a Hub repo ID. The
900-second load never ran, Mini never started, and an environment retry remains
unverified. Candidates remain isolated and final acceptance remains false.
