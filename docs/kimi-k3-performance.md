# Kimi-K3 serving performance target

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

Run both servers on the same hardware through gpuq. Keep development and CPU
validation remote. Record the model/tokenizer revision, both source commits and
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
