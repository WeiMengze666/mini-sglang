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

Mini exposes `--recurrent-state-dtype float32|bfloat16` for persistent state,
independent of the model/convolution dtype. The default remains FP32. Built-in
adapters use FP32 working state and the pool rounds at commit; checkpoints
retain the selected storage dtype. CPU tests cover allocation, memory budgets,
mixed-dtype alignment, batch remapping, and checkpoint restore in both pool
layouts. GPU integration tests also cover BF16/FP32 state commits, actual KDA
layer graph replay, and request-index remapping after arena compaction
(48 cases across eight GPUs, compared with the same Mini eager kernels).
This is an operator/state integration check; full-profile official numerical
and serving acceptance remain pending.

FP8 KV storage/attention adapters have component coverage. Full-profile precision
and serving validation remain unfinished; component checks and the state-storage
option alone do not establish full-profile equivalence.

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
requested serving profile. The native S0 measurement is complete; Mini R0
measurement and final target-profile acceptance remain pending.

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
This is one reference run; Mini has no corresponding serving result yet.

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
