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
These source-derived expectations still require runtime confirmation.
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

Explicit FP8 KV-format support and full-profile precision validation remain
unfinished. The state-storage option alone does not establish equivalence.

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
