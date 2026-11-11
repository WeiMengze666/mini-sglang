# Fresh canonical Kimi-K3 serving baseline — 2026-11-11

The fresh paired run completed on 8 B300 GPUs. Mini remains substantially slower than the pinned full SGLang reference. This record accepts the measured baseline; it does not claim the performance goal or general model-quality equivalence.

| Formal AIPerf metric | Full SGLang | Mini safe baseline |
|---|---:|---:|
| Output tokens/s | 364.196617 | 216.377196 |
| Completed requests / deadline cancellations / errors | 493 / 3 / 0 | 323 / 7 / 0 |
| TTFT mean / p95 / p99 (ms) | 1695.662 / 4929.032 / 11541.050 | 3429.877 / 9813.852 / 18150.185 |
| ITL mean / p95 / p99 (ms) | 32.961 / 54.203 / 85.201 | 68.167 / 102.472 / 124.235 |
| Completed output tokens | 342346 | 203395 |
| Throughput observation window (s) | 940.003239497 | 940.002012486 |
| Completed-request prompt cache read | 93.767237% | 88.686460% |

Mini achieves 59.4122% of reference throughput; its mean TTFT and ITL are 2.023x and 2.068x reference. This is an entire-service comparison, not a kernel slowdown ratio. The configured concurrency 48 is a limit; effective mean HTTP concurrency was 12.3647 and 15.6998. The 393 dataset entries are trajectories, not the number of completed HTTP requests.

Both services were fresh and ran the requested unchanged 393-entry scenario, 900-second sending period, concurrency 48, and seed 42 AIPerf scenario. Throughput uses the installed calculator's exact phase start-to-requests-end window: sending plus a 30-second grace period and a 10-second cancellation-credit wait. The separately exported `benchmark_duration` values 923.279425 / 926.087974 seconds is a different first-request-to-last-response span. All 51 warmup requests completed separately on each service. Deadline cancellations and Mini's recovered allocator warnings during warmup remain in the raw evidence; neither is relabeled as a zero-error clean run. The workload seed does not fix the server RNG: actual Chat defaults were stochastic temperature1/top_p1/top_k-1 on both services.

The common requested boundary is TP=8, DCP=8, EP=1, physical page size 64, context 1,048,576, memory fraction 0.90, state ratio 0.31, maximum requests 48, prefill/chunk size 32,768, graph maximum batch 48, FP8 E4M3 KV, BF16 recurrent and convolution state, and MXFP8 routed-expert activations. Native resolved FlashInfer KDA multi-token prefill and Triton KDA decode, with TRTLLM MLA prefill/CuteDSL MLA decode. The initial `linear_attn_backend=triton` leaf is not the final prefill selection; the actual startup dispatcher is recorded in the [dispatch correction](canonical-kda-prefill-dispatch-correction-20261111.md). Mini used its safe official-KDA adapter and mature MoE/MLA kernels. Identical configuration ratios do not make the two cache allocators equal: actual native logs show 372 Mamba slots and 32.59 GB KV/rank; Mini shows 48 working + 312 cached + 0 scratch states, a 9.74 GiB state budget, and 31.42 GiB KV/rank. Cache statistics therefore remain reported.

The baseline runtime is exactly the runtime bytes at production commit `290ad026` (safe decode implementation `a3f3341f`); this documentation commit changes no runtime. The pinned reference is SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`, FlashInfer 0.7.1, Torch 2.14.1+cu130, and AIPerf 0.13.0. Model config SHA256 is `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`. Checkpoint metadata and shard sizes are pinned; no full weight-content hash is claimed.

Separate greedy offline GSM8K cohorts scored 1284/1319 on this Mini and the reference; Mini had 2 invalid and 1 truncated response, versus 0 / 0. These are distinct quality observations, not evidence of token identity or quality on this stochastic serving scenario. Strict token/logit equality is diagnostic; correctness requires matching algorithm,precision and state-update semantics.

Optimization work remains isolated: generic DistributedImpl GEMM-all-gather/add adaptation, MoE workload-bound preparation, FlashInfer prefill integration, explicit merged KDA projections, and generic native sigmoid gating. Each candidate must retain its own component/model/serving evidence. No candidate is promoted by this baseline measurement.

Remote evidence is under `experiment/recovery-20261110/serving-safe-baseline-paired-prepared`, GPUq 42. The accompanying JSON pins the full paired audit and root measurement receipt. Original exports, raw records, phase logs, server configuration, cache metrics and source/calculator hashes are retained remotely.
