# Fresh canonical Kimi-K3 serving baseline — 2026-11-11

The fresh paired run completed on8 B300 GPUs. Mini remains substantially slower than the pinned full SGLang reference. This record accepts the measured baseline; it does not claim the performance goal or general model-quality equivalence.

| Formal AIPerf metric | Full SGLang | Mini safe baseline |
|---|---:|---:|
| Output tokens/s | 364.196617 | 216.377196 |
| Completed requests / deadline cancellations / errors | 493 / 3 / 0 | 323 / 7 / 0 |
| TTFT mean / p95 / p99 (ms) | 1695.662 / 4929.032 / 11541.050 | 3429.877 / 9813.852 / 18150.185 |
| ITL mean / p95 / p99 (ms) | 32.961 / 54.203 / 85.201 | 68.167 / 102.472 / 124.235 |
| Completed output tokens | 342346 | 203395 |
| Throughput observation window (s) | 940.003239497 | 940.002012486 |
| Completed-request prompt cache read | 93.767237% | 88.686460% |

Mini achieves59.4122% of reference throughput; its mean TTFT and ITL are2.023x and2.068x reference. This is an entire-service comparison, not a kernel slowdown ratio. The configured concurrency48 is a limit; effective mean HTTP concurrency was12.3647 and15.6998.393 dataset entries are trajectories, not the number of completed HTTP requests.

Both services were fresh and ran the requested unchanged393-entry,900-second sending,concurrency48,seed42 AIPerf scenario. Throughput uses the installed calculator's exact phase start-to-requests-end window: sending plus30-second grace and10-second cancellation-credit wait. The separately exported benchmark_duration923.279425/926.087974 seconds is a different first-request-to-last-response span.51 warmup requests completed separately on each service. Deadline cancellations and Mini's recovered allocator warnings during warmup remain in the raw evidence; neither is relabeled as a zero-error clean run. The workload seed does not fix the server RNG: actual Chat defaults were stochastic temperature1/top_p1/top_k-1 on both services.

The common requested boundary is TP8/DCP8/EP1,physical page64,context1048576,memory fraction0.90,state ratio0.31,max requests48,prefill/chunk32768,graph max48,FP8 E4M3 KV,BF16 recurrent and convolution state,and MXFP8 routed-expert activations. Native resolved Triton KDA/TRTLLM MLA prefill/CuteDSL MLA decode. Mini used its safe official-KDA adapter and mature MoE/MLA kernels. Identical configuration ratios do not make the two cache allocators equal: actual native logs show372 Mamba slots and32.59GB KV/rank; Mini shows48working+312cached+0scratch,9.74GiB state budget and31.42GiB KV/rank. Cache statistics therefore remain reported.

The baseline runtime is exactly production290ad026 runtime bytes (safe decode implementationa3f3341f); this documentation commit changes no runtime. The pinned reference is SGLangbaa9befa381511cd4d56f83980951e5a45fffc6d,FlashInfer0.7.1,Torch2.14.1+cu130,AIPerf0.13.0. Model config SHA256 is9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213. Checkpoint metadata and shard sizes are pinned; no full weight-content hash is claimed.

Separate greedy offline GSM8K cohorts scored1284/1319 on this Mini and the reference; Mini had2invalid/1truncated responses versus0/0. These are distinct quality observations, not evidence of token identity or quality on this stochastic serving scenario. Strict token/logit equality is diagnostic; correctness requires matching algorithm,precision and state-update semantics.

Optimization work remains isolated: generic DistributedImpl GEMM-all-gather/add adaptation,MoE workload-bound preparation,FlashInfer prefill integration,explicit merged KDA projections,and generic native sigmoid gating. Each candidate must retain its own component/model/serving evidence. No candidate is promoted by this baseline measurement.

Remote evidence is under experiment/recovery-20261110/serving-safe-baseline-paired-prepared, GPUq42. The accompanying JSON pins the full paired audit and root measurement receipt. Original exports,raw records,phase logs,server configuration,cache metrics and source/calculator hashes are retained remotely.
