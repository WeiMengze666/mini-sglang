# Kimi-K3 text inference

The released Kimi-K3 checkpoint runs through Mini's text server on eight B300 GPUs. The measured runtime is `c3c8b04`. It produced 326.28 output tokens/s against 328.86 for SGLang on the 900-second AgentX workload. The comparison includes tail latency and the client drain behavior. [Serving results](kimi-k3-performance.md) · [Machine-readable results](validation/kimi-k3-delivery-16k.json)

## Run the measured configuration

Use the [saved server and AIPerf commands](../benchmark/online/profiles/kimi-k3-sglang.json) with the respective environments. The Mini runtime uses SGLang kernel sources at `baa9bef`, FlashInfer 0.7.1 and Torch 2.14.1+cu130. The reference server uses SGLang `94602c9`, FlashInfer 0.6.18 and Torch 2.13.0. Both measurements use AIPerf 0.13.0. The checkpoint and trusted tokenizer are at `/data/models/Kimi-K3`.

```bash
python -m minisgl --trust-remote-code \
  --model-path /data/models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 \
  --tp-size 8 --dcp-size 8 --page-size 64 --context-length 1048576 \
  --mem-fraction-static 0.90 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer --mamba-full-memory-ratio 0.31 \
  --max-running-requests 48 --max-prefill-tokens 16384 --chunked-prefill-size 16384 \
  --cuda-graph-max-bs-decode 48 --prefill-decode-interval 8 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-metrics --enable-cache-report \
  --distributed-timeout 1800 --weight-loader-prefetch-checkpoints \
  --weight-loader-prefetch-num-threads 8 --host 0.0.0.0 --port 38080
```

EP is 1 for this comparison. The checkpoint stores MXFP4 expert weights. The selected expert kernels quantize activations to MXFP8. MLA token pages store FP8 E4M3 and KDA keeps BF16 recurrent state. The recurrent pool derives its capacity from the post-load memory budget. A state/KV ratio of 0.31 assigns a fraction `0.31 / 1.31` of that budget to recurrent slots. Working requests, overlap tracking and reusable prefix snapshots draw from the same pool.

## Where to extend the runtime

| Responsibility | Implementation |
| --- | --- |
| Model composition and checkpoint translation | `models/kimi_k3.py` wires projection, residual and expert layers. The model configuration supplies the layer predicate and projection dimensions. |
| Attention dispatch | `attention/base.py` composes prefill and decode backends. The hybrid backend selects KDA or MLA by layer ID. Algorithm packages adapt the native kernels and own their metadata and workspaces. |
| Token and recurrent state | `kvcache/mla_pool.py` owns logical-to-physical MLA pages. `recurrent_pool.py` owns state leases and checkpoints. `hybrid_cache.py` matches a prefix up to a restorable recurrent checkpoint. |
| Parallel groups and communication | `distributed/layout.py` derives DCP, expert and expert-tensor coordinates from TP. `DistributedImpl` selects a collective provider, including MNNVL. DCP gathers the required decode heads and combines partition results. |
| Expert execution | `moe/flashinfer.py` adapts expert arithmetic. `moe/token_dispatch.py` handles dispatch and combine for the optional `flashinfer-ep` backend. |
| Request lifecycle | `scheduler/` owns admission, chunking, page allocation and completion. The frontend passes chat formatting to the tokenizer and selects reasoning and tool parsers by name. |

Native KDA decode is selected by the attention backend. The MLA package uses FlashInfer/TRT-LLM and CuTe-DSL kernels. New models can reuse these packages by supplying the same layer and cache contracts. [Attention backends](../python/minisgl/attention/README.md) · [Distributed backends](distributed-backends.md) · [Tokenizer and chat](kimi-k3-tokenizer.md)

## Validation

Remote CPU coverage includes dense and packed checkpoint loading, full versus chunked prefill, concurrent requests, state restoration, slot reuse and cancellation. Independent Moonshot and FLA checks compare per-layer outputs, attention residuals and recurrent updates. The reference tokenizer and parsers also run against the released checkpoint. [Reference checks and known HF differences](kimi-k3-official-cpu-reference.md)

A separate eight-GPU functional run on ancestor `4436c21` exercised EP2 and EP8. Each layout completed 102 eight-token requests with cancellation and slot reuse. The serving result above uses TP8/DCP8/EP1. Historical results and earlier precision configurations are recorded separately. [Validation history](kimi-k3-history.md)
