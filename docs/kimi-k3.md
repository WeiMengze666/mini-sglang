# Kimi-K3 text inference

Kimi-K3 runs on eight B300 GPUs with Mini's packaged kernels. Its standalone serving run delivered 327.02 output tokens/s against 328.86 for SGLang on the specified 900-second AgentX workload. The comparison includes client drain and tail latency. [Serving results](kimi-k3-performance.md) · [Machine-readable results](validation/kimi-k3-standalone-serving.json)

## Following a request through the runtime

The model declares the attention algorithm for each layer. `HybridAttnBackend` builds one set of packed query offsets and gives the KDA and MLA backends their own metadata. KDA selects its prefill or decode kernel when it runs the layer. For the measured cache policy, prefill uses FlashInfer with BF16 state checkpoints and decode uses the indexed Triton recurrence. MLA uses FlashInfer paged attention and combines the DCP ranks' partial outputs using their log-sum-exp values.

The router defines the score, expert IDs and weights. Local expert implementations in `moe/experts` own weight packing, activation precision and GEMMs. An executor supporting EP declares the token payloads it can exchange. The dispatcher transports those rows and combines the results using the inverse mapping. After EP and expert-TP complete, only the shared expert output needs a TP reduction. The layer supplies its projection, normalizer and residual to the MoE epilogue.

## Run the measured configuration

Use the [saved server and AIPerf commands](../benchmark/online/profiles/kimi-k3-sglang.json) with the respective environments. The Mini runtime uses SGLang kernel sources at `baa9bef`, FlashInfer 0.7.1 and Torch 2.14.1+cu130. The reference server uses SGLang `94602c9`, FlashInfer 0.6.18 and Torch 2.13.0. Both measurements use AIPerf 0.13.0. The checkpoint and trusted tokenizer are at `/data/models/Kimi-K3`.

Install Mini with `python -m pip install -e ".[native]"` in its own environment. Mini calls FlashInfer directly and packages the selected CUDA and Triton kernels used by KDA, MLA, MoE and MNNVL. The K3 reasoner and tool-call parser run in-process. The published baseline used the original external kernel checkout. Keep the reference server in its own environment. [Kernel source and licenses](../python/minisgl/kernel/licenses/UPSTREAM.txt) · [Reference server revision](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e)

Both measured servers used the following CUDA and NCCL environment on this host.

```bash
export NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_GRAPH_MIXING_SUPPORT=0
export CUDA_DEVICE_MAX_CONNECTIONS=8 TRTLLM_ENABLE_PDL=1

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

EP is 1 for this comparison. The checkpoint stores MXFP4 expert weights. The selected expert kernels quantize activations to MXFP8. MLA token pages store FP8 E4M3 and KDA keeps BF16 recurrent state. The recurrent pool derives its capacity from the post-load memory budget. A state/KV ratio of 0.31 assigns a fraction `0.31 / 1.31` of that budget to recurrent slots. Working requests, overlap tracking and reusable prefix snapshots draw from the same pool. In the measured runs Mini allocated 367 shared state slots and 19,963,904 logical KV tokens. SGLang allocated 372 slots and 20,252,160 logical KV tokens. Each 64-token physical page per rank spans 512 logical tokens across DCP8.

## Where to extend the runtime

| Responsibility | Implementation |
| --- | --- |
| Model composition and checkpoint translation | `models/kimi_k3.py` wires projection, residual and expert layers. The model configuration supplies the layer predicate and projection dimensions. |
| Attention dispatch | `attention/base.py` composes prefill and decode backends. The hybrid backend selects KDA or MLA by layer ID. Algorithm packages adapt the native kernels and own their metadata and workspaces. |
| Token and recurrent state | `kvcache/mla_pool.py` owns logical-to-physical MLA pages. `recurrent_pool.py` owns state leases and checkpoints. `hybrid_cache.py` matches a prefix up to a restorable recurrent checkpoint. |
| Parallel groups and communication | `distributed/layout.py` derives DCP, expert and expert-tensor coordinates from TP. `DistributedImpl` selects a collective provider, including MNNVL. DCP gathers the required decode heads and combines partition results. |
| Expert execution | `moe/experts/flashinfer.py` adapts expert arithmetic. `moe/dispatch.py` handles dispatch and combine for the optional `flashinfer-ep` backend. |
| Request lifecycle | `scheduler/` owns admission, chunking, page allocation and completion. The frontend passes chat formatting to the tokenizer and selects reasoning and tool parsers by name. |

Native KDA decode is selected by the attention backend. The MLA package uses FlashInfer/TRT-LLM and CuTe-DSL kernels. New models can reuse these packages by supplying the same layer and cache contracts. [Attention backends](../python/minisgl/attention/README.md) · [Distributed backends](distributed-backends.md) · [Tokenizer and chat](kimi-k3-tokenizer.md)

## Validation

The released checkpoint completed all 1,319 GSM8K test questions. Every run used the same pretokenized inputs and gold answers, greedy decoding, batch size 8, a 4,096-token context and a 1,024-token output limit. Prefix reuse was disabled. The scorer was rerun on every saved response.

| Run | Correct | Accuracy | Invalid final answers | Reached the output limit |
| --- | ---: | ---: | ---: | ---: |
| Mini, EP1 | 1,283 | 97.27% | 0 | 0 |
| Mini, EP2 | 1,283 | 97.27% | 0 | 1 |
| Mini, EP8 | 1,280 | 97.04% | 1 | 1 |
| SGLang, with worker precision records | 1,281 | 97.12% | 1 | 2 |
| SGLang, first run | 1,281 | 97.12% | 0 | 0 |

Mini EP1 and the observed SGLang run gave the same numeric answer on 1,310 questions. Five were correct only on Mini and three only on SGLang. Their entire generated token sequences matched on 654 questions. The two official runs produced identical token sequences on 661. All eight workers on each side reported the same Torch math settings. Mini’s real sampling rows had finite logits throughout. Combined with the layer and state checks below, the results support accepting the measured configuration for text inference. [Paired quality evidence](validation/kimi-k3-delivery-quality.json)

The combined CPU/Gloo run passed 2,442 tests, skipped 780 and retained two expected differences against HF. Coverage includes dense and packed checkpoint loading, full versus chunked prefill, concurrent requests, state restoration, slot reuse and cancellation. Independent Moonshot and FLA checks compare per-layer outputs, attention residuals and recurrent updates. The reference tokenizer and parsers also run against the released checkpoint. [Reference checks and known HF differences](kimi-k3-official-cpu-reference.md)

In the full quality runs, all eight ranks used the native FlashInfer expert dispatcher for EP2 and EP8. The 92 routed layers each loaded 448 local experts with four-way tensor parallelism at EP2 and 112 full-width local experts at EP8. One EP2 response reached the output cap after giving the correct answer. One EP8 response reached the same cap without a final answer.

A separate eight-GPU HTTP lifecycle run on ancestor `4436c21` exercised EP2 and EP8. Each layout completed 102 eight-token requests with cancellation and slot reuse. The serving result above uses TP8/DCP8/EP1. Historical results and earlier precision configurations are recorded separately. [Validation history](kimi-k3-history.md)

### Reproduce the paired GSM8K run

Use the pinned [GSM8K test split](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl). Prepare the token IDs once in the Mini environment. The evaluator checks the dataset digest before applying the checkpoint chat template with thinking disabled.

```bash
python tests/validation/gsm8k_suite.py --prepare --model /data/models/Kimi-K3 \
  --dataset test.jsonl --cases cases.json
```

Use a fresh output directory for each role. Run these commands in their respective environments on the eight-GPU allocation. Both use the same case file, deterministic greedy sampling and native precision defaults.

```bash
common=(--model /data/models/Kimi-K3 --cases cases.json --batch-size 8 \
  --max-running-requests 48 --max-new-tokens 1024 --max-context 4096 \
  --tp-size 8 --dcp-size 8 --ep-size 1 --moe-backend auto --mxfp4-backend flashinfer \
  --activation-precision mxfp8 --kv-cache-dtype fp8_e4m3 --recurrent-state-dtype bfloat16 \
  --cuda-graph --cuda-graph-max-bs 48)
torchrun --standalone --nnodes=1 --nproc-per-node=8 tests/validation/gsm8k_suite.py \
  "${common[@]}" --engine mini --attention-backend auto --distributed-backend mnnvl \
  --distributed-provider minisgl.distributed.multimem.provider --output quality-mini
python tests/validation/gsm8k_suite.py "${common[@]}" --engine sglang --attention-backend auto \
  --expected-reference-revision 94602c9c2b7cbdb8efd5c52802dac6a1c180089e \
  --weight-loader-prefetch-checkpoints --weight-loader-prefetch-num-threads 8 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":12}' \
  --output quality-official
python tests/validation/compare_accuracy.py --left quality-mini --right quality-official \
  --expected-examples 1319 --output quality-comparison.json
```

### Reproduce an expert-parallel quality run

Keep the official output from the EP1 command above. For Mini, change the values of `--ep-size` and `--moe-backend` in the common options to `2` and `flashinfer-ep`, and write to `quality-mini-ep2`. With TP8, this gives two expert partitions and four-way tensor parallelism within each expert. Use `8` for one full expert partition per rank and write to `quality-mini-ep8`.

```bash
python tests/validation/compare_accuracy.py --left quality-mini-ep2 --right quality-official \
  --expected-ep-sizes 2 1 --output quality-ep2-comparison.json
python tests/validation/compare_accuracy.py --left quality-mini-ep8 --right quality-official \
  --expected-ep-sizes 8 1 --output quality-ep8-comparison.json
```

The size pair follows the order of the two inputs. The scorer checks the requested EP sizes, all other precision and layout fields, the generation protocol and the full 1,319-question split.
