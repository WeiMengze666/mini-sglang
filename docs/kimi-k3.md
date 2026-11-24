# Kimi-K3 text inference

Kimi-K3 runs on eight B300 GPUs with Mini’s packaged kernels. The release wheel and the retained SGLang reference each delivered **328.86 output tokens/s** on the 900-second AgentX workload. Mini’s p99 inter-token latency was 136.71 ms versus 86.65 ms for SGLang. The first standalone snapshot reached 327.02 output tokens/s. [Serving results](kimi-k3-performance.md) · [Machine-readable measurements](validation/kimi-k3-standalone-serving.json)

After the runtime refactor, commit `76f51ef` delivered **324.04 output tokens/s**
on the same profile, 98.53% of the retained SGLang run. Median TTFT was 1.05 s
for Mini and 1.36 s for SGLang. P95 TTFT was 14.44 s and 10.83 s respectively.
[Refactor validation and latency results](validation/kimi-k3-runtime-refactor.json)

## Following a request through the runtime

The model declares the attention algorithm for each layer. `HybridAttnBackend` builds one set of packed query offsets and gives the KDA and MLA backends their own metadata. KDA selects its prefill or decode kernel when it runs the layer. For the measured cache policy, prefill uses FlashInfer with BF16 state checkpoints and decode uses the indexed Triton recurrence. MLA uses FlashInfer paged attention and combines the DCP ranks' partial outputs using their log-sum-exp values.

The router defines the score, expert IDs and weights. Local expert implementations in `moe/experts` own weight packing, activation precision and GEMMs. An executor supporting EP declares the token payloads it can exchange. The dispatcher transports those rows and combines the results using the inverse mapping. After expert exchange and expert tensor parallelism complete, the MoE epilogue reduces the shared expert output across TP. The layer supplies its projection, normalizer and residual to the MoE epilogue.

## Loading a checkpoint

The model class selects a checkpoint format. The [streaming reader](../python/minisgl/checkpoint/loader.py) resolves the snapshot, checks the index and packed/scale pairs, filters remote experts and reads rank-local slices. KDA, MLA and gated experts declare the row or column ranges next to their projections. The same ranges size the local weights and select which bytes to read. A K3 expert’s `w2` uses the expert tensor-parallel group for its input columns. The reader translates those logical columns to the matching packed bytes and block scales, including when they reside in separate files.

Llama and Qwen use the [Hugging Face checkpoint layout](../python/minisgl/checkpoint/huggingface.py). It combines rank-local Q/K/V and gate/up tensors and stacks local experts in global-ID order. KV head replication follows the QKV layer’s head partition. [Checkpoint interfaces and extension points](../python/minisgl/checkpoint/README.md)

### Checkpoint refactor validation

The standalone wheel from `e81cc31` was checked against the previous delivery on the released K3 index. All 497,220 source names and all 24 rank layouts (TP8/DCP8, EP1/2/8) produced identical text names, expert ownership and logical slices. Reading 31 selected source tensors in three of those layouts also reproduced the previous raw bytes for all 63 emitted tensors. A separate two-layer Hugging Face Qwen2 checkpoint bound all 17 local parameters on each of four ranks with identical shapes, dtypes and bytes.

| B300 layout | GSM8K questions | Correct | Previous Mini on these questions | Full token sequences equal to previous Mini |
| --- | ---: | ---: | ---: | ---: |
| EP2, MNNVL | 1,319 | 1,283 | 1,283 | 1,319 |
| EP1, PyNCCL | 32 | 31 | 31 | 32 |

The EP1 run uses the first 32 questions from the full split. The online AgentX measurement in this guide belongs to the earlier wheel. [Checkpoint validation evidence](validation/kimi-k3-checkpoint-layout.json)

## Run the measured configuration

Use the [saved server and AIPerf commands](../benchmark/online/profiles/kimi-k3-sglang.json) with the respective environments. The Mini runtime uses SGLang kernel sources at `baa9bef`, FlashInfer 0.7.1 and Torch 2.14.1+cu130. The reference server uses SGLang `94602c9`, FlashInfer 0.6.18 and Torch 2.13.0. Both measurements use AIPerf 0.13.0. The checkpoint and trusted tokenizer are at `/data/models/Kimi-K3`.

Install Mini with `python -m pip install -e ".[native]"` in its own environment. Mini calls FlashInfer directly and packages the selected CUDA and Triton kernels used by KDA, MLA, MoE and MNNVL. The K3 reasoner and tool-call parser run in-process. The historical 16K baseline on Mini `c3c8b04` used an external kernel checkout. Keep the reference server in its own environment. [Kernel source and licenses](../python/minisgl/kernel/licenses/UPSTREAM.txt) · [Reference server revision](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e)

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
| Model composition and checkpoint translation | [`models/kimi_k3/model.py`](../python/minisgl/models/kimi_k3/model.py) composes the layers. [`checkpoint.py`](../python/minisgl/models/kimi_k3/checkpoint.py) maps the released text names, quantization aliases and historical KDA head layout to their parameters. |
| Attention dispatch | `attention/base.py` composes prefill and decode backends. The hybrid backend selects KDA or MLA by layer ID. Algorithm packages adapt the native kernels and own their metadata and workspaces. |
| Token and recurrent state | `kvcache/mla_pool.py` owns logical-to-physical MLA pages. `recurrent_pool.py` owns state leases and checkpoints. `hybrid_cache.py` matches a prefix up to a restorable recurrent checkpoint. |
| Parallel groups and communication | `distributed/layout.py` derives DCP, expert and expert-tensor coordinates from TP. `DistributedImpl` selects a collective provider, including MNNVL. DCP gathers the required decode heads and combines partition results. |
| Expert execution | `moe/experts/flashinfer.py` adapts expert arithmetic. `moe/dispatch.py` handles dispatch and combine for the optional `flashinfer-ep` backend. |
| Request lifecycle | `scheduler/` owns admission, chunking, page allocation and completion. The frontend passes chat formatting to the tokenizer and selects reasoning and tool parsers by name. |

Native KDA decode is selected by the attention backend. The MLA package uses FlashInfer/TRT-LLM and CuTe-DSL kernels. New models can reuse these packages by supplying the same layer and cache contracts. [Attention backends](../python/minisgl/attention/README.md) · [Distributed backends](distributed-backends.md) · [Tokenizer and chat](kimi-k3-tokenizer.md)

## Validation

All three Mini expert layouts completed the 1,319-question GSM8K test split on eight B300 GPUs. Each used the same pretokenized inputs and gold labels as the retained official reference, greedy decoding, batch size 8, a 4,096-token context and a 1,024-token output limit. Prefix reuse was disabled. The scorer reparsed every saved answer and Mini checked all real sampling rows for finite logits.

| Runtime | Correct / 1,319 | Accuracy | Invalid final answers | Reached output limit | Entire token sequence equal to official |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGLang TP8/DCP8/EP1 | 1,281 | 97.12% | 1 | 2 | Reference |
| Mini EP1, PyNCCL | 1,281 | 97.12% | 1 | 0 | 635 |
| Mini EP2, MNNVL | 1,283 | 97.27% | 0 | 1 | 650 |
| Mini EP8, MNNVL | 1,280 | 97.04% | 1 | 1 | 644 |

EP1 used the release wheel and the TP transport selected by the AgentX server. It gave the same valid numeric answer as SGLang on 1,310 questions. Mini alone answered 4 correctly and SGLang alone answered 4 correctly. Two runs of the official server independently scored 1,281 and produced identical token sequences on 661 questions. [Scores, paired differences, execution precision and artifact hashes](validation/kimi-k3-standalone-quality.json)

The EP2 and EP8 evaluations used the standalone wheel built at `4ad9d9d` and the FlashInfer expert dispatcher. Every generated token sequence in each layout matched its earlier full-model Mini run. The checkpoint has 92 routed layers. With TP8, EP2 assigns 448 experts to each partition with four-way expert tensor parallelism. EP8 assigns 112 full-width experts per rank. The AgentX throughput comparison uses EP1.

The CPU suite, including multi-process Gloo and independent HF/FLA references, recorded 2,397 passes, 701 skips and two expected HF differences. On B300, 81 direct comparisons with upstream covered KDA gates, state updates, chunk continuation, graph replay, expert routing and arithmetic. Fifteen DCP tests exercised shard reduction against a dense reference and graph replay. Separate checks covered MNNVL collectives and FlashAttention 4 on dense and MoE models. [Component results](validation/kimi-k3-standalone-quality.json) · [Known HF differences](kimi-k3-official-cpu-reference.md) · [Earlier validation](kimi-k3-history.md)

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
  "${common[@]}" --engine mini --attention-backend auto --distributed-backend pynccl \
  --weight-loader-prefetch-checkpoints --weight-loader-prefetch-num-threads 8 --output quality-mini
python tests/validation/gsm8k_suite.py "${common[@]}" --engine sglang --attention-backend auto \
  --expected-reference-revision 94602c9c2b7cbdb8efd5c52802dac6a1c180089e \
  --weight-loader-prefetch-checkpoints --weight-loader-prefetch-num-threads 8 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":12}' \
  --output quality-official
python tests/validation/compare_accuracy.py --left quality-mini --right quality-official \
  --expected-examples 1319 --output quality-comparison.json
```

### Reproduce an expert-parallel quality run

Keep the official output from the EP1 command above. For Mini, change the values of `--ep-size` and `--moe-backend` in the common options to `2` and `flashinfer-ep`, and write to `quality-mini-ep2`. Both EP runs use `--distributed-backend mnnvl --distributed-provider minisgl.distributed.multimem.provider`. With TP8, this gives two expert partitions and four-way tensor parallelism within each expert. Use `8` for one full expert partition per rank and write to `quality-mini-ep8`.

```bash
python tests/validation/compare_accuracy.py --left quality-mini-ep2 --right quality-official \
  --expected-ep-sizes 2 1 --output quality-ep2-comparison.json
python tests/validation/compare_accuracy.py --left quality-mini-ep8 --right quality-official \
  --expected-ep-sizes 8 1 --output quality-ep8-comparison.json
```

The size pair follows the order of the two inputs. The scorer checks the requested EP sizes, all other precision and layout fields, the generation protocol and the full 1,319-question split.
