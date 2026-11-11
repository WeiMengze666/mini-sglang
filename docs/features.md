# Features of Mini-SGLang

## Online Serving

Mini-SGLang supports online serving with an OpenAI-compatible API server. It provides the standard `/v1/chat/completions` endpoint, allowing seamless integration with existing tools and clients. For detailed command-line arguments and configuration options, run `python -m minisgl --help`.

## Interactive Shell Mode

For demonstration and testing purposes, an interactive shell mode is available. In this mode, users can input prompts directly, and the LLM will generate responses in real-time. The shell automatically caches chat history to maintain context. To clear the conversation history and start a new session, use the `/reset` command.

Example:

```bash
python -m minisgl --model-path "Qwen/Qwen3-0.6B" --shell
```

## Distributed Serving

To scale performance across multiple GPUs, Mini-SGLang supports Tensor Parallelism (TP). You can enable distributed serving by specifying the number of GPUs with the `--tp-size n` argument, where `n` is the degree of parallelism.

DCP (`--dcp-size`) partitions token ownership for context-sharded MLA and merges partial attention using output/LSE exchange. EP (`--ep-size`) partitions experts independently of their tensor shards, for both ordinary and latent MoE. The replicated-token scheduler combines local expert contributions by TP reduction. The integrated paged DCP path currently serves MLA; the exchange itself is independent of the attention algorithm.
Collective backends declare per-operation capabilities, so an optional backend can handle only supported all-reduces or gathers while the tested fallback handles the rest. Grouped MoE also accepts a generic RoutingPolicy; stable_moe_routing=true selects deterministic integer route grouping for calibrated references, while the default keeps the native grouping path.
PyNCCL uses its symmetric internal reduction buffer for the production path. A direct-buffer experiment was measured and rejected because it regressed complete Kimi decode throughput; no direct-buffer runtime knob is shipped.

## Supported Models

Our framework currently supports the following model architectures:

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) series
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) series (including MoE)
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) series
- Experimental Kimi-K3 text decoding, assembled from reusable KDA, gated MLA, residual and MoE layers. See [scope and validation](kimi-k3.md).

## Chunked Prefill

Chunked Prefill, a technique introduced by [Sarathi-Serve](https://arxiv.org/abs/2403.02310), is enabled by default. This feature splits long prompts into smaller chunks during the prefill phase, significantly reducing peak memory usage and preventing Out-Of-Memory (OOM) errors in long-context serving. The chunk size can be configured using `--max-prefill-length n`. Note that setting `n` to a very small value (e.g., 128) is not recommended as it may significantly degrade performance.

## Page Size

You can specify the page size of the system using the `--page-size` argument.

## Attention Backends

Mini-SGLang integrates high-performance attention kernels, including [`FlashAttention`](https://github.com/Dao-AILab/flash-attention) (`fa`), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer) (`fi`) and [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM) (`trtllm`). It supports using different backends for the prefill and decode phases to maximize efficiency. For example, on NVIDIA Hopper GPUs, `FlashAttention 3` is used for prefill and `FlashInfer` for decode by default.

You can specify the backend using the `--attention-backend` argument. If two values are provided (e.g., `--attention-backend fa,fi`), the first specifies the prefill backend and the second the decode backend. Note that some attention backend might override the user-provided page size (e.g. `trtllm` only supports page size 16,32,64).

## CUDA Graph

To minimize CPU launch overhead during decoding, Mini-SGLang supports capturing and replaying CUDA graphs. This feature is enabled by default. The maximum batch size for CUDA graph capture can be set with `--cuda-graph-max-bs-decode n`. Setting `n` to `0` disables this feature. Explicit buckets can be supplied with `--cuda-graph-bs 1 2 4 8`.

Hybrid recurrent attention requires explicit graph enablement and a compatible native backend. It uses exact batch sizes with eager fallback for uncaptured sizes, because padded rows would mutate request state. Capture temporarily borrows idle state slots at startup; replay resolves current request and page indirection.

## Radix Cache

Adopting the original design from [SGLang](https://github.com/sgl-project/sglang.git), Mini-SGLang implements a Radix Cache to manage the Key-Value (KV) cache. This allows the reuse of KV cache for shared prefixes across requests, reducing redundant computation. This feature is enabled by default but can be switched to a naive cache management strategy using `--cache naive`.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Illustration of Radix Attention from [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/).*

Hybrid prefix reuse preserves paired recurrent/convolution checkpoints alongside token pages. `--mamba-radix-cache-strategy extra_buffer` uses native tracking kernels and donates completed checkpoint slots to the radix tree. `--mamba-full-memory-ratio` divides the post-load state/KV memory budget; state capacity reserves a working slot plus one tracking slot per request, or two tracking slots with overlap. Remaining slots form the LRU state cache. Admission copies shared state into a private working slot. `--enable-unified-memory` is a separate compacting-arena mode and does not support native checkpoint tracking.

## Overlap Scheduling

To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in [NanoFlow](https://arxiv.org/abs/2408.12757). This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Illustration of Overlap Scheduling from [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).*

Overlap is enabled by default. `--disable-overlap-schedule` selects serial scheduling. Mutable hybrid pools preserve completed frontiers on the producer stream before the next forward and donate them only after the prior completion event succeeds. Working state, tracking leases and cached checkpoints have separate ownership. `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` also disables overlap for every pool.

`--max-prefill-tokens` and `--chunked-prefill-size` are independent shared batch budgets, following SGLang admission and physical-page charging. The current comparison recipe uses16384 for both; see `benchmark/online/profiles/kimi-k3-sglang.json`. Public names replaced the former options; no compatibility aliases are accepted.
