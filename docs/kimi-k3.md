# Kimi-K3 text support

K3 composes reusable KDA, compressed MLA, attention residual and latent MoE
operations. Model composition stays in `models/`; attention, cache ownership,
expert compute and communication remain framework components. Code structure
and correctness take priority over performance.

## Architecture and kernels

`HybridAttnBackend` in `attention/base.py` dispatches through
`ModelConfig.is_kda_layer(layer_id)`. KDA and MLA implementations live in their
respective packages; `PrefillDecodeAttnBackend` composes execution stages.
`KDALayer` owns projections/kernel plans, while its backend adapts native
convolution, recurrence, gated normalization and fused decode operators.
MLA adapters manage projections, physical page geometry and workspaces around
FlashInfer/TRTLLM prefill and CuTe-DSL decode kernels.

`HybridCache` composes MLA token pages and recurrent working/checkpoint state.
Allocation/ownership stay in `kvcache/`; kernel adaptation is under `kernel/`.
`DistributedInfo` describes rank groups. `ParallelLayout` derives TP, DCP and
expert groups; `ShardSpec` describes tensor slices. DCP widens MLA decode head
ownership; KDA remains sharded across the full TP group.

`Linear` owns dense/packed storage. `moe/` selects expert compute from format
and activation precision; routing and collectives use shared interfaces.
There is no standalone quantization package. For the validated BF16 path,
`attention-backend=auto` selects the native hybrid composition, and packed
BF16 experts select Marlin. Native precision policy is scoped and restored.
The actual eight-rank K3 audit observed no Mini numerical CUDA/Triton calls;
Mini's PyNCCL FFI is a communication shim. Other/reference backends are separate.

## Current validation

[Versioned evidence](validation/kimi-k3-native.json) records the baseline and
pending work. The official reference is SGLang commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`.

The paired complete GSM8K test uses the same 1,319 prompts/gold answers:
TP8/DCP2/EP8, batch 8, graphs through 8, BF16 model/MoE activations/KV, FP32 KDA
state and BF16 convolution state. Mini scores 1281/1319; SGLang 1282/1319.
Both have zero invalid/truncated answers; 1315 final answers agree. This is
quality evidence for this configuration, not target serving performance.
The frozen native baseline passes 806 CPU checks with 611 GPU-only skips.
The current protocol/workload suite passes 841 checks with the same GPU skips,
including 20 opt-in checks against the pinned official parsers and released
checkpoint tokenizer on CPU.
GPU component checks cover native outputs/states, graph replay, scoped DCP,
BF16/FP8 MLA pages, convolution, SiTU and attention residuals.

Strict logits/greedy equality is diagnostic. Final acceptance requires correct
semantics, precision boundaries, state updates and model quality. Verified
floating reduction/rounding differences, including native tuning choices, are
allowed. Reference data and original diagnostic results remain unchanged.

`tests/validation/tensor_trace.py` is an opt-in eager observer. It saves logical
intermediates with dtype/shape/stride, alias/owner metadata, state indices and
Triton configurations; NVTX links them to Nsight's actual CUDA launch geometry.
Weights, lookup tables and workspaces retain metadata; indexed pools save
referenced state rows. GPU end-to-end trace validation remains pending.

## Remaining work

The [requested serving profile](kimi-k3-performance.md) is a separate target:
TP8/DCP8/EP1, MXFP8 expert activations, FP8 KV and BF16 KDA state, 1M context,
48 requests and 32K chunks. Its correctness/performance acceptance is unfinished.
Current real-checkpoint chunk/cache/cancellation/reuse checks, FlashInfer expert
adaptation, exact state-buffer budgeting, built-in generic MNNVL and optimized
EP exchange remain priorities. The BF16 task benchmark establishes no claim of
near-SGLang serving performance. Earlier refactor evidence remains in
`docs/validation/` and Git history.

The exact AIPerf AgentX command is saved in the serving profile. A generic
`benchmark/online/run_profile.py` launches a saved server/benchmark pair and
preserves commands, startup logs, server info, metrics and raw exports. It must
run inside one GPUq job and cleans only its own process groups. Frontend usage
now counts real template/input and generated token IDs, including empty decoded
fragments and EOS; streaming usage follows `stream_options.include_usage`.
`/metrics` exposes frontend request counters and completed token totals, not GPU
or cache telemetry. Chat formatting and output parsing reuse the trusted checkpoint tokenizer and
optional native reasoning/tool parsers behind generic interfaces. Per-request
streaming state, tool messages, template options and the served model alias are
covered on CPU; generation stop IDs come from the checkpoint generation config
without changing its corpus EOS. Target-profile GPU serving/cache/precision
qualification and Mini AIPerf measurements remain pending.

The canonical native S0 AIPerf run is now recorded in
[the serving report](kimi-k3-performance.md#native-s0-result): 378.53 output
tokens/s, TTFT P50 590.35 ms and ITL P50 26.76 ms. Mini R0 remains pending.
Cache reads are exposed through standard prompt-token details and a completed
cache-read token counter, preserving the original admission hit across chunks.
