# FlashInfer packed MXFP4 backend design

> Layout update: the former quantization package is superseded by unified
> `layers/linear.py`, shared `layers/weight.py`, and expert execution in
> `moe/mxfp4.py` / `moe/flashinfer.py`. The old file names below record
> the original design; current ownership is described in the module READMEs.

## Goal

Add an opt-in FlashInfer TRTLLM FP4 grouped-expert executor to mini-sglang while preserving the existing BF16 Triton executor as the default. The backend must be generic over local/global expert partitions and activation policy; no Kimi model name may appear in quantization or executor modules.

## Decisions

- `mxfp4_backend=triton` remains the shipping default for grouped MXFP4 experts.
- `mxfp4_backend=flashinfer` selects a separate packed expert executor and requires CUDA plus the installed FlashInfer TRTLLM FP4 API.
- `mxfp4_activation_precision=bf16` is valid only when the selected activation has a FlashInfer BF16 tactic. SiTU requires `mxfp4_activation_precision=mxfp8`.
- Weight preparation happens once after loading: stack local expert weights, decode E8M0 scale bytes, apply the official FlashInfer byte and scale permutations, and retain transformed tensors.
- Expert IDs remain global at the routing boundary. The executor maps them to contiguous local IDs and passes global count, local offset, and local count to FlashInfer.
- The FlashInfer path is explicit and opt-in. It does not replace the shipping BF16 Triton path or weaken its strict gate.

## Interfaces

`MXFP4Linear.set_backend(backend, activation_precision='bf16')` stores the selected generic packed-expert execution policy.

`MXFP4Experts` remains the portable Triton implementation. A new `FlashInferMXFP4Experts` owns transformed stacked weights and exposes the same `forward(x, weights, ids) -> FP32 [tokens, routed_width]` contract.

`EngineConfig` adds `mxfp4_activation_precision` with values `bf16` and `mxfp8`; invalid values and unsupported backend/activation combinations fail during startup.

## Validation

- CPU tests cover policy validation without importing CUDA or FlashInfer.
- GPU tests use generated packed weights and a real Kimi layer-12 expert fixture.
- Full DCP/EP strict BF16 tests remain mandatory for the shipping Triton path.
- A paired GPUq performance run compares Triton BF16 and FlashInfer MXFP8 at the same DCP/EP and CUDA Graph boundary.
