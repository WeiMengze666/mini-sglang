# Projection and weight ownership

`Linear` owns dense and MXFP4 weight storage. `weight_format` is explicit:
packed uint8 alone cannot identify E2M1 nibble order or E8M0 scale semantics.
The selected format determines both checkpoint tensor names and the forward path.

- `linear.py`: projection operators, format selection, bounded Torch fallback.
- `linear_kernel.py`: dense BF16 and packed MXFP4 GEMV/GEMM.
- `weight.py`: aligned allocation, format decoding, logical shard reads, and
  compressed checkpoint selection rules shared by model adapters.
- `weight_kernel.py`: shared Triton E2M1/E8M0 decoding used by projections and experts.

Dense parameters remain `weight` and optional `bias`. Packed parameters remain
`weight_packed`, `weight_scale` and optional `bias`; a packed Linear does not
retain a decoded dense matrix. CPU/reference decoding uses bounded row chunks.

Model adapters supply checkpoint aliases, sharding and dense-only constraints.
They do not implement arithmetic or select model-specific projection kernels.
