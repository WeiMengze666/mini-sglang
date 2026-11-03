# MoE backends

The model describes routing, expert projections, shared branches and optional
latent normalization. `BaseMoeBackend` selects expert execution after loading
and combines distributed partials before the model applies nonlinear operations.
Its collective calls use `TensorParallel`, which delegates to `DistributedImpl`.
Kernel adapters do not construct another framework's scheduler or model runner.

`--moe-backend fused` retains the portable/dense implementation and selects
packed experts using their format and configured linear backend.
`--moe-backend marlin` opts into the mature Marlin MXFP4 GEMMs, gated activation
and top-k sum from the optional SGLang kernel SDK. Integer route planning remains
model independent. This backend requires BF16 activations and owns repacked
weights, so the loader does not retain them in a second raw-weight arena.
The initial integration targets SGLang commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`; CPU fallback needs no optional SDK.

Expert executors may declare `input_dtype`. The projection producer retains that
precision until the executor consumes it. In particular, FlashInfer MXFP8
quantization can consume FP32 latent projection accumulators; rounding them to
BF16 before quantization changes the activation contract. Marlin consumes BF16.
An unsupported FP32 projection format fails explicitly instead of silently
rounding away the requested precision.

Activation quantization, KV-cache storage, recurrent-state storage and
convolution storage are independent precision choices. Numerical and performance
comparisons must match all four, as well as TP/DCP/EP topology and the request
workload. Passing an adapter's component tests is not full-model acceptance.

The Marlin adapter zero-pads unsupported small hidden widths to the mature
GEMM portfolio's alignment, using zero FP4 codes and unit E8M0 scales. Logical
model dimensions are preserved at the boundary. Aligned widths reuse the
original tensors, including the released checkpoint's expert hidden width.
No additional numerical kernel is defined by this padding adapter.
