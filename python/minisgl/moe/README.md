# MoE execution

Routing produces global expert IDs and mixing weights. Local expert execution
returns an FP32 mixture before collectives and optional model-level normalization.

`create_expert_executor` selects from weight format, layout and backend policy:
portable expert calls, Triton packed kernels in `mxfp4.py`, or the native library
adapter in `flashinfer.py`. Models use the same forward interface in each case.
The portable executor in `base.py` also handles empty or remote-only partitions.

`fused.py` implements the existing `BaseMoeBackend` path that combines top-k
selection and dense expert execution. The expert-executor entry point accepts
already-routed IDs so alternative routers can share local compute without
duplicating routing or changing collective boundaries.

`routing.py` and the block alignment primitive remain shared. Expert compute
contains no Kimi-K3, attention, TP/DCP topology, or residual-layer assumptions.
