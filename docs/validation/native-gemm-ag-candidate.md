# Native GEMM/all-gather adapter candidate

This independent topic adds a tensor-only optional capability, DistributedImpl.try_linear_all_gather_add, through the communicator and TensorParallel facades. Linear owns completed-load weight preparation and the ordinary fallback; LatentMoE composes the already reduced and normalized projection with the shared contribution. Provider eligibility stays outside model code.

The adapter reuses the pinned SGLang GEMM/all-gather implementation and existing MNNVL communicator. It does not implement a numerical kernel. The current official provider supports its documented BF16 TP8/SM100 or SM103 shape domain, including rows1..12; other supported framework paths retain the ordinary projection/add path. Strict BF16 keeps its original operation. These source contracts are not proof of GPU execution or universal model support.

CPU verification:935 passed,44 existing skips. GPUq component job43 has been queued; no GPU result is included or accepted by this publication. Full-model quality and canonical serving performance remain unqualified. This is separate from the MoE preparation and reduce/norm candidates: the default290ad026 runtime is unchanged, and the norm candidate is not included. The adjacent JSON records exact source/patch and test identities.
