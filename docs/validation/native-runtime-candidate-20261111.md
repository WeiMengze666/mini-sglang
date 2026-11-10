# Native runtime preparation and normalized reductions: isolated candidate

This branch integrates mature kernels through the existing framework boundaries. It is an experimental candidate, not the default branch.

- DistributedImpl adds optional try_all_reduce_norm(x, weight, eps). Unsupported calls return before writes or collectives. TensorParallel and named communicators keep their existing group ownership.
- The native MNNVL provider binds the pinned SDK module to its own communicator instance. It uses no SDK World or global communicator registry. Compiled shape and device restrictions stay in the provider.
- The MoE backend tries the normalized latent reduction, then reduces the shared contribution once. Unsupported inputs retain the original packed reduction and ordinary normalization.
- Public BaseOP preparation hooks let a backend inspect and prepare expert executors after weights load. Engine invokes preparation before cache budgeting only under the actual fresh-worker ownership contract and default precision. Strict precision and embedding calls without that owner retain their previous path.
- The fresh launcher owns fatal startup exits and peer supervision. Unknown preparation failures do not attempt recovery collectives.

The candidate reuses SGLang SDK baa9befa381511cd4d56f83980951e5a45fffc6d and FlashInfer 0.7.1. It adds no numerical kernel, model-name branch in distributed/MoE logic, private model visitor or new serving CLI switch.

## Evidence and limitations

The canonical local expert geometry passed an eight-rank component check. Of 6,537,216 independently reloaded output elements, 12 differ between default and prepared tactics, by at most 3 BF16 ULP. Graph replay, input/weight identity and actual runtime cache lookups were checked. This is local packed TP8/EP1 evidence, not EP exchange or full prefill qualification.

The normalized-reduction adapter passed all eight ranks: 920 saved output pairs agree with the direct SDK entry and satisfy the independent FP64 error envelope. Tests cover batches 1-48, two independent same-sized groups, side streams, graph replay, layout normalization and unsupported zero-write behavior. This does not include deferred expert finalization or shared-branch overlap.

The complete greedy GSM8K cohort scored **1279/1319**, versus **1284/1319** for both official SGLang and the safe-decode Mini baseline. There are no invalid or truncated answers, but five questions regress with no correctness wins against that Mini baseline: 187, 306, 425, 796 and 1003. All responses and the fixed scorer are preserved. Component success and improved completion do not establish model-quality equivalence.

Preparation actually ran on all eight ranks: 92 expert preparation hooks per rank, followed by graph capture and observer restoration before evaluation. MoE lookups during capture hit the prepared cache; MLA default-tactic lookups are recorded separately.

A full original 393-entry, concurrency-48, 900-second AIPerf run is a **diagnostic measurement only**. No performance result is claimed here. The score regression remains unresolved, prepare-only ablation is being prepared, and this branch is not approved for default promotion. The default branch remains the independently verified safe-decode baseline.

Runtime files on this branch are byte-identical to the frozen GPUq38 candidate. Remote artifacts are under /home/axis/mengze/mini-sglang/experiment/recovery-20261110, including quality-moe-norm-prepared/review, moe-prepare-packed-root-review and native-reduce-norm-gate-prepared.
