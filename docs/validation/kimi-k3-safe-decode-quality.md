# Official safe-gate decode: full-model quality baseline

The bounded KDA decode adapter now calls the official public safe-gate entry used by the pinned SGLang Triton backend. The implementation stays in the generic KDA backend; it adds no model-name condition or numerical kernel. Runtime source: a3f3341f451ce660a0316d87beefd05a05a20c11. Reference SDK: baa9befa381511cd4d56f83980951e5a45fffc6d.

## Full 1319-case cohort

| Configuration | Correct / 1319 | Invalid final answers | Truncated |
| --- | ---: | ---: | ---: |
| Official SGLang | 1284 | 0 | 0 |
| Previous Mini baseline | 1281 | 2 | 2 |
| Mini with public safe-gate decode | 1284 | 2 | 1 |

All three use identical frozen input IDs and gold answers, greedy generation, B8, capacity/graphs48, max1024 generated tokens, context4096 and no prefix cache. Precision is BF16 model/SSM/conv, MXFP8 MoE activation, FP8 E4M3 KV, TP8/DCP8/EP1, page64. The unmodified scorer was recomputed for all3957 response records. This is one offline quality cohort, not the million-token online workload.

Against the reference,1280 cases are correct on both sides,31 wrong on both,4 correct only in Mini and4 only in SGLang. Mini-only correct IDs:12,187,796,1038; SGLang-only:37,234,552,754. Token vectors match for678 cases; this is diagnostic and is not an acceptance requirement. All15 answer-discordant cases across the two comparisons were reviewed.

Negative evidence remains: case93 stops normally with a fraction400/11 rejected by the frozen numeric scorer (the reference answer36.36 also misses gold36). Case754 repeatedly derives89, continues reconsidering and reaches the1024-token cap without a valid final answer; the reference completes correctly. Cases234 and552 also regress against the reference. No manual regrading or length-cap increase was used.

GPU32 separately passed all440 actual KDA adapter/state cases on8 ranks, including graph replay, changed inputs, null indices, nontrivial storage pitches and inactive-state preservation. It compares the same mature SDK entry and therefore establishes adapter/ABI/state behavior, not independent full-model arithmetic.

The official entry is accepted into the current Mini baseline because its component contract is verified and this complete cohort improves the previous baseline while matching the official aggregate. This does not claim general model-quality equivalence or eliminate every full-model implementation risk. Full intermediate tensor tracing has not been performed. Reduction and rounding differences alone, including the documented RMSNorm denominator boundary, are not classified as algorithm errors.

## Serving performance remains separate

The controlled replacement-machine canonical AIPerf baseline remains364.197 output tokens/s for SGLang versus220.632 for the previous Mini runtime (60.58%). This safe-decode revision has not yet run the canonical900-second serving test. Offline evaluation seconds must not be used as serving throughput or latency.

Native AR+RMSNorm and MoE runtime preparation are separate isolated candidates. Their CPU/component results do not qualify whole-model quality or serving performance. Checkpoint tracking has a scoped8-rank component pass; automatic cache scheduling remains disabled pending integration.

See [machine-readable result and evidence hashes](kimi-k3-safe-decode-quality.json). Full responses, scorer outputs, semantic review, source pins and component artifacts are preserved remotely under /home/axis/mengze/mini-sglang/experiment/recovery-20261110.
