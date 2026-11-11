# Official KDA dispatch correction — 2026-11-11

The pinned official baseline used **FlashInfer for covered multi-token KDA prefill and Triton for KDA decode**. Its final startup log explicitly records `decode=triton, prefill=flashinfer, verify=triton` and `extend=FlashInferKDAPrefillKernel`. The earlier interpretation that `linear_attn_backend=triton` also identified prefill was incorrect.

At SGLang commit `baa9befa381511cd4d56f83980951e5a45fffc6d`, the attention factory supplies a FlashInfer prefill default for the supported safe-gate Blackwell configuration. The resolver applies an explicit prefill leaf first, that computed default second, and the base backend last. An unset prefill leaf is therefore not evidence of Triton prefill. The dispatcher retains its Triton fallback for a packed total of at most one token or unsupported safe-gate cases.

The previous comparison of SG and FI QK normalization formulas remains valid between those implementations. It cannot justify pausing FI integration as noncanonical: FI was the actual reference multi-token prefill path. The original audits and their negative results remain preserved; the accompanying correction JSON identifies the exact superseded interpretations.

The FI-prefill candidate's separate 1319-example cohort still scored 1275, versus 1284 for the retained reference. Its adverse cases and differing MoE tuning selections remain evidence requiring investigation. Correcting kernel attribution does not establish model-quality equivalence, a single cause, a performance improvement, or default promotion.

Both Mini `fi` and `sgl` factories construct the same MLA adapter, and the relevant MLA/factory source files match the safe baseline. Their names alone do not demonstrate an MLA numerical-kernel change. The adapter already calls the mature TRTLLM prefill operator and uses the CuTe decode portfolio under their supported conditions; whole-path tensor/layout/precision evidence must be checked separately.

No runtime code or measured baseline value changes in this correction. Actual GPUq42 log SHA256: `79d497a05d88a8818b9b4e9c159481d466ed394f756ad88c76eae5420309bb04`. Remote correction receipt: `canonical-kda-prefill-dispatch-correction/correction.json`, SHA256 `16b5c77c45a57eb9c23d2c93b4ca3c1fdf3fb145be7bde72023ef258b21f1937`.
