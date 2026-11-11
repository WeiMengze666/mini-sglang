# Generic sigmoid gating: CPU candidate

This isolated topic reuses the mature SDK gating kernel through a generic activation adapter. All 140 runtime Python files match the frozen V2 candidate. CPU: 168 passed, one existing opt-in multiprocess skip.

GPU correctness, full-model quality and serving performance remain pending. No default promotion or measured performance improvement is claimed.

# Generic sigmoid gating V2 candidate

This is a separate copy of the frozen V1 candidate. The two runtime files still add the generic sigmoid_and_mul activation adapter and call it from MLA. V2 adds only a host numel<2**32 guard because the mature SDK wrapper casts its flattened count to uint32. Public covered already validates positive/divisible-by8 contiguous same-shape BF16 CUDA tensors. The largest legal divisible count is2**32−8; its vector count plus255 and vector thread index fit uint32, so no stricter grid bound is needed.

The SDK explicitly rounds FP32 sigmoid to BF16 before multiplying and rounding the final product; the original Torch fallback preserves that double-rounding contract. Device, no-grad and16B pointer guards remain intact. Missing SDK hierarchy falls back; broken internal dependencies and actual execution failures propagate. No SDK or numerical kernel has been changed.

168 CPU checks passed, with one existing opt-in multiprocess skip. All28 original new contracts remain; four FakeMeta boundary cases are added. The actual old V1 source fails the three overflow cases and passes the largest safe boundary, while V2 passes all four without huge allocation or CUDA. Metadata-only execution of the actual public covered/helper ASTs and uint32 grid arithmetic is retained.

No GPU component, full-model quality, serving performance or default promotion is accepted. All V1 source/pins/manifest remain unchanged. A future small GPU component must bind this V2 source explicitly.
