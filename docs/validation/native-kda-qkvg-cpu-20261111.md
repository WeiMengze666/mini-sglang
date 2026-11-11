# Merged KDA projections: CPU candidate

This topic is isolated from the default runtime. It reuses mature numerical kernels through generic layer and attention adapters.

CPU validation: {'passed': 860, 'conditional_skipped': 43, 'new_cases': 30}. All 140 runtime Python files in this publication match the frozen candidate. No new tests were run on the local machine.

GPU correctness, full-model quality and serving performance are pending. This is not a measured performance improvement and is not an integration of other optimization candidates.

The frozen implementation scope follows. Statements about candidate isolation describe its preparation, before this topic publication.

# Generic merged KDA projection candidate

Isolated source: native-kda-qkvg-candidate, base290ad026. Five runtime files only. No EngineV2/MoE norm/GEMM-AG/FI indexed-prefill/automatic-checkpoint union, no runtime flag, no numerical kernel.

MergedLinear.forward now accepts explicit output_dtype; default float32 keeps the original branches including mixed input/weight dtypes. Input-dtype output delegates to the existing dense linear kernel. KDALayer.after_load clears old kernel/weight plans and uses MergedLinear.try_create for supported dense full-rank [q,k,v,g]. Original public parameters become views into one canonical allocation. Unsupported quantization/missing weight/layout/bias/nonfull gate keeps the original independent linears. Neither model name nor shape constants control fusion.

KDAProjection explicitly carries five recurrence inputs plus optional output gate. SG tiny [f_a|beta]/f_b GEMMs stay unchanged. Functional, batched, per-request and indexed decode carry the gate to finish; native fully normalized outputs retain their existing normed contract. No per-forward gate Tensor is stored on the layer. QKV/gate split rows can be strided; no contiguous assumption or fallback to a reference kernel was added.

Weight replacement uses the existing load/after_load contract. CPU tests prove complete reload and partial replacement followed by after_load release the prior plan and rebind all current weight views. Reloading live CUDA graphs is not claimed; destroy/rebuild captured graphs under existing model-reload ownership.

Final CPU:860passed/43skipped, 903collected,33.68s. The 43skips are retained conditional hardware/SDK cases, not GPU qualification. Includes30new meaningful CPU cases: independent dyadic FP64 dot reference, M1/8/12/48/65 BF16/F32 and noncontiguous input/output views, canonical storage, reload/partial replacement, quantized/missing/layout/bias/nonfull fallbacks, explicit per-forward gate, functional torch/FI/SG and real backend chunk/allT1/decode state, default mixed-dtype FP32 output. Ruff/diffcheck pass. Earlier148focused/856core results and initial lint log retained.

No GPU execution/submission, no performance/quality or native graph qualification. Component plan is a proposed next gate; fullmodel quality and paired canonical serving are later gates. Base290 does not include FI indexed-prefill facade; explicit future integration is required, not inferred from a same-name callback.
