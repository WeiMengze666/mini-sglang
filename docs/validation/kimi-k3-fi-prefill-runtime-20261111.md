# Tested FI-prefill runtime topic (2026-11-11)

This isolated topic preserves the exact **142 Python / 156 runtime files** used in
GPUq46. It integrates native FI KDA prefill through the existing attention backend
and generic MoE startup preparation. It contains no norm, GEMM-AG, QKVG,
large-prefill preparation or automatic-checkpoint changes. Select the existing
--attention-backend fi,sgl option. The default branch is not promoted.

The complete 1319-case cohort scored **1275/1319**, compared with **1284/1319**
for the pinned official reference: **−0.682335 percentage points**, with **zero
invalid and zero truncated candidate answers**. The user's current priority
permits this small task-quality deviation for exploratory performance measurement.
This is not acceptance of all tasks, whole-path precision, or identical outputs.
The 638.613-second offline evaluation is not a serving throughput measurement;
four of seven MoE tuning buckets differ from the prior preparation cohort.

The new canonical fresh reference→Mini pair is **GPUq64, queued at this snapshot**.
Reference and 900-second AIPerf arguments are unchanged; the only Mini CLI change
is auto→fi,sgl. No fresh performance conclusion is available. The retained
[canonical safe baseline](canonical-safe-baseline-20261111.md) remains
**216.377196 output tokens/s Mini versus 364.196617 SGLang**. Nsight58 is separate
diagnostics and does not replace that benchmark.

Publication checks compare every runtime byte to source46, the four necessary
test files, Python syntax, JSON and whitespace. No new full suite was executed.
Earlier EngineV2 CPU evidence (145 passed, 9 skipped), component39 native adapter
evidence (576 prefill fixtures and 440 safe-decode cases), and the actual full
GPUq46 evaluation retain their original limits. The
[machine-readable record](kimi-k3-fi-prefill-runtime-20261111.json) binds exact
source, evidence and submission hashes.
