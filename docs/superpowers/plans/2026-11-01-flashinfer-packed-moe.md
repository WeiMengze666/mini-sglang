# FlashInfer packed MXFP4 backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with checkpoints.

**Goal:** Add an opt-in generic FlashInfer FP4 grouped-MoE backend with explicit MXFP8 activation policy, while preserving the current BF16 Triton default.

**Architecture:** Extend the existing backend policy from EngineConfig through MXFP4Linear into LatentMoE.after_load. Keep MXFP4Experts unchanged and add a separate FlashInfer executor that prepares transformed weights once and implements the same forward contract.

**Tech Stack:** Python, PyTorch, FlashInfer TRTLLM FP4, Triton, pytest, torch.distributed/gpuq.

**Spec:** docs/superpowers/specs/2026-11-01-flashinfer-packed-moe-design.md

## Global Constraints

- Remote source and CPU work stay in `/home/axis/mengze/mini-sglang-k3/mini-sglang`; GPU work runs only through `gpuq`.
- `mxfp4_backend=triton` and current BF16 arithmetic remain the default.
- Kimi model names do not enter generic quantization, MoE, attention, cache, tokenizer, or scheduler modules.
- No tolerance relaxation is allowed for the shipping strict BF16 gate.

---

### Task 1: Policy seam and CPU contracts

**Files:** Modify `python/minisgl/engine/config.py`, `python/minisgl/engine/engine.py`, `python/minisgl/quantization/mxfp4.py`, `python/minisgl/layers/latent_moe.py`; test `tests/core/test_mxfp4_policy.py`.

**Interfaces:** Add `EngineConfig.mxfp4_activation_precision='bf16'`; extend `MXFP4Linear.set_backend(backend, activation_precision='bf16')`; select the executor from the generic policy in `LatentMoE.after_load`.

- [ ] Write CPU tests for valid policies and invalid combinations.
- [ ] Implement policy validation and pass the policy through model construction.
- [ ] Run `CUDA_VISIBLE_DEVICES= PYTHONPATH=python pytest -o addopts='' tests/core/test_mxfp4_policy.py`.
- [ ] Commit the policy seam.

### Task 2: FlashInfer weight preparation

**Files:** Create `python/minisgl/quantization/mxfp4_flashinfer.py`; test `tests/models/test_mxfp4_flashinfer.py`.

**Interfaces:** Create `FlashInferMXFP4Experts(experts, activation, beta, linear_beta, num_global_experts, activation_precision)` with `forward(x, weights, ids) -> FP32 [tokens, routed_width]`.

- [ ] Add CPU import and policy tests without initializing CUDA.
- [ ] Implement one-time stacking, E8M0 decode, official byte/scale shuffle, and contiguous storage.
- [ ] Implement local/global expert ID metadata without model-specific names.
- [ ] Run CPU tests and static compile checks, then commit.

### Task 3: FlashInfer forward and activation policy

**Files:** Modify `python/minisgl/quantization/mxfp4_flashinfer.py` and `python/minisgl/layers/latent_moe.py`; test `tests/models/test_mxfp4_flashinfer.py`.

- [ ] Add a GPU test for BF16 SiLU and a GPU test for MXFP8 SiTU with beta=4 and linear_beta=25.
- [ ] Implement the routed FlashInfer call, explicit activation enum, MXFP8 quantization, and output conversion.
- [ ] Verify remote routes contribute zero and duplicate rows remain identical.
- [ ] Run focused GPU tests through gpuq and commit only after the gate passes.

### Task 4: Real Kimi integration and strict/performance gates

**Files:** Modify `docs/kimi-k3.md`; add a reusable validation gate only if needed.

- [ ] Run the real Kimi layer-12 expert probe with the new executor.
- [ ] Run current shipping Triton strict DCP1/EP8 and DCP2/EP8 gates to prove no default regression.
- [ ] Run FlashInfer MXFP8 DCP2/EP8 performance at batch 1/2/4/8 with CUDA Graphs.
- [ ] Record strict correctness status, precision mode, and performance ratios separately.
- [ ] Keep FlashInfer opt-in unless both policy and correctness criteria pass.
- [ ] Synchronize remote commits to local before pushing GitHub.
