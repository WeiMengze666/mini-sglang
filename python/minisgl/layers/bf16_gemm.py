"""Optional baa9bef SDK dense GEMM policy, without SGLang runtime config.

The SDK's initialize_bf16_gemm_config reads get_exec and mutates dispatch
state; this adapter never calls it. Its policy owns only AUTO eligibility,
Split-K's SDK environment switch, and references to the SDK's pure selectors
and original tactic table. Imports happen only after a real CUDA input exists.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import torch


@dataclass(frozen=True)
class Policy:
    sdk: object
    tgv: object
    splitk_enabled: bool
    batch_invariant: object

    def select(self, m, n, k, *, output_dtype, bias, caller_out):
        # unquant.apply / apply_into preserve the latter's bias-free Split-K
        # gate. Retained FP32 outputs follow _k3_bf16_gemm's TGV/torch route.
        if output_dtype not in (torch.bfloat16, torch.float32):
            return None
        if output_dtype == torch.float32 and bias:
            return None
        if (
            output_dtype == torch.bfloat16
            and self.splitk_enabled
            and (not caller_out or not bias)
            and self.sdk.use_bf16_splitk_gemm(m, n, k)
        ):
            return "splitk"
        return "tgv" if self.tgv.use_cutedsl_bf16_gemm(m, n, k) else None


@cache
def _load_policy():
    try:
        context = importlib.import_module("sglang.srt.runtime_context")
        # SDK AUTO hardware fact works before execution-config publication.
        if not context.get_platform().is_sm100:
            return None
        sdk = importlib.import_module("sglang.srt.layers.quantization.unquant")
        tgv = importlib.import_module("sglang.kernels.ops.gemm.cutedsl_bf16_gemm")
        invariant = importlib.import_module("sglang.srt.batch_invariant_ops")
    except (ImportError, OSError):
        return None
    required = (
        getattr(sdk, "use_bf16_splitk_gemm", None),
        getattr(sdk, "should_enable_bf16_splitk_gemm", None),
        getattr(tgv, "use_cutedsl_bf16_gemm", None),
        getattr(tgv, "cutedsl_bf16_gemm", None),
        getattr(tgv, "cutedsl_bf16_gemm_out", None),
        getattr(invariant, "is_batch_invariant_mode_enabled", None),
    )
    backend = getattr(getattr(sdk, "Bf16GemmBackend", None), "CUTEDSL", None)
    if (
        backend is None
        or not isinstance(getattr(sdk, "_BF16_SPLITK_TUNED_TACTICS", None), Mapping)
        or not all(callable(fn) for fn in required)
    ):
        return None
    return Policy(
        sdk,
        tgv,
        bool(sdk.should_enable_bf16_splitk_gemm(backend)),
        invariant.is_batch_invariant_mode_enabled,
    )


@cache
def _load_splitk():
    try:
        direct = importlib.import_module("flashinfer.gemm.kernels.dense_bf16_gemm_direct")
        split = importlib.import_module("flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk")
    except (ImportError, OSError):
        return None
    required = (
        getattr(direct, "prefer_direct_bf16_gemm_sm100", None),
        getattr(direct, "default_tactic", None),
        getattr(direct, "run_direct_dense", None),
        getattr(split, "SplitKTactic", None),
        getattr(split, "run_splitk_dense", None),
    )
    return (direct, split) if all(callable(fn) for fn in required) else None


def _allocate(x, m, n, dtype):
    return torch.empty((m, n), dtype=dtype, device=x.device)


def try_bf16_gemm(x, weight, bias=None, *, out=None, output_dtype=None):
    """Return native output, or decline before launching; never retry a kernel.

    Normal graph warmup must cover each physical shape/tactic. Unsupported
    layouts and batch-invariant SDK execution use the original Torch overload.
    """
    if not x.is_cuda:
        return None
    if (
        x.ndim < 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or x.device != weight.device
        or x.shape[-1] <= 0
        or weight.shape[0] <= 0
        or x.shape[-1] != weight.shape[1]
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
        or (x.ndim > 2 and not x.is_contiguous())
        or x.requires_grad
        or weight.requires_grad
        or x.data_ptr() % 32
        or weight.data_ptr() % 32
    ):
        return None
    m, n, k = x.numel() // x.shape[-1], weight.shape[0], weight.shape[1]
    dtype = out.dtype if out is not None else output_dtype or x.dtype
    if out is not None and output_dtype is not None and out.dtype != output_dtype:
        return None
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != x.dtype
        or bias.device != x.device
        or not bias.is_contiguous()
        or bias.requires_grad
    ):
        return None
    if out is not None and (
        out.shape != (m, n)
        or out.device != x.device
        or not out.is_contiguous()
        or out.data_ptr() % 32
        or out.requires_grad
        or any(torch._C._overlaps(out, value) for value in (x, weight, bias) if value is not None)
    ):
        return None
    if not _load_policy.cache_info().currsize and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm native BF16 GEMM before graph capture")
    policy = _load_policy()
    if policy is None or policy.batch_invariant():
        return None
    route = policy.select(
        m, n, k, output_dtype=dtype, bias=bias is not None, caller_out=out is not None
    )
    if route is None:
        return None
    flat = x if x.ndim == 2 else x.view(m, k)
    if route == "tgv":
        if out is None and dtype == x.dtype:
            return policy.tgv.cutedsl_bf16_gemm(flat, weight, bias).view(*x.shape[:-1], n)
        output = out if out is not None else _allocate(x, m, n, dtype)
        policy.tgv.cutedsl_bf16_gemm_out(flat, weight, output, bias)
    else:
        # FlashInfer's mature runners require dense row/column-major layouts.
        if not flat.is_contiguous() or not weight.is_contiguous():
            return None
        if not _load_splitk.cache_info().currsize and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm native Split-K GEMM before graph capture")
        runners = _load_splitk()
        if runners is None:
            return None
        direct, split = runners
        output = out if out is not None else _allocate(x, m, n, dtype)
        if bias is None and direct.prefer_direct_bf16_gemm_sm100(m, n, k):
            tactic = direct.default_tactic(m, n, k)
            direct.run_direct_dense(flat, weight.T, output, True, tactic)
        else:
            # Reference the SDK table object; no local tactic/launch table.
            tactic = split.SplitKTactic(*policy.sdk._BF16_SPLITK_TUNED_TACTICS[(m, n, k)])
            split.run_splitk_dense(flat, weight.T, bias, output, True, tactic)
    return output if out is not None else output.view(*x.shape[:-1], n)
