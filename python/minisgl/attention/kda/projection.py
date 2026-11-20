"""Shared projection and packed QKV views for native KDA kernels."""

from functools import cache

import torch


@cache
def _projection_kernels():
    from minisgl.kernel.gemm.tiny import skinny_gemm
    from tvm_ffi import use_torch_stream

    return skinny_gemm, use_torch_stream


def _projection_plan(layer):
    from minisgl.layers.linear import MergedLinear

    return layer.prepare_kernel(
        "gate_projection",
        lambda: MergedLinear.try_create(
            {"forget": layer.f_a_proj, "beta": layer.b_proj}, pad_rows_to=8
        ),
    )


def _gate_projection_plan(layer, x):
    """Prepare immutable skinny weights before either projection branch launches."""
    weight = getattr(layer.f_b_proj, "weight", None)
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not layer._full_gate
        or not isinstance(weight, torch.Tensor)
        or weight.device != x.device
        or weight.dtype != x.dtype
        or not weight.is_contiguous()
    ):
        return None
    plan = _projection_plan(layer)
    return None if plan is None else (plan, weight)


def _project_forget_beta(layer, x):
    """Offer the SDK's skinny gate dispatch without projecting Q, K or V."""
    prepared = _gate_projection_plan(layer, x)
    if prepared is None:
        return None
    plan, weight = prepared
    gemm, stream = _projection_kernels()
    with stream():
        bfa = gemm(x, plan.weight)
        fa_width, beta_width = plan.widths
        forget = gemm(bfa[..., :fa_width], weight)
    return (
        forget.reshape(-1, layer._heads, layer._dim),
        bfa[..., fa_width : fa_width + beta_width],
    )


def project(layer, x):
    """Keep the public five-tensor projection ABI for prefill and providers."""
    gates = _project_forget_beta(layer, x)
    if gates is None:
        return layer.project(x)
    return (layer.q_proj.forward(x), layer.k_proj.forward(x), layer.v_proj.forward(x), *gates)


@cache
def _projection_overlap_limit(device):
    from minisgl.utils import is_sm100_supported

    # Same hardware predicate as the official K3 bfa policy; no Engine globals.
    return 128 if is_sm100_supported() else 64


def _project_with_gate(layer, x, *, projection_stream=None):
    """Reuse the wide QKVG projection while keeping its output gate explicit."""
    plan = getattr(layer, "_qkvg", None)
    if plan is None or x.dtype != torch.bfloat16 or x.device != plan.weight.device:
        return project(layer, x), None
    # Resolving the gate plan may materialize weights on the invoking stream.
    # Do that before the fork edge and before either projection GEMM launches.
    gate_plan = _gate_projection_plan(layer, x)
    limit = (
        _projection_overlap_limit(x.device)
        if projection_stream is not None and gate_plan is not None
        else 0
    )
    overlap = 0 < len(x) <= limit and torch.cuda.is_current_stream_capturing()
    if overlap:
        if projection_stream.device != x.device:
            raise ValueError("KDA projection stream differs from the local input device")
        main = torch.cuda.current_stream(device=x.device)
        projection_stream.wait_stream(main)
        x.record_stream(projection_stream)
        # Official CUDA capture order: main wide GEMM first, then side skinny GEMMs.
        merged = layer.project_qkvg(x)
        with torch.cuda.stream(projection_stream):
            gates = _project_forget_beta(layer, x)
        main.wait_stream(projection_stream)
        if gates is None:
            raise RuntimeError("Prepared KDA gate projection declined after selection")
        for value in gates:
            value.record_stream(main)
    else:
        merged = layer.project_qkvg(x)
        gates = _project_forget_beta(layer, x)
    if merged is None:
        raise RuntimeError("Prepared KDA QKVG projection declined after selection")
    if gates is None:
        gates = (
            layer.f_b_proj.forward(layer.f_a_proj.forward(x)).reshape(-1, layer._heads, layer._dim),
            layer.b_proj.forward(x),
        )
    return (merged["q"], merged["k"], merged["v"], *gates), merged["gate"]


def _conv_qkv_input(q, k, v):
    """Reuse adjacent projection slices without imposing a packed row stride.

    Native convolution receives every input stride. The combined view ends at
    the already-valid V slice, and PyTorch still checks its storage bounds.
    Independent projections or autograd retain the original concatenation.
    """
    parts = (q, k, v)
    if (
        q.ndim == 2
        and all(
            part.shape == q.shape
            and part.dtype == q.dtype
            and part.device == q.device
            and not part.requires_grad
            and not (part.is_conj() or part.is_neg())
            and part.stride() == q.stride()
            and part.stride(-1) == 1
            for part in parts
        )
        and all(torch._C._is_alias_of(q, part) for part in parts[1:])
        and k.storage_offset() == q.storage_offset() + q.shape[-1]
        and v.storage_offset() == q.storage_offset() + 2 * q.shape[-1]
    ):
        return q.as_strided((q.shape[0], 3 * q.shape[-1]), q.stride())
    return torch.cat(parts, dim=-1)
