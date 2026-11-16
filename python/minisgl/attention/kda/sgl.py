"""Adapt the mature SGLang KDA kernels to Mini's functional K,V state contract."""

from functools import cache

import torch


@cache
def _kernels():
    try:
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_kda_packed_decode,
        )
        from sglang.kernels.ops.attention.fla.kda import chunk_kda
        from tvm_ffi import use_torch_stream
    except ImportError as exc:
        raise RuntimeError("The sgl attention backend requires the SGLang kernel SDK") from exc
    return chunk_kda, fused_recurrent_kda_packed_decode, use_torch_stream


@cache
def _safe_decode_kernel():
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    return fused_sigmoid_gating_delta_rule_update


def _safe_decode(mixed, forget, beta, a_log, dt_bias, state, indices, cu_seqlens, lower_bound):
    """Pass raw gates and indexed V,K state to the official safe-gate operator."""
    heads, values, keys = state.shape[-3:]
    q, k, v = mixed.split((heads * keys, heads * keys, heads * values), dim=-1)
    return _safe_decode_kernel()(
        q=q.unflatten(-1, (heads, keys)).unsqueeze(0),
        k=k.unflatten(-1, (heads, keys)).unsqueeze(0),
        v=v.unflatten(-1, (heads, values)).unsqueeze(0),
        a=forget,
        b=beta.unsqueeze(0),
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state_source=state,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        lower_bound=lower_bound,
    )


def recurrent_kda(q, k, v, forget, beta, a_log, dt_bias, state, lower_bound, *, is_decode=False):
    tokens, heads, keys = q.shape
    if is_decode and tokens != 1:
        raise ValueError("KDA decode requires one token per request")
    values = v.shape[-1]
    # The SDK updates a V,K pool in place. Own the temporary storage so its
    # updates cannot mutate a caller's K,V checkpoint, including graph inputs.
    pool = (
        state.transpose(-1, -2).clone(memory_format=torch.contiguous_format).unsqueeze(0)
        if state is not None
        else torch.zeros((1, heads, values, keys), device=q.device, dtype=torch.float32)
    )
    if not tokens:
        return v.clone(), pool[0].transpose(-1, -2).contiguous()
    chunk, decode, stream = _kernels()
    indices = torch.zeros(1, device=q.device, dtype=torch.int32)
    with stream():
        if is_decode and lower_bound is not None:
            output = _safe_decode(
                torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1),
                forget.reshape(1, -1).contiguous(),
                beta.reshape(1, heads).contiguous(),
                a_log.reshape(-1).contiguous(),
                dt_bias.reshape(-1).contiguous(),
                pool,
                indices,
                torch.arange(2, device=q.device, dtype=torch.int32),
                lower_bound,
            )
        elif is_decode:
            output = v.new_empty((1, 1, heads, values))
            decode(
                mixed_qkv=torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1),
                a=forget.reshape(1, -1).contiguous(),
                b=beta.reshape(1, heads).contiguous(),
                A_log=a_log.reshape(-1).contiguous(),
                dt_bias=dt_bias.reshape(-1).contiguous(),
                scale=keys**-0.5,
                initial_state=pool,
                out=output,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                lower_bound=lower_bound,
            )
        else:
            count = (tokens + 63) // 64
            chunks = torch.stack(
                (
                    torch.zeros(count, device=q.device, dtype=torch.int32),
                    torch.arange(count, device=q.device, dtype=torch.int32),
                ),
                dim=1,
            )
            output = chunk(
                q=q.contiguous().unsqueeze(0),
                k=k.contiguous().unsqueeze(0),
                # The SDK reuses V as output; retain Mini's input ownership.
                v=v.contiguous().clone().unsqueeze(0),
                g=forget.contiguous().unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=pool,
                initial_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=torch.tensor([0, tokens], device=q.device, dtype=torch.int32),
                A_log=a_log.reshape(-1),
                dt_bias=dt_bias.reshape(-1),
                lower_bound=lower_bound,
                beta_is_raw=True,
                chunk_indices=chunks,
            )
    return output[0], pool[0].transpose(-1, -2).contiguous()


@cache
def _projection_kernels():
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
    from tvm_ffi import use_torch_stream

    return kimi_k3_tiny_gemm, use_torch_stream


def _projection_plan(layer):
    from minisgl.layers.linear import MergedLinear

    return layer.prepare_kernel(
        "sgl_projection",
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
    from sglang.srt.utils.common import is_blackwell_supported

    # Same hardware predicate as the official K3 bfa policy; no Engine globals.
    return 128 if is_blackwell_supported() else 64


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


@cache
def _decode_kernels():
    from sglang.kernels.ops.attention import kda_fused_decode
    from tvm_ffi import use_torch_stream

    return kda_fused_decode, use_torch_stream


def _decode_plan(layer):
    weights = [op.weight for op in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
    if any(weight.dtype != torch.float32 for weight in weights):
        return None
    return (
        *(weight.flatten(1).t().contiguous() for weight in weights),
        weights[0].new_zeros(3 * layer._heads * layer._dim),
        layer.A_log.reshape(-1).contiguous(),
        layer.o_norm.weight.float().contiguous(),
    )


def decode(layer, x, state):
    """Adapt owned K,V checkpoints to the native fused convolution/KDA/norm ABI."""
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not layer._full_gate
        or layer._dim != 128
        or layer._config.conv_kernel_size != 4
        or (state is not None and state.recurrent.dtype != torch.float32)
    ):
        return None
    static = layer.prepare_kernel("sgl_decode", lambda: _decode_plan(layer))
    if static is None:
        return None
    batch, heads, dim = x.shape[0], layer._heads, layer._dim
    if state is None:
        conv = x.new_zeros((batch, 3, 3 * heads * dim))
        pool = torch.zeros((batch, heads, dim, dim), device=x.device, dtype=torch.float32)
    else:
        conv = (
            state.conv.permute(0, 3, 1, 2).flatten(2).clone(memory_format=torch.contiguous_format)
        )
        pool = state.recurrent.transpose(-1, -2).clone(memory_format=torch.contiguous_format)
    projected, gate = _project_with_gate(layer, x)
    q, k, v, forget, beta = projected
    mixed = torch.cat((q, k, v), dim=-1)
    gate = gate if gate is not None else layer.g_proj.forward(x)
    indices = torch.arange(batch, device=x.device, dtype=torch.int32)
    native, stream = _decode_kernels()
    forget = forget.flatten(1)
    if not native.covered(mixed, forget, beta, conv, pool, indices, gate):
        return None
    wq, wk, wv, bias, a_log, norm = static
    with stream():
        output = native.kda_fused_decode(
            mixed,
            forget,
            beta,
            conv,
            wq,
            wk,
            wv,
            bias,
            a_log,
            layer.dt_bias,
            gate,
            norm,
            pool,
            indices,
            scale=dim**-0.5,
            onorm_eps=layer.o_norm.eps,
            lower_bound=layer._config.gate_lower_bound,
        )
    from .reference import KDAState
    from .registry import KDAResult

    history = conv.reshape(batch, 3, 3, heads * dim).permute(0, 2, 3, 1).contiguous()
    recurrent = pool.transpose(-1, -2).contiguous()
    return KDAResult(output[0], KDAState(history, recurrent), normed=True)


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


def indexed_decode(layer, x, pool, layer_id, indices, metadata):
    """Update BF16 working rows with official convolution and KDA kernels.

    FP32 state retains the existing full-fusion path. Immutable checkpoints are
    separate allocations; only the active indices may mutate these pool views.
    """
    from .causal_conv import _kernels as conv_kernels
    from .registry import KDAResult

    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or pool.state_dtype != torch.bfloat16
        or pool.conv_dtype != torch.bfloat16
        or layer._dim != 128
    ):
        return None
    bound = layer._config.gate_lower_bound
    if bound is not None:
        # Resolve the optional operator before convolution mutates working rows.
        _safe_decode_kernel()
    conv, recurrent = pool.indexed_kda_state(layer_id)
    weights = layer.prepare_kernel(
        "sgl_indexed_conv",
        lambda: torch.cat(
            [op.weight.squeeze(1) for op in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
        ),
    )
    projected, gate = _project_with_gate(
        layer, x, projection_stream=getattr(metadata, "projection_stream", None)
    )
    q, k, v, forget, beta = projected
    _, update_conv = conv_kernels()
    _, update_kda, stream = _kernels()
    with stream():
        mixed = update_conv(
            _conv_qkv_input(q, k, v),
            conv,
            weights,
            conv_state_indices=indices,
            activation="silu",
        )
        if bound is not None:
            output = _safe_decode(
                mixed,
                forget.reshape(len(x), -1),
                beta.reshape(len(x), layer._heads),
                layer.A_log,
                layer.dt_bias,
                recurrent,
                indices,
                metadata.cu_seqlens_q,
                bound,
            )[0]
        else:
            packed = x.new_empty((len(x), 1, layer._heads, layer._dim))
            update_kda(
                mixed_qkv=mixed,
                a=forget.reshape(len(x), -1),
                b=beta.reshape(len(x), layer._heads),
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer._dim**-0.5,
                initial_state=recurrent,
                out=packed,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
            )
            output = packed[:, 0]
    return KDAResult(output=output, gate=gate)


def indexed_prefill(layer, x, pool, layer_id, indices, metadata):
    """Use official varlen convolution and chunk extend on owned native rows."""
    from .prefill import prepare, supports

    if not supports(layer, x, pool):
        return None
    chunk, _, stream = _kernels()
    # Plan chunk metadata before convolution mutates its pool.
    chunks = metadata.kernel_data.get("sgl_prefill_chunks")
    if chunks is None:
        chunks = torch.tensor(
            [
                (seq, chunk)
                for seq, length in enumerate(metadata.lengths)
                for chunk in range((length + 63) // 64)
            ],
            device=x.device,
            dtype=torch.int32,
        )
        metadata.kernel_data["sgl_prefill_chunks"] = chunks
    prepared, gate = prepare(layer, x, pool, layer_id, indices, metadata)
    q, k, v, g, beta, recurrent, plan, indices = prepared
    with stream():
        output = chunk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=plan["cu_seqlens"],
            A_log=layer.A_log.reshape(-1),
            dt_bias=layer.dt_bias.reshape(-1),
            lower_bound=layer._config.gate_lower_bound,
            beta_is_raw=True,
            chunk_indices=chunks,
        )
    from .registry import KDAResult

    return KDAResult(output=output[0], gate=gate)
