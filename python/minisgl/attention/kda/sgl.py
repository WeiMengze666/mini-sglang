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


def recurrent_kda(q, k, v, forget, beta, a_log, dt_bias, state, lower_bound, *, is_decode=False):
    """Use the native extend or decode portfolio according to the batch phase."""
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
        if is_decode:
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


def project(layer, x):
    """Use the SDK's skinny projection dispatch, retaining its aligned gate views."""
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
        return layer.project(x)
    plan = _projection_plan(layer)
    if plan is None:
        return layer.project(x)
    gemm, stream = _projection_kernels()
    with stream():
        bfa = gemm(x, plan.weight)
        fa_width, beta_width = plan.widths
        forget = gemm(bfa[..., :fa_width], weight)
    return (
        layer.q_proj.forward(x),
        layer.k_proj.forward(x),
        layer.v_proj.forward(x),
        forget.reshape(-1, layer._heads, layer._dim),
        bfa[..., fa_width : fa_width + beta_width],
    )


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
    q, k, v, forget, beta = project(layer, x)
    mixed = torch.cat((q, k, v), dim=-1)
    gate = layer.g_proj.forward(x)
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
    from .registry import KDADecodeResult

    history = conv.reshape(batch, 3, 3, heads * dim).permute(0, 2, 3, 1).contiguous()
    recurrent = pool.transpose(-1, -2).contiguous()
    return KDADecodeResult(output[0], KDAState(history, recurrent), normed=True)


def indexed_decode(layer, x, pool, layer_id, indices):
    """Update BF16 working rows with official convolution and KDA kernels.

    FP32 state retains the existing full-fusion path. Immutable checkpoints are
    separate allocations; only the active indices may mutate these pool views.
    """
    from .causal_conv import _kernels as conv_kernels
    from .registry import KDADecodeResult

    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or pool.state_dtype != torch.bfloat16
        or pool.conv_dtype != torch.bfloat16
        or layer._dim != 128
    ):
        return None
    conv, recurrent = pool.indexed_kda_state(layer_id)
    weights = layer.prepare_kernel(
        "sgl_indexed_conv",
        lambda: torch.cat(
            [op.weight.squeeze(1) for op in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
        ),
    )
    q, k, v, forget, beta = project(layer, x)
    _, update_conv = conv_kernels()
    _, update_kda, stream = _kernels()
    with stream():
        mixed = update_conv(
            torch.cat((q, k, v), -1),
            conv,
            weights,
            conv_state_indices=indices,
            activation="silu",
        )
        output = x.new_empty((len(x), 1, layer._heads, layer._dim))
        update_kda(
            mixed_qkv=mixed,
            a=forget.reshape(len(x), -1),
            b=beta.reshape(len(x), layer._heads),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer._dim**-0.5,
            initial_state=recurrent,
            out=output,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            lower_bound=layer._config.gate_lower_bound,
        )
    return KDADecodeResult(output=output[:, 0])


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
    q, k, v, g, beta, recurrent, plan, indices = prepare(
        layer, x, pool, layer_id, indices, metadata
    )
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
    return output[0]
