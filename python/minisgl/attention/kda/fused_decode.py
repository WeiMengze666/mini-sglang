"""CUDA convolution, delta recurrence and gated output normalization."""

from functools import cache

import torch

from .projection import _project_with_gate


@cache
def _decode_kernels():
    from minisgl.kernel.kda import fused_decode as kda_fused_decode
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
    static = layer.prepare_kernel("fused_decode", lambda: _decode_plan(layer))
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
