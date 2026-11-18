"""Shape/ownership adapters for the normalization kernels used by SGLang."""


def rms_norm(x, weight, eps, *, gate=None, round_before_weight=False, dtype=None):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if not len(flat):
        return x.clone().to(dtype or x.dtype)
    from tvm_ffi import use_torch_stream

    with use_torch_stream():
        if gate is not None:
            from sglang.kernels.ops.attention.fla.fused_norm_gate import layer_norm_gated_fwd

            # Ask the native kernel for separate output storage; preserve the input.
            result, _, _, _ = layer_norm_gated_fwd(
                x=flat.contiguous(),
                g=gate.reshape_as(flat).contiguous(),
                weight=weight,
                bias=None,
                activation="sigmoid",
                eps=eps,
                out_dtype=flat.dtype,
                is_rms_norm=True,
            )
        elif round_before_weight:
            from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf

            result = rmsnorm_hf(flat.contiguous(), weight, eps)
        else:
            from sglang.kernels.ops.layernorm import rmsnorm

            result = rmsnorm(flat, weight, eps)
    return result.reshape(shape).to(dtype or x.dtype)
