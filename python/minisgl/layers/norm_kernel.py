"""Shape/ownership adapters for the normalization kernels used by SGLang."""


def rms_norm(x, weight, eps, *, gate=None, round_before_weight=False, dtype=None):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if not len(flat):
        return x.clone().to(dtype or x.dtype)
    from tvm_ffi import use_torch_stream

    with use_torch_stream():
        if gate is not None:
            from minisgl.kernel.kda.kda import rms_norm_gated

            # The SDK may consume its input as output. Preserve caller ownership.
            result = rms_norm_gated(
                flat.clone(), gate.reshape_as(flat), weight, None, eps=eps, activation="sigmoid"
            )
        elif round_before_weight:
            from minisgl.kernel.norm_hf import rmsnorm_hf

            result = rmsnorm_hf(flat.contiguous(), weight, eps)
        else:
            from flashinfer.norm import rmsnorm

            # Native RMSNorm supports row strides but requires contiguous features.
            if flat.stride(-1) != 1:
                flat = flat.contiguous()
            result = rmsnorm(flat, weight, eps)
    return result.reshape(shape).to(dtype or x.dtype)
