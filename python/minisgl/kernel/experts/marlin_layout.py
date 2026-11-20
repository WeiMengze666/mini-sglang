from __future__ import annotations

from collections.abc import Callable

import torch

from .marlin_repack import gptq_marlin_repack


def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False)


def _get_optional_param(layer: torch.nn.Module, *names: str) -> torch.Tensor | None:
    for name in names:
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def _normalize_scale_tensor(scales: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    # The kernel consumes E8M0 exponents. Regardless of the placeholder dtype
    # the loader used, we want the *numerical* value 2**e in ``target_dtype``.
    # float32/bfloat16/float16 containers hold the numerical 2**e directly
    # (they were filled via a dtype-promoting copy from uint8/e8m0).
    # uint8/int8 containers hold the raw E8M0 byte and must be reinterpreted.
    if scales.dtype == torch.float8_e8m0fnu:
        return scales.to(target_dtype)
    if scales.dtype == torch.uint8:
        return scales.view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype == torch.int8:
        return scales.view(torch.uint8).view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return scales.to(target_dtype)
    raise TypeError(f"Unsupported MXFP4 scale dtype for Marlin: {scales.dtype}")


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(marlin_scales.size(0), -1)
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        # exponent_bias (fp4->fp8) = 2 ** 3 - 2 ** 1 = 6
        marlin_scales = marlin_scales + 6
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)
    return marlin_scales


def _repack_moe_fp4_weight_for_marlin(
    weight: torch.Tensor,
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    perm: torch.Tensor,
) -> torch.Tensor:
    assert weight.shape == (num_experts, size_n, size_k // 2)

    tensor_list = []
    for i in range(num_experts):
        qweight = weight[i].view(torch.int32).T.contiguous()
        marlin_qweight = gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
        )
        tensor_list.append(marlin_qweight)
    return torch.stack(tensor_list)


def _permute_moe_fp4_scales_for_marlin(
    scales: torch.Tensor,
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    group_size: int,
    process_scales: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    tensor_list = []
    for i in range(num_experts):
        scale = scales[i].T.contiguous()
        marlin_scales = marlin_permute_scales(
            s=scale,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
        )
        tensor_list.append(process_scales(marlin_scales))
    return torch.stack(tensor_list)


def prepare_moe_mxfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    group_size = 32
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w2_scale = _get_optional_param(layer, "w2_weight_scale", "w2_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")
    w2_bias = _get_optional_param(layer, "w2_weight_bias", "w2_bias")

    if w13_scale is None or w2_scale is None:
        raise ValueError("MXFP4 Marlin requires w13/w2 weight scales.")

    w13_scale_data = w13_scale.data if hasattr(w13_scale, "data") else w13_scale
    w2_scale_data = w2_scale.data if hasattr(w2_scale, "data") else w2_scale
    w13_bias_data = w13_bias.data if hasattr(w13_bias, "data") else w13_bias
    w2_bias_data = w2_bias.data if hasattr(w2_bias, "data") else w2_bias

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2
    if hidden_size % 128 == 0:
        padded_intermediate_size = ((intermediate_size + 63) // 64) * 64
    else:
        if hidden_size % 64 != 0:
            raise ValueError(
                f"MXFP4 Marlin requires hidden_size to be divisible by 64, got {hidden_size}."
            )
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
    param_dtype = getattr(
        layer,
        "orig_dtype",
        w13_bias_data.dtype if w13_bias_data is not None else torch.bfloat16,
    )

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _pad_w13(x: torch.Tensor) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        x = x.view(num_experts, 2, intermediate_size, x.shape[-1])
        x = torch.nn.functional.pad(x, (0, 0, 0, padded_intermediate_size - intermediate_size))
        return x.reshape(num_experts, 2 * padded_intermediate_size, -1)

    def _pad_w2(x: torch.Tensor, packing: int) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        return torch.nn.functional.pad(
            x, (0, (padded_intermediate_size - intermediate_size) // packing)
        )

    w13 = _pad_w13(w13)
    w2 = _pad_w2(w2, packing=2)
    w13_scale_data = _pad_w13(_normalize_scale_tensor(w13_scale_data, param_dtype))
    w2_scale_data = _pad_w2(
        _normalize_scale_tensor(w2_scale_data, param_dtype),
        packing=group_size,
    )
    if w13_bias_data is not None:
        w13_bias_data = _pad_w13(w13_bias_data.unsqueeze(-1)).squeeze(-1)

    w13_size_n, w13_size_k = padded_intermediate_size * 2, hidden_size
    w2_size_n, w2_size_k = hidden_size, padded_intermediate_size

    def _process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
        return mxfp4_marlin_process_scales(marlin_scales, input_dtype=param_dtype)

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        tensor_list = []
        for i in range(num_experts):
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))
        return torch.stack(tensor_list)

    w13_marlin = _repack_moe_fp4_weight_for_marlin(
        w13, num_experts=num_experts, size_n=w13_size_n, size_k=w13_size_k, perm=perm
    )
    w2_marlin = _repack_moe_fp4_weight_for_marlin(
        w2, num_experts=num_experts, size_n=w2_size_n, size_k=w2_size_k, perm=perm
    )
    w13_scale_marlin = _permute_moe_fp4_scales_for_marlin(
        w13_scale_data,
        num_experts=num_experts,
        size_n=w13_size_n,
        size_k=w13_size_k,
        group_size=group_size,
        process_scales=_process_scales,
    )
    w2_scale_marlin = _permute_moe_fp4_scales_for_marlin(
        w2_scale_data,
        num_experts=num_experts,
        size_n=w2_size_n,
        size_k=w2_size_k,
        group_size=group_size,
        process_scales=_process_scales,
    )

    layer.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    if w13_bias_data is not None:
        layer.w13_weight_bias = torch.nn.Parameter(
            _permute_bias(w13_bias_data), requires_grad=False
        )
    if w2_bias_data is not None:
        layer.w2_weight_bias = torch.nn.Parameter(_permute_bias(w2_bias_data), requires_grad=False)

    # Marlin uses the repacked scales; release the loader-format parameters.
    for stale in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        if hasattr(layer, stale):
            delattr(layer, stale)
