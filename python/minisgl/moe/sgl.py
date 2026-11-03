"""Routing adapter for the standalone kernels used by SGLang's MoE backends."""


def grouped_topk(logits, *, bias, topk, num_groups, topk_groups, scoring_func, renormalize, scale):
    if not logits.is_cuda or topk > 32 or (bias is None and scoring_func != "softmax"):
        return None
    if scoring_func not in ("sigmoid", "softmax") or (
        bias is not None and scoring_func != "sigmoid"
    ):
        return None
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
    from tvm_ffi import use_torch_stream

    with use_torch_stream():
        return moe_fused_gate(
            logits,
            bias,
            topk=topk,
            scoring_func=scoring_func,
            num_expert_group=num_groups,
            topk_group=topk_groups,
            renormalize=renormalize,
            routed_scaling_factor=scale,
            apply_routed_scaling_factor_on_output=scale != 1.0,
        )
