"""FlashInfer tactic selection for one fresh worker and its expert-TP group."""

import os
from contextlib import contextmanager
from typing import NamedTuple

import torch

from ..dispatch import DispatchLayout
from ..prepare import _agree


class ExpertPreparation(NamedTuple):
    """The existing metadata signature, with names for backend resource planning."""

    local_rows: tuple[int, ...]
    tp_size: int
    ep_size: int
    expert_start: int
    global_experts: int
    local_experts: int
    hidden: int
    intermediate: int
    activation_precision: str
    activation: str
    beta: float
    linear_beta: float | None
    input_dtype: str
    top_k: int
    num_groups: int
    topk_groups: int
    scoring_func: str
    renormalize: bool
    routed_scale: float
    logits_dtype: str
    bias_abi: tuple | None
    weights_abi: tuple
    dispatch_layouts: tuple[DispatchLayout, ...]

    @staticmethod
    def tuning_session(context, count):
        return tuning_session(context, count)


def _tensor(value, device=None):
    if isinstance(value, (tuple, list)):
        return tuple(_tensor(item, device) for item in value)
    if not isinstance(value, torch.Tensor):
        return value
    if device is not None and value.device != torch.device(device):
        raise ValueError("Expert preparation tensor device differs from context.device")
    return (tuple(value.shape), str(value.dtype), value.stride(), value.data_ptr() % 256)


def _signature(executor, routing, context):
    return ExpertPreparation(
        context.local_rows,
        context.layout.tensor.size,
        context.layout.ep_size,
        executor._start,
        executor._global_experts,
        executor._experts,
        executor._hidden,
        executor._intermediate,
        executor._precision,
        executor._activation,
        executor._beta,
        executor._linear_beta,
        str(executor.input_dtype),
        routing.topk,
        routing.num_groups,
        routing.topk_groups,
        routing.scoring_func,
        routing.renormalize,
        routing.scale,
        str(routing.logits.dtype),
        _tensor(routing.bias, context.device),
        tuple(
            _tensor(getattr(executor, name), context.device)
            for name in ("_w13", "_w13_scale", "_w2", "_w2_scale", "_alpha", "_linear")
        ),
        executor.dispatch_layouts(routing.topk),
    )


def _domain(context, version):
    properties = torch.cuda.get_device_properties(context.device)
    # Finite SM100 tactic capability inputs, not an experimental hardware report.
    return (
        properties.major,
        properties.minor,
        properties.multi_processor_count,
        version,
        torch.__version__,
        torch.version.cuda,
        tuple(
            sorted(
                (key, value) for key, value in os.environ.items() if key.startswith("FLASHINFER_")
            )
        ),
    )


def _calibrate_routes(executor, routes):
    # FI.7 profiles small buckets from these actual rows. Use its own sampler
    # rather than calibrating every token against the same few local experts.
    from flashinfer.fused_moe.shared.tuning import moe_topk_ids_init

    packed = isinstance(routes, torch.Tensor)
    target = routes if packed else routes[0]
    sampled = moe_topk_ids_init(executor._global_experts, packed=packed)(
        tuple(target.shape), target.dtype, target.device
    )
    if executor._experts < executor._global_experts:
        ids = ((sampled >> 16) & 0xFFFF) if packed else sampled
        # One global bijection retains distinct/diverse routes and guarantees
        # local work even for a single-token EP calibration, without resampling.
        ids = (ids + executor._start - ids[0, 0]).remainder(executor._global_experts)
        sampled = (ids << 16) | (sampled & 0xFFFF) if packed else ids
    target.copy_(sampled)
    if not packed:
        routes[1].fill_(1)


def prepare_local(backend, executor, routing_builder, context, *, dry_run=False):
    x = logits = routing = prepared = ids = routes = output = None
    try:
        owned = context.layout.expert_range(executor._global_experts)
        if (executor._start, executor._experts) != (owned.start, len(owned)):
            raise ValueError("Executor placement differs from the expert partition")
        # Routing metadata must not initialize or allocate on the device in preflight.
        logits = torch.empty((1, executor._global_experts), dtype=torch.float32, device="meta")
        routing = routing_builder(logits)
        signature = _signature(executor, routing, context)
        if dry_run:
            return signature
        topk = routing.topk
        for rows in context.local_rows:
            x = torch.zeros(
                (rows, executor._hidden), dtype=executor.input_dtype, device=context.device
            )
            columns = torch.arange(executor._global_experts, device=context.device)
            relative = (columns - executor._start) % executor._global_experts
            logits = torch.where(relative < topk, 4.0 - relative.float() * 0.01, -4.0)
            routing = routing_builder(logits.expand(rows, -1).contiguous())
            prepared = executor.prepare_inputs(backend, x, routing)
            routes = prepared[2]
            _calibrate_routes(executor, routes)
            ids = routes[0] if isinstance(routes, tuple) else ((routes >> 16) & 0xFFFF)
            if not bool(
                ((ids >= executor._start) & (ids < executor._start + executor._experts)).any()
            ):
                raise ValueError("Preparation routing produced no local expert work")
            # Actual packed/unpacked routes, quantized activations/scales and
            # all fixed SM100 runner inputs agree before its tactic reductions.
            _agree(
                (signature, executor._tune_tokens(rows), _tensor(prepared, context.device)), context
            )
            output = executor._run(*prepared)
            x = logits = routing = prepared = ids = routes = output = None
    finally:
        del x, logits, routing, prepared, ids, routes, output


def tuning_session(context, count):
    import flashinfer
    from flashinfer.autotuner import (
        AutoTuner,
        autotune,
        get_autotune_process_group,
        set_autotune_process_group,
    )

    tuner = AutoTuner.get()
    previous_group = get_autotune_process_group()
    if (
        tuner.is_tuning_mode
        or previous_group is not None
        or os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0") == "1"
    ):
        raise ValueError("External FI owner/cache is incompatible with fresh preparation")
    _agree(_domain(context, flashinfer.__version__), context)
    tuner.clear_cache()
    _agree(("owned-empty-start", count), context)

    @contextmanager
    def session():
        try:
            set_autotune_process_group(context.cpu_group)
            with autotune(True, cache=None, skip_ops=set()):
                yield
        finally:
            set_autotune_process_group(previous_group)

    return session()
