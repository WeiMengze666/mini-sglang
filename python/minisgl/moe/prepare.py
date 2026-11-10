"""Optional fresh-worker FI preparation. The Engine does not call this entry.

fresh_worker is a launcher ownership contract, not ambient-store detection:
no previous FI context/v2/store, bundled cache, serving or concurrent FI user.
clear_cache does not detach ambient stores. Grouped failures require an external
worker supervisor; never try a recovery collective after an unknown failure.
"""

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

import torch
import torch.distributed as dist
from minisgl.distributed.layout import ParallelLayout

from .flashinfer import FlashInferMXFP4Experts


@dataclass(frozen=True)
class MoePreparation:
    device: torch.device
    layout: ParallelLayout
    local_rows: tuple[int, ...]
    cpu_group: dist.ProcessGroup | None = None  # Borrowed; startup owner creates/closes once.
    precision: str = "default"
    fresh_worker: bool = False
    abort: Callable[[BaseException], NoReturn] | None = None


def _agree(value, context, *, error=None):
    # Only CPU preflight supplies an error; unknown device failures use the supervisor.
    failure = None if error is None else (type(error).__name__, str(error))
    status = failure, value
    values = [status]
    if context.cpu_group is not None:
        values = [None] * dist.get_world_size(context.cpu_group)
        dist.all_gather_object(values, status, group=context.cpu_group)
    failures = [item[0] for item in values if item[0] is not None]
    if failures:
        raise ValueError(f"Expert preparation preflight failed: {failures}") from error
    if any(item[1] != value for item in values):
        raise ValueError("Expert preparation plan/ABI differs within expert-TP")


def _tensor(value, device=None):
    if isinstance(value, (tuple, list)):
        return tuple(_tensor(item, device) for item in value)
    if not isinstance(value, torch.Tensor):
        return value
    if device is not None and value.device != torch.device(device):
        raise ValueError("Expert preparation tensor device differs from context.device")
    return (tuple(value.shape), str(value.dtype), value.stride(), value.data_ptr() % 256)


def _signature(executor, routing, context):
    return (
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


def prepare_local(backend, executor, routing_builder, context, *, dry_run=False):
    if not isinstance(executor, FlashInferMXFP4Experts):
        return None
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
            ids = routes[0] if isinstance(routes, tuple) else routes >> 16
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


def prepare(backend, model, context):
    """Two public traversals, one owned tuning session; no executor registry."""
    previous_group = None
    try:
        size = context.layout.expert_tensor.size
        if size == 1:
            if context.cpu_group is not None:
                raise ValueError("Singleton expert execution must prepare locally")
        elif (
            context.cpu_group is None
            or str(dist.get_backend(context.cpu_group)).lower() != "gloo"
            or dist.get_world_size(context.cpu_group) != size
            or dist.get_rank(context.cpu_group) != context.layout.expert_tensor.rank
        ):
            raise ValueError("Grouped preparation requires its ETP Gloo group")
        plan = []

        def collect(op):
            signature = op.prepare_runtime(context, dry_run=True)
            if signature is not None:
                plan.append(signature)

        preflight_error = None
        try:
            model.apply(collect)
            if plan:
                if not context.fresh_worker or context.precision != "default":
                    raise ValueError(
                        "Preparation requires a fresh exclusive default-precision worker"
                    )
                if not context.local_rows or any(
                    type(rows) is not int or rows < 1 for rows in context.local_rows
                ):
                    raise ValueError("Preparation needs explicit positive local row bounds")
                if not callable(context.abort):
                    raise ValueError("FI preparation requires an external worker supervisor")
        except BaseException as error:
            preflight_error = error
        _agree(tuple(plan), context, error=preflight_error)
        if not plan:
            return
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
        tuner.clear_cache()  # Owned process caches only; fresh/no-ambient contract is required.
        _agree(("owned-empty-start", len(plan)), context)
        try:
            set_autotune_process_group(context.cpu_group)
            with torch.inference_mode(), autotune(True, cache=None, skip_ops=set()):
                model.apply(lambda op: op.prepare_runtime(context))
        finally:
            set_autotune_process_group(previous_group)
        # Per-representative frames are gone, including routing/quant outputs.
        torch.cuda.synchronize(context.device)
        gc.collect()
        torch.cuda.empty_cache()
        _agree(("complete", len(plan)), context)
    except BaseException as error:
        if context.abort is not None:
            context.abort(error)
            raise RuntimeError("Worker supervisor callback returned") from error
        raise
