"""Coordinate worker-local expert preparation and agreement between ranks."""

import gc
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import NoReturn

import torch
import torch.distributed as dist
from minisgl.distributed.layout import ParallelLayout


@dataclass(frozen=True)
class MoePreparation:
    device: torch.device
    layout: ParallelLayout
    local_rows: tuple[int, ...]
    cpu_group: dist.ProcessGroup | None = None  # Borrowed; startup owner creates/closes once.
    fresh_worker: bool = False
    abort: Callable[[BaseException], NoReturn] | None = None
    tp_cpu_group: dist.ProcessGroup | None = None  # Borrowed full-TP bootstrap.
    max_prefill_tokens: int | None = None
    max_decode_tokens: int | None = None


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


def prepare(backend, model, context):
    """Two public traversals, one owned tuning session; no executor registry."""
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
                if not context.fresh_worker:
                    raise ValueError("Preparation requires a fresh exclusive worker")
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
        sessions = [
            kind.tuning_session(context, count) for kind, count in Counter(map(type, plan)).items()
        ]
        backend.prepare_transport(tuple(plan), context)
        with ExitStack() as stack, torch.inference_mode():
            for session in sessions:
                stack.enter_context(session)
            model.apply(lambda op: op.prepare_runtime(context))
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
