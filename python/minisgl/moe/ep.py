"""Experimental native token dispatch for existing grouped expert executors."""

import gc
from dataclasses import replace

from minisgl.distributed.layout import get_parallel_layout

from .base import BaseMoeBackend
from .dispatch import DispatchLayout, FlashInferTokenDispatcher
from .experts.fused import FusedMoe
from .routing import grouped_topk


class FlashInferEPBackend(BaseMoeBackend):
    """Opt-in EP/ETP; routing and local expert math retain their existing ABI.

    One serial execution stream owns every prepared channel. Compatible layers
    share the same bounded transport resources; no executor registry is retained.
    This backend remains experimental until native component qualification.
    """

    route = staticmethod(grouped_topk)

    def __init__(self):
        self.local = FusedMoe()
        self._mesh = None
        self._dispatchers = {}
        self._ready = False
        self._abort = None

    @property
    def requires_preparation(self) -> bool:
        return get_parallel_layout().ep_size > 1

    def prepare(self, model, context):
        from .prepare import prepare

        prepare(self, model, context)
        if context.layout.ep_size > 1 and not self._ready:
            raise NotImplementedError("Native EP requires compatible prepared expert layers")

    def prepare_local(self, executor, routing_builder, context, *, dry_run=False):
        plan = self.local.prepare_local(executor, routing_builder, context, dry_run=dry_run)
        if dry_run and context.layout.ep_size > 1 and not getattr(plan, "dispatch_layouts", ()):
            raise NotImplementedError("Native EP requires an executor with a token exchange layout")
        return plan

    def prepare_transport(self, plan, context):
        if context.layout.ep_size == 1:
            return
        if self._mesh is not None or self._ready:
            raise RuntimeError("Native EP resources must be prepared once per worker")
        from minisgl.distributed.expert import ExpertParallel

        from .prepare import _agree

        specs = tuple(dict.fromkeys(spec for item in plan for spec in item.dispatch_layouts))
        if (
            not specs
            or context.tp_cpu_group is None
            or not context.fresh_worker
            or not callable(context.abort)
            or any(not item.dispatch_layouts for item in plan)
            or any(
                type(n) is not int or n < 1
                for n in (context.max_prefill_tokens, context.max_decode_tokens)
            )
        ):
            raise ValueError("Native EP needs an owned full-TP plan and explicit token budgets")
        # ETP tactic agreement cannot substitute for EP resource agreement.
        _agree(
            (specs, context.max_prefill_tokens, context.max_decode_tokens),
            replace(context, cpu_group=context.tp_cpu_group),
        )
        self._abort = context.abort
        self._mesh = ExpertParallel.prepare(context.layout, context.tp_cpu_group, context.device)
        if self._mesh is None:
            raise NotImplementedError("Native EP mesh is unavailable")
        for spec in specs:
            dispatcher = FlashInferTokenDispatcher.prepare(
                self._mesh.info,
                self._mesh.cpu_group,
                context.device,
                num_experts=spec.global_experts,
                top_k=spec.top_k,
                payload_layout=spec.payloads,
                output_layout=spec.output,
                max_prefill_tokens=context.max_prefill_tokens,
                max_decode_tokens=context.max_decode_tokens,
            )
            if dispatcher is None:
                raise NotImplementedError("Native EP transport is unavailable for this layout")
            self._dispatchers[spec] = dispatcher
        self._ready = True

    def forward_experts(self, executor, x, routing, *, out=None, deferred=False):
        if get_parallel_layout().ep_size == 1:
            return self.local.forward_experts(executor, x, routing, out=out, deferred=deferred)
        if not self._ready:
            raise RuntimeError("Native EP must be prepared before inference")
        if out is not None or deferred:
            raise NotImplementedError("Native EP keeps serial finalized expert outputs")
        if not len(x):
            return x.new_empty((0, x.shape[1]), dtype=executor.dispatch_output_dtype)
        try:
            # Preserve the full-batch frontend choice, especially its N<=64 boundary.
            ids, payloads = executor.prepare_dispatch_inputs(self, x, routing)
            key = DispatchLayout.from_payloads(
                executor.global_experts, ids, payloads, (executor.dispatch_output_dtype, x.shape[1])
            )
            dispatcher = self._dispatchers.get(key)
            if dispatcher is None:
                raise NotImplementedError("Expert frontend was not in the prepared EP portfolio")
            rank = self._mesh.info.rank
            counts = self._mesh.source_counts(len(x))
            handle = dispatcher.dispatch(
                ids[rank :: self._mesh.info.size].contiguous(),
                tuple(value[rank :: self._mesh.info.size].contiguous() for value in payloads),
                source_counts=counts,
            )
            partials = executor.forward_dispatched(handle.expert_ids, handle.payloads)
            source = dispatcher.combine(handle, partials)
            source = self._mesh.reduce_source(source)
            return self._mesh.gather_source(source, len(x))
        except BaseException as error:
            self._abort_failure(error)

    def reduce_outputs(
        self, parallel, *outputs, packed=None, output_key=None, norm=None, output_dtype=None
    ):
        if get_parallel_layout().ep_size == 1:
            return self.local.reduce_outputs(
                parallel,
                *outputs,
                packed=packed,
                output_key=output_key,
                norm=norm,
                output_dtype=output_dtype,
            )
        if not self._ready or not outputs or packed is not None:
            raise ValueError("Native EP requires prepared serial mixture outputs")
        # Routed rows are already restored and summed across EP/ETP. Only shared
        # experts still need their original full-TP reduction, exactly once.
        try:
            mixed, *shared = outputs
            shared = tuple(parallel.all_reduce(value) for value in shared)
            if output_dtype is not None:
                mixed = mixed.to(output_dtype)
            if norm is not None:
                mixed = norm.forward(mixed)
            return mixed, *shared
        except BaseException as error:
            self._abort_failure(error)

    def forward(self, *args, **kwargs):
        if get_parallel_layout().ep_size > 1:
            raise NotImplementedError("Native EP requires the prepared expert-executor interface")
        return self.local.forward(*args, **kwargs)

    def _abort_failure(self, error):
        # Native work may have advanced peer epochs. Never retry or enter a
        # cleanup collective after an unknown device/communication failure.
        if self._abort is not None:
            self._abort(error)
            raise RuntimeError("Worker abort callable returned") from error
        raise error

    def close(self):
        if not self._dispatchers and self._mesh is None:
            return
        try:
            for dispatcher in self._dispatchers.values():
                dispatcher.close()
            dispatcher = None  # Drop the loop's last native owner before its group.
            self._dispatchers.clear()
            gc.collect()
            if self._mesh is not None:
                self._mesh.close()
                self._mesh = None
            self._ready = False
            self._abort = None
        except BaseException as error:
            self._abort_failure(error)
