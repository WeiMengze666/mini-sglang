"""Explicit typed-payload dispatch/combine; no routing, quantization or model policy."""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from minisgl.distributed.flashinfer_moe import FlashInferMoEWorkspace


class DispatchLayout(NamedTuple):
    """One prepared exchange. Widths and dtypes describe complete token rows."""

    global_experts: int
    top_k: int
    payloads: tuple[tuple[torch.dtype, int], ...]
    output: tuple[torch.dtype, int]

    @classmethod
    def from_payloads(cls, global_experts, ids, payloads, output):
        return cls(
            global_experts, ids.shape[1], tuple((x.dtype, x.shape[1]) for x in payloads), output
        )


@dataclass
class DispatchedTokens:
    """Borrowed receive views plus the single-use inverse-transport handle.

    Only padding slots in expert_ids become -1; foreign expert IDs are unchanged.
    Opaque packed routes retain their bits. Consumers must mask padded packed
    slots and filter local expert ranges before compute.
    Keep this handle through expert compute and combine; views cannot escape it.
    """

    payloads: tuple[torch.Tensor, ...]
    expert_ids: torch.Tensor
    source_counts: tuple[int, ...]
    _owner: "FlashInferTokenDispatcher" = field(repr=False)
    _inputs: tuple[torch.Tensor, ...] = field(repr=False)
    _offset: int | None = field(repr=False)
    _consumed: bool = field(default=False, repr=False)


class FlashInferTokenDispatcher:
    """Optional prepared transport. EP1/default model execution is unaffected.

    source_counts is the group-agreed host plan, supplied by the framework's
    replicated batch metadata. It determines common geometry and all-rank empty
    skips; a local-empty rank must still participate when any peer has tokens.
    """

    @classmethod
    def prepare(
        cls,
        info,
        group,
        device,
        *,
        num_experts,
        top_k,
        payload_layout,
        output_layout,
        max_prefill_tokens,
        max_decode_tokens,
    ):
        if info.size == 1:
            return None
        layout = tuple(payload_layout)
        output_dtype, output_width = output_layout
        allowed = (
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.uint8,
            torch.int32,
            torch.int64,
        )
        error = None
        if (
            type(num_experts) is not int
            or num_experts < info.size
            or num_experts % info.size
            or type(top_k) is not int
            or top_k not in (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22)
            or not 0 < len(layout) <= 5
            or info.size > 64
            or any(
                dtype not in allowed or type(width) is not int or width <= 0
                for dtype, width in layout
            )
            or output_dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or type(output_width) is not int
            or output_width <= 0
            or any(type(n) is not int or n < 0 for n in (max_prefill_tokens, max_decode_tokens))
        ):
            error = "MoE payload/expert layout is outside the native transport contract"
        limits_valid = all(
            type(n) is int and n >= 0 for n in (max_prefill_tokens, max_decode_tokens)
        )
        capacity = (
            (max(max_prefill_tokens, max_decode_tokens) + info.size - 1) // info.size
            if limits_valid
            else 0
        )
        dispatch_bytes = (
            top_k * 4 + sum(dtype.itemsize * width for dtype, width in layout)
            if error is None
            else 0
        )
        combine_bytes = output_dtype.itemsize * output_width if error is None else 0
        signature = (
            num_experts,
            top_k,
            tuple((str(dtype), width) for dtype, width in layout),
            str(output_dtype),
            output_width,
        )
        resource = FlashInferMoEWorkspace.prepare(
            info,
            group,
            device,
            capacity=capacity,
            dispatch_bytes=dispatch_bytes,
            combine_bytes=combine_bytes,
            layout=signature,
            error=error,
        )
        return (
            None if resource is None else cls(resource, num_experts, top_k, layout, output_layout)
        )

    def __init__(self, resource, num_experts, top_k, payload_layout, output_layout):
        self.resource = resource
        self.num_experts, self.top_k = num_experts, top_k
        self.payload_layout, self.output_layout = payload_layout, output_layout
        # Native payload offsets require descending element-size alignment. This
        # only reorders fields; there is no cast or numerical/bit packing.
        layouts = ((torch.int32, top_k), *payload_layout)
        self._order = sorted(
            range(len(layouts)), key=lambda i: layouts[i][0].itemsize, reverse=True
        )
        self._active = None

    def dispatch(self, expert_ids, payloads, *, source_counts):
        r = self.resource
        r.check()
        if r.busy:
            raise RuntimeError("Consume the previous MoE dispatch before reusing its workspace")
        counts = tuple(source_counts)
        n = expert_ids.shape[0]
        if (
            len(counts) != r.info.size
            or any(type(count) is not int or count < 0 for count in counts)
            or counts[r.info.rank] != n
            or max(counts) > r.capacity
            or expert_ids.shape != (n, self.top_k)
            or expert_ids.dtype != torch.int32
            or len(payloads) != len(self.payload_layout)
            or any(
                value.shape != (n, width) or value.dtype != dtype
                for value, (dtype, width) in zip(payloads, self.payload_layout)
            )
            or any(
                not value.is_cuda
                or value.device != r.device
                or not value.is_contiguous()
                or value.requires_grad
                for value in (expert_ids, *payloads)
            )
        ):
            r.failed = True
            raise ValueError("MoE producer violated its prepared layout or shared source plan")
        inputs = (expert_ids, *payloads)
        if not n:
            # Empty views pass PyTorch contiguity; native TVM also checks their strides.
            inputs = tuple(value.new_empty(value.shape) for value in inputs)
        for value in inputs:
            value.record_stream(r.stream)
        m = max(counts)
        r.busy = True
        try:
            if m:
                with r.api.stream():
                    received, offset, _ = r.api.dispatch(
                        inputs[0],
                        [inputs[i] for i in self._order],
                        r.workspace,
                        r.metainfo,
                        m,
                        r.info.rank,
                        r.info.size,
                        self.top_k,
                        self.num_experts,
                        backend="trtllm",
                        recv_view_cache=r.views,
                    )
                    received = tuple(received[self._order.index(i)] for i in range(len(inputs)))
                    r.api.sanitize(
                        received[0], r.workspace, r.metainfo, r.info.rank, -1, backend="trtllm"
                    )
            else:
                received = tuple(
                    value.new_empty((r.info.size, 0, value.shape[1])) for value in inputs
                )
                offset = None
            handle = DispatchedTokens(
                tuple(received[1:]), received[0], counts, self, inputs, offset
            )
            self._active = handle
            return handle
        except BaseException:
            r.failed = True
            raise

    def combine(self, handle, partials):
        r = self.resource
        r.check()
        if (
            not isinstance(handle, DispatchedTokens)
            or handle._owner is not self
            or handle._consumed
            or self._active is not handle
        ):
            raise RuntimeError(
                "MoE combine requires this dispatcher's outstanding single-use handle"
            )
        m = max(handle.source_counts)
        dtype, width = self.output_layout
        if (
            partials.shape != (r.info.size * m, width)
            or partials.dtype != dtype
            or not partials.is_cuda
            or partials.device != r.device
            or not partials.is_contiguous()
            or partials.requires_grad
        ):
            r.failed = True
            raise ValueError(
                "MoE combine must preserve the prepared expert-output precision/layout"
            )
        handle._consumed = True  # A native failure cannot retry this handle.
        partials.record_stream(r.stream)
        try:
            if m:
                with r.api.stream():
                    result = r.api.combine(
                        partials.view(r.info.size, m, width),
                        handle.source_counts[r.info.rank],
                        r.workspace,
                        r.metainfo,
                        m,
                        r.info.rank,
                        r.info.size,
                        self.top_k,
                        handle._offset,
                        payload_in_workspace=False,
                        output_dtype=dtype,
                        use_low_precision=False,
                        backend="trtllm",
                    )
            else:
                result = partials.new_empty((0, width))
            self._active = None
            r.busy = False
            return result
        except BaseException:
            r.failed = True
            raise

    def close(self):
        self.resource.close()
