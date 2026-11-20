"""Shared/routed output scheduling, projection and reduction."""

import torch

from .base import DeferredMoEOutput


class MoEEpilogue:
    def __init__(self, reduce_outputs):
        self._reduce = reduce_outputs

    def prepare_output_stream(self, output: torch.Tensor) -> None:
        """One backend-owned stream, created after reservation and before capture."""
        if not output.is_cuda:
            return
        stream = getattr(self, "_shared_stream", None)
        if stream is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare the MoE output stream before graph capture")
            self._shared_stream = torch.cuda.Stream(device=output.device)
        elif stream.device != output.device:
            raise ValueError("A MoE backend cannot reuse its output stream on another device")

    def project_outputs(
        self,
        parallel,
        mixed,
        shared,
        projection=None,
        *,
        output_dtype=None,
        prepared=False,
        residual=None,
    ):
        """Project the reduced mixture and add the already-reduced shared result.

        prepared identifies the existing same-dtype producer slots. The optional
        collective consumes replicated mixture/shared tensors and a FULL replicated
        dense projection weight; it returns owned storage. Other paths preserve
        the projection's own forward and the original shared cast-before-add.
        Stream dependencies and input lifetimes belong to forward_outputs.
        residual is consumed only here, after any deferred finalize/reduction/norm;
        the internal DeferredMoEOutput never includes this final residual.
        """
        from minisgl.layers.elementwise import add3
        from minisgl.layers.linear import Linear

        if type(prepared) is not bool:
            raise ValueError("Projection preparation hint must be a bool")
        dtype = mixed.dtype if output_dtype is None else output_dtype
        if (
            prepared
            and mixed.is_cuda
            and shared is not None
            and type(projection) is Linear
            and projection.weight_format == "dense"
            and projection.bias is None
            and mixed.dtype == shared.dtype == projection.weight.dtype == dtype == torch.bfloat16
        ):
            from minisgl.distributed.layout import get_parallel_layout

            if get_parallel_layout().ep_size == 1 and parallel.supports_linear_all_gather(
                mixed, projection.weight, shared
            ):
                result = parallel.linear_all_gather(mixed, projection.weight, shared)
                return result if residual is None else result + residual
        result = projection.forward(mixed) if projection is not None else mixed
        if shared is not None:
            return add3(result, shared.to(dtype), residual)
        return result if residual is None else result + residual

    def forward_outputs(
        self,
        parallel,
        expert_producer,
        shared_producer=None,
        *,
        expert_out=None,
        shared_out=None,
        packed=None,
        output_key=None,
        norm=None,
        output_dtype=None,
        shared_inputs=(),
        deferred_producer=None,
        topk=None,
    ):
        """Own local producers and reduction order, including optional overlap.

        Only already-prepared same-dtype outputs use the side stream. Its shared
        pull must finish before latent AR touches the communicator's same
        semaphore plane. Returned tensors are ready on the invoking stream;
        this is a scheduling hint, not a Future or caller-wait contract.
        All other paths preserve serial mixture precision.
        """
        stream = getattr(self, "_shared_stream", None)
        overlap = (
            stream is not None
            and shared_producer is not None
            and packed is not None
            and output_key is not None
            and expert_out is not None
            and shared_out is not None
            and packed.dtype == output_dtype == torch.bfloat16
        )
        epilogue = None
        if overlap and deferred_producer is not None:
            from minisgl.layers.norm import RMSNormEpilogue, TorchRMSNorm

            if (
                isinstance(norm, TorchRMSNorm)
                and norm.weight.dtype == output_dtype
                and expert_out.ndim == 2
                and expert_out.shape[1] == norm.weight.numel()
            ):
                candidate = RMSNormEpilogue(norm, expert_out.shape[0])
                supports = getattr(parallel, "supports_weighted_all_reduce", None)
                if callable(supports) and supports(expert_out, norm=candidate, topk=topk):
                    epilogue = candidate
        if overlap:
            if packed.device != stream.device:
                raise ValueError("Prepared MoE outputs differ from their backend stream device")
            main = torch.cuda.current_stream(device=packed.device)
            stream.wait_stream(main)  # Front is ready; this edge precedes expert launch.
            for value in shared_inputs:
                if value.is_cuda:
                    value.record_stream(stream)
            # Select the output-only consumer before any expert route/quant/GEMM launch.
            mixed = (
                deferred_producer(expert_out)
                if epilogue is not None
                else expert_producer(expert_out)
            )
            with torch.cuda.stream(stream):
                shared = shared_producer(shared_out)
                shared = parallel.all_reduce_residual(shared, output_key=output_key, overlap=True)
            main.wait_stream(stream)  # Shared pull is complete before latent AR/norm.
            if shared.is_cuda:
                # A generic reduction may return side-stream-owned storage,
                # rather than an alias of the persistent shared destination.
                shared.record_stream(main)
            if epilogue is not None:
                if (
                    not isinstance(mixed, DeferredMoEOutput)
                    or mixed.weights.shape != (expert_out.shape[0], topk)
                    or mixed.indices.numel() != expert_out.shape[0] * topk
                ):
                    raise ValueError(
                        "Deferred expert output does not match its selected row mapping"
                    )
                # Hold all three native outputs through their same-stream consumer.
                # It writes the reserved latent destination, finalizes, reduces and
                # normalizes once; no ordinary AR or second normalizer follows.
                mixed = parallel.weighted_all_reduce(
                    mixed.values,
                    mixed.indices,
                    mixed.weights,
                    out=expert_out,
                    norm=epilogue,
                )
            else:
                (mixed,) = self._reduce(
                    parallel,
                    mixed,
                    packed=expert_out.view(-1),
                    output_key=output_key,
                    norm=norm,
                    output_dtype=output_dtype,
                )
            return mixed, shared
        mixed = expert_producer(expert_out)
        shared = shared_producer(shared_out) if shared_producer is not None else None
        options = {"norm": norm, "output_dtype": output_dtype}
        if packed is not None:
            options.update(packed=packed, output_key=output_key)
        if shared is None:
            return self._reduce(parallel, mixed, **options)[0], None
        return self._reduce(parallel, mixed, shared, **options)


def reduce_outputs(parallel, *outputs, packed=None, output_key=None, norm=None, output_dtype=None):
    """Reduce local partials, then cast and optionally normalize the mixture.

    A prepared same-dtype RMS epilogue can normalize the latent prefix in
    the selected communication backend. Other paths keep the original
    reduce -> mixture cast -> normalizer order, including FP32 partials.
    """
    from minisgl.layers.norm import RMSNormEpilogue, TorchRMSNorm

    if not outputs:
        raise ValueError("MoE reduction needs at least one output")
    sizes = [part.numel() for part in outputs]
    epilogue = None
    if packed is not None:
        if packed.ndim != 1 or not packed.is_contiguous() or packed.numel() != sum(sizes):
            raise ValueError(
                "Caller-owned packed MoE output must be an exact contiguous flat tensor"
            )
        offset = 0
        for part in outputs:
            if (
                part.dtype != packed.dtype
                or part.device != packed.device
                or not part.is_contiguous()
                or part.data_ptr() != packed.data_ptr() + offset * packed.element_size()
            ):
                raise ValueError(
                    "MoE partials must be contiguous consecutive blocks of packed output"
                )
            offset += part.numel()
        if (
            isinstance(norm, TorchRMSNorm)
            and packed.dtype == output_dtype == norm.weight.dtype == torch.bfloat16
            and outputs[0].ndim == 2
            and outputs[0].shape[1] == norm.weight.numel()
        ):
            epilogue = RMSNormEpilogue(norm, outputs[0].shape[0])
        options = {} if epilogue is None else {"norm": epilogue}
        reduced = parallel.all_reduce_residual(packed, output_key=output_key, **options)
        parts = tuple(
            part.view(original.shape) for part, original in zip(reduced.split(sizes), outputs)
        )
    elif len(outputs) == 1:
        parts = (parallel.all_reduce(outputs[0]),)
    else:
        value = torch.cat([part.to(outputs[0].dtype).flatten() for part in outputs])
        reduced = parallel.all_reduce(value)
        parts = tuple(
            part.view(original.shape) for part, original in zip(reduced.split(sizes), outputs)
        )
    if epilogue is not None:
        return parts  # Native or callable fallback already normalized the prefix once.
    mixed, *tail = parts
    if output_dtype is not None:
        mixed = mixed.to(output_dtype)
    if norm is not None:
        mixed = norm.forward(mixed)
    return (mixed, *tail)
