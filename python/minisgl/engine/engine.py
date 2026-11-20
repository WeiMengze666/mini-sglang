from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Iterator, NamedTuple, NoReturn, Tuple

import torch
from minisgl.attention import (
    SUPPORTED_ATTENTION_BACKENDS,
    create_attention_backend,
    load_attention_provider,
)
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import (
    destroy_distributed,
    enable_mnnvl_distributed,
    enable_pynccl_distributed,
    load_mnnvl_provider,
    prepare_mnnvl_provider,
    set_tp_info,
)
from minisgl.distributed.context import init_context_parallel
from minisgl.distributed.impl import TorchDistributedImpl
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.layout import ParallelLayout, configure_parallel, get_parallel_layout
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend, select_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


@dataclass(frozen=True)
class WorkerOwner:
    """Borrow a launcher's fresh FI ownership assertion and fatal worker abort.

    Freshness excludes ambient FI contexts/stores and concurrent FI users; it
    does not mean weights or other startup work have not used the GPU.
    """

    fresh_worker: bool
    abort_callback: Callable[[BaseException], NoReturn]

    def abort(self, error: BaseException) -> NoReturn:
        """Enforce owned-worker termination before device resources can unwind."""
        try:
            self.abort_callback(error)
        finally:
            os._exit(1)


def _prepare_head_all_gather(model, config):
    """Optional dense-head metadata; unsupported model heads keep transport fallback."""
    from minisgl.distributed.parallel import TensorParallel
    from minisgl.layers.linear import Linear

    parallel = TensorParallel()
    head = getattr(model, "lm_head", None)
    rows = width = dtype = None
    if (
        type(head) is Linear
        and head.weight_format == "dense"
        and head.bias is None
        and head.weight.is_meta
        and head.weight.ndim == 2
        and head.weight.dtype == config.dtype == torch.bfloat16
        and head.weight.shape[0] * parallel.size == config.model_config.vocab_size
    ):
        rows = max(
            128,
            config.max_running_req,
            getattr(config, "cuda_graph_max_bs", None) or 0,
            max(getattr(config, "cuda_graph_bs", None) or (0,)),
        )
        width, dtype = head.weight.shape[0], head.weight.dtype
    return parallel.prepare_all_gather(max_rows=rows, shard_width=width, dtype=dtype)


class Engine:
    def __init__(self, config: EngineConfig, *, worker_owner: WorkerOwner | None = None):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        configure_parallel(dcp_size=config.dcp_size, ep_size=config.ep_size)
        _set_runtime_environment(config)
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        if config.attention_provider is not None:
            load_attention_provider(config.attention_provider)
        _adjust_config(config)

        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        self.result_copy_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.worker_owner = worker_owner
        self.ctx = Context(config.logical_page_size)
        self.ctx.max_forward_len = config.max_forward_len
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_context_parallel(backend="nccl", device=self.device, cpu_group=self.tp_cpu_group)
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)
            self.moe_backend.validate_precision(config.mxfp4_activation_precision)
            if getattr(self.moe_backend, "requires_preparation", False) and (
                worker_owner is None or not worker_owner.fresh_worker
            ):
                raise ValueError("This MoE backend requires a fresh owned worker")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        from minisgl.layers.linear import Linear

        self.model.apply(
            lambda op: (
                op.set_backend(
                    config.mxfp4_backend, activation_precision=config.mxfp4_activation_precision
                )
                if isinstance(op, Linear) and op.weight_format == "mxfp4"
                else None
            )
        )
        # Native persistent gather storage must be counted before cache sizing.
        try:
            _prepare_head_all_gather(self.model, config)
        except BaseException as error:
            if worker_owner is not None:
                worker_owner.abort(error)
            raise

        # Hybrid rank capacities must agree with the least-free TP device.
        init_free_memory = self._sync_get_memory()[
            0 if (config.model_config.cache_config is not None) else 1
        ]
        if config.model_config.cache_config is not None:
            self._initial_free_memory = self._local_free_memory
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")
        if config.model_config.cache_config is not None:
            self._check_capacity(init_free_memory, config)
            self.model.load_weights(self._iter_weights(config), device=self.device)
        else:
            self.model.load_state_dict(self._load_weight_state_dict(config))

        if config.model_config.is_moe and worker_owner is not None and worker_owner.fresh_worker:
            try:
                from minisgl.distributed.context import create_expert_cpu_group
                from minisgl.moe.prepare import MoePreparation

                layout = get_parallel_layout()
                resource = create_expert_cpu_group(layout, self.tp_cpu_group)
                self.moe_backend.prepare(
                    self.model,
                    MoePreparation(
                        device=self.device,
                        layout=layout,
                        local_rows=(config.max_running_req,),
                        cpu_group=resource.group if resource is not None else None,
                        fresh_worker=worker_owner.fresh_worker,
                        abort=worker_owner.abort,
                        tp_cpu_group=self.tp_cpu_group,
                        max_prefill_tokens=config.max_forward_len,
                        max_decode_tokens=config.max_running_req,
                    ),
                )
                # Normal completion only: unknown device failures terminate the owned
                # worker through abort, rather than attempting cleanup collectives.
                if resource is not None:
                    resource.close()
            except BaseException as error:
                worker_owner.abort(error)
                raise RuntimeError("Worker abort callable returned") from error

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.logical_page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.logical_page_size,
            device=self.device,
            dtype=self.dtype,
            kv_dtype=config.cache_dtype,
            max_running_req=config.max_running_req,
            state_slot_capacity=self.state_slot_capacity,
            unified_memory=config.enable_unified_memory,
            enable_cache_overlap=config.enable_cache_overlap,
            state_dtype=config.recurrent_state_dtype,
            device_abort=(worker_owner.abort if worker_owner is not None else None),
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            state_cache_strategy=getattr(config, "state_cache_strategy", "auto"),
        )
        self.checkpoint_quantum = _prepare_checkpoint_model(
            self.attn_backend,
            config.model_config.attention_configs,
            self.tp_cpu_group if config.model_config.has_recurrent_state else None,
            (
                str(self.dtype),
                str(config.recurrent_state_dtype),
                config.logical_page_size,
                repr(config.model_config.cache_config),
                config.tp_info.size,
                config.dcp_size,
                config.ep_size,
            ),
            enabled=worker_owner is not None and worker_owner.fresh_worker,
        )
        if (
            getattr(config, "state_cache_strategy", "auto") == "extra_buffer"
            and self.checkpoint_quantum is None
        ):
            raise ValueError(
                "extra_buffer requires an available checkpoint/lease backend capability"
            )

        # Layer-owned replicas are deliberately allocated after the cache budget.
        try:
            self.attn_backend.prepare_communication(dtype=self.dtype)
            self.model.apply(lambda op: op.prepare_inference())
        except BaseException as error:
            if worker_owner is not None:
                worker_owner.abort(error)
            raise

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        max_graph_batch = _graph_batch_capacity(config.max_running_req, num_tokens, self.kv_cache)
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            max_batch_size=max_graph_batch,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        backend = config.distributed_backend
        if backend == "auto":
            backend = "pynccl" if config.tp_info.size == 1 or config.use_pynccl else "torch"
        if backend == "pynccl":
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
            return tp_cpu_group
        if backend == "mnnvl":
            prepare_mnnvl_provider(
                config.distributed_provider, tp_info=config.tp_info, device=self.device
            )
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            device_group = torch.distributed.new_group(backend="nccl")
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            provider = load_mnnvl_provider(
                config.distributed_provider,
                tp_info=config.tp_info,
                cpu_group=tp_cpu_group,
                device_group=device_group,
                device=self.device,
                max_bytes=max_bytes,
            )
            enable_mnnvl_distributed(provider)
            return tp_cpu_group
        if backend == "torch":
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
            TorchDistributedImpl().warmup(self.device)
            return tp_cpu_group
        raise ValueError(f"Unsupported distributed backend: {backend}")

    def _check_capacity(self, free_memory: int, config: EngineConfig) -> None:
        """Reject an impossible rank before downloading or materializing weights.

        This is a lower bound, not a peak-memory guarantee: forward temporaries,
        communication buffers and allocator overhead still need headroom.
        """
        layout = config.model_config.cache_config
        weights = sum(t.numel() * t.element_size() for t in self.model.state_dict().values())
        state_slots = config.state_slot_budget
        fixed = state_slots * layout.state_bytes_per_request(
            self.dtype, tp_size=config.tp_info.size, state_dtype=config.recurrent_state_dtype
        )
        # Two usable pages plus the dummy page are the minimum engine allocation.
        pages = (config.num_page_override or 2) + 1
        cache = pages * layout.cache_bytes_per_page(config.cache_dtype, config.logical_page_size)
        minimum = weights + fixed + cache
        budget = int(config.memory_ratio * free_memory)
        if minimum >= budget:
            raise MemoryError(
                f"Hybrid rank {config.tp_info.rank} needs at least {mem_GB(minimum)} "
                f"for weights/state/cache, but its memory budget is {mem_GB(budget)}. "
                "Increase valid TP size or reduce request/cache capacity. "
                "This estimate excludes forward temporaries."
            )
        logger.info_rank0(
            f"Hybrid per-rank weight storage: {mem_GB(weights)}; "
            f"fixed request state: {mem_GB(fixed)} (peak forward memory not included)."
        )

    def _iter_weights(self, config: EngineConfig) -> Iterator[tuple[str, torch.Tensor]]:
        expected = self.model.state_dict()
        if config.use_dummy_weight:
            for name, tensor in expected.items():
                if tensor.dtype == torch.uint8:
                    # Finite deterministic packed values/scales; randn has no U8 kernel.
                    value = 127 if name.endswith("weight_scale") else 0x22
                    yield name, torch.full_like(tensor, value, device=self.device)
                else:
                    yield name, torch.randn_like(tensor, device=self.device)
            return
        for name, tensor in load_weight(
            config.model_path,
            torch.device("cpu"),
            prefetch=config.weight_loader_prefetch_checkpoints,
            prefetch_num_threads=config.weight_loader_prefetch_num_threads,
        ):
            target = expected.get(name)
            if target is not None and target.is_floating_point() and tensor.is_floating_point():
                tensor = tensor.to(target.dtype)
            yield name, tensor

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            expected = self.model.state_dict()
            return {
                k: v.to(expected[k].dtype) if k in expected else v
                for k, v in load_weight(
                    config.model_path,
                    self.device,
                    prefetch=config.weight_loader_prefetch_checkpoints,
                    prefetch_num_threads=config.weight_loader_prefetch_num_threads,
                )
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        layout = config.model_config.cache_config
        new_free_memory = self._sync_get_memory()[0 if layout is not None else 1]
        self.state_slot_capacity = None
        num_pages = config.num_page_override
        if layout is not None:
            from minisgl.kvcache.config import state_slot_capacity

            cache_per_page = layout.cache_bytes_per_page(
                config.cache_dtype, config.logical_page_size
            )
            state_bytes = layout.state_bytes_per_request(
                self.dtype, tp_size=config.tp_info.size, state_dtype=config.recurrent_state_dtype
            )
            unified = getattr(config, "enable_unified_memory", False)
            if unified:
                from minisgl.kvcache.unified import align_bytes

                cache_per_page, state_bytes = align_bytes(cache_per_page), align_bytes(state_bytes)
            # Keep each rank's before/after pairing. State bytes can differ
            # between ranks: agree on slot count first, then reduce the local
            # token budget after subtracting each rank's own state reservation.
            before = self._initial_free_memory
            available = int(config.memory_ratio * before) - (before - self._local_free_memory)
            ratio = getattr(config, "state_memory_ratio", None)
            total_slots = None
            if ratio is not None:
                state_budget = available
                if unified:
                    pages = available // cache_per_page if num_pages is None else num_pages + 1
                    state_budget = min(available, pages * cache_per_page)
                try:
                    total_slots = state_slot_capacity(state_budget, state_bytes, ratio=ratio)
                except MemoryError:
                    total_slots = 0  # all ranks must reach the capacity collective
                capacity = torch.tensor(total_slots, device="cpu", dtype=torch.int64)
                torch.distributed.all_reduce(
                    capacity, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
                )
                total_slots = int(capacity.item())
                if total_slots < 1:
                    raise MemoryError("State budget on a TP rank cannot hold a physical state slot")
            state_slots = config.max_running_req if total_slots is None else total_slots
            fixed_state = 0 if unified else state_slots * state_bytes
            budget = torch.tensor(available - fixed_state, device="cpu", dtype=torch.int64)
            torch.distributed.all_reduce(
                budget, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
            )
            available = int(budget.item())
            self.state_slot_capacity = total_slots
            logger.info(
                f"Recurrent state pool: {state_slots} shared slots for working, tracking "
                f"and checkpoints; request table capacity {config.max_running_req}; "
                f"budget {mem_GB(state_slots * state_bytes)}"
            )
            max_pages = available // cache_per_page - 1  # dummy page
            if num_pages is None:
                num_pages = max_pages
            elif num_pages > max_pages:
                raise MemoryError(
                    f"Hybrid requested {num_pages} cache pages, but the least-safe TP "
                    f"rank permits {max_pages} within its post-load memory budget"
                )
        else:
            cache_per_page = (
                2
                * config.model_config.head_dim
                * div_even(
                    config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True
                )
                * config.logical_page_size
                * config.cache_dtype.itemsize
                * config.model_config.num_layers
            )
            if num_pages is None:
                available = int(config.memory_ratio * old_free_memory) - (
                    old_free_memory - new_free_memory
                )
                num_pages = available // cache_per_page
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.logical_page_size
        logger.info(
            f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(num_pages * cache_per_page)}"
        )
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        self._local_free_memory = free_memory  # same snapshot as the reduced extrema
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                with self.ctx.kv_cache.model_forward(batch):
                    logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # ForwardOutput owns the sampled IDs; record_stream also protects an
        # early-dropped result from allocator reuse while its D2H is in flight.
        self.result_copy_stream.wait_stream(self.stream)
        with torch.cuda.stream(self.result_copy_stream):
            next_tokens_gpu.record_stream(self.result_copy_stream)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(self.result_copy_stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.graph_runner.destroy_cuda_graphs()
        if hasattr(self, "moe_backend"):
            self.moe_backend.close()
        destroy_distributed()
        torch.distributed.destroy_process_group()


def _prepare_checkpoint_model(backend, records, cpu_group, signature, *, enabled=True):
    """Borrow full-TP Gloo; agree before any request can reserve tracking state."""
    quantum, error = None, None
    try:
        backend.prepare_model(records)
        quantum = backend.checkpoint_quantum() if enabled else None
        if quantum is not None and (type(quantum) is not int or quantum <= 0):
            raise ValueError("Automatic checkpoint quantum must be a positive integer")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    local = (repr(records), signature, quantum, error)
    gathered = [local]
    if cpu_group is not None:
        gathered = [None] * torch.distributed.get_world_size(cpu_group)
        torch.distributed.all_gather_object(gathered, local, group=cpu_group)
    errors = [item[3] for item in gathered if item[3] is not None]
    if errors or any(item[:2] != local[:2] for item in gathered):
        raise RuntimeError(f"Checkpoint startup agreement failed: {errors or 'model signature'}")
    return quantum if quantum is not None and all(item[2] == quantum for item in gathered) else None


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _set_runtime_environment(config: EngineConfig) -> None:
    # Ordinary NCCL defaults match the native runtime. Custom AR's multicast
    # allocations are independent of NCCL's symmetric-memory/NVLS switches.
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    if config.dcp_size > 1:
        os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
    # Native runtime defaults must precede CUDA context/provider creation.
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.cache_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
    ):
        raise ValueError("KV cache storage supports float16, bfloat16, float32 or FP8 E4M3")
    if config.recurrent_state_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Recurrent state storage supports float32 or bfloat16")
    if config.mxfp4_backend not in ("auto", "torch", "triton", "flashinfer"):
        raise ValueError("mxfp4_backend must be auto, torch, triton or flashinfer")
    if config.mxfp4_activation_precision not in ("auto", "bf16", "mxfp8"):
        raise ValueError("mxfp4_activation_precision must be auto, bf16 or mxfp8")
    quantization = getattr(
        getattr(config.model_config, "hf_config", None), "quantization_config", None
    )
    native_mxfp4 = (
        isinstance(quantization, dict)
        and quantization.get("format") == "mxfp4-pack-quantized"
        and config.dtype == torch.bfloat16
        and is_sm100_supported()
    )
    if config.mxfp4_backend == "auto":
        override("mxfp4_backend", "flashinfer" if native_mxfp4 else "torch")
    if config.mxfp4_activation_precision == "auto":
        override(
            "mxfp4_activation_precision",
            "mxfp8" if config.mxfp4_backend == "flashinfer" else "bf16",
        )
    if config.distributed_backend == "auto" and config.tp_info.size > 1 and is_sm100_supported():
        override("distributed_backend", "mnnvl")
        if config.distributed_provider is None:
            override("distributed_provider", "minisgl.distributed.multimem.provider")
    if config.distributed_backend not in ("auto", "torch", "pynccl", "mnnvl"):
        raise ValueError("distributed_backend must be auto, torch, pynccl or mnnvl")
    if config.distributed_backend == "mnnvl" and not config.distributed_provider:
        raise ValueError("distributed_provider is required for the mnnvl backend")
    ParallelLayout(
        DistributedInfo(config.tp_info.rank, config.tp_info.size), config.dcp_size, config.ep_size
    )
    if config.dcp_size > 1 and config.model_config.cache_config is None:
        raise ValueError("DCP requires a context-sharded attention/cache implementation")
    if config.ep_size > 1:
        if not config.model_config.is_moe:
            raise ValueError("EP requires a model with routed experts")
        if config.model_config.num_experts % config.ep_size:
            raise ValueError("num_experts must be divisible by ep_size")
    if config.page_size <= 0:
        raise ValueError("page_size must be positive")
    if config.enable_unified_memory and not config.model_config.has_recurrent_state:
        raise ValueError("Unified memory requires a hybrid recurrent/token cache")
    if (
        config.state_memory_ratio is None
        and config.model_config.has_recurrent_state
        and getattr(config, "cache_type", "naive") != "naive"
    ):
        override("state_memory_ratio", 0.9)  # Native fallback for prefix-cache state/KV bytes.
    if getattr(config, "state_cache_strategy", "auto") == "extra_buffer" and (
        not config.model_config.has_recurrent_state
        or getattr(config, "cache_type", None) == "naive"
    ):
        raise ValueError("extra_buffer requires recurrent prefix caching")
    if config.state_memory_ratio is not None:
        from math import isfinite

        if not isfinite(config.state_memory_ratio) or config.state_memory_ratio <= 0:
            raise ValueError("State/KV memory ratio must be finite and positive")
        if (
            not config.model_config.has_recurrent_state
            or getattr(config, "cache_type", None) == "naive"
        ):
            raise ValueError("State/KV memory ratio requires recurrent prefix caching")

    if config.model_config.is_moe and config.moe_backend == "auto":
        quantization = getattr(config.model_config.hf_config, "quantization_config", None)
        override(
            "moe_backend",
            select_moe_backend(
                quantization,
                dtype=config.dtype,
                activation_precision=config.mxfp4_activation_precision,
            ),
        )
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

    if config.model_config.cache_config is not None:
        config.model_config.cache_config.state_bytes_per_request(
            config.dtype, config.tp_info.size, state_dtype=config.recurrent_state_dtype
        )
        if config.model_config.num_qo_heads % config.tp_info.size:
            raise ValueError("MLA heads must be divisible by TP size")
        requested_graphs = bool(config.cuda_graph_bs) or (
            config.cuda_graph_bs is None and (config.cuda_graph_max_bs or 0) > 0
        )
        if config.attention_backend != "auto":
            backends = [
                SUPPORTED_ATTENTION_BACKENDS[name] for name in config.attention_backend.split(",")
            ]
            if any(not backend.stateful_layers for backend in backends):
                raise ValueError("Selected attention backend does not support stateful layers")
            if requested_graphs and not backends[-1].cuda_graphs:
                raise ValueError("Selected decode attention backend does not support CUDA graphs")
        if not requested_graphs:
            override("cuda_graph_bs", [])
            override("cuda_graph_max_bs", 0)
        if hasattr(config, "cache_type"):
            if config.cache_type != "naive":
                override(
                    "cache_type",
                    "hybrid" if config.state_memory_ratio is not None else "naive",
                )
        logger.warning_rank0(
            f"Hybrid attention {config.attention_backend}: TP={config.tp_info.size}, "
            f"DCP={config.dcp_size}, EP={config.ep_size}; sparse recurrent checkpoints; "
            f"CUDA graphs {'requested' if requested_graphs else 'disabled'}; "
            f"cache overlap {'enabled' if config.enable_cache_overlap else 'disabled'}."
        )
        return

    if config.attention_backend == "auto":
        backend = (
            "trtllm"
            if is_sm100_supported()
            else ("flashattention,flashinfer" if is_sm90_supported() else "flashinfer")
        )
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")


def _graph_batch_capacity(max_running_req: int, num_tokens: int, cache) -> int:
    """Bound capture by cold-request demand without reserving startup leases."""
    maximum = min(max_running_req, num_tokens)
    states = getattr(cache, "recurrent", None)
    if states is not None:
        demand = 1 + states.tracking_buffers
        state_maximum = states.slot_capacity // demand
        if state_maximum < 1:
            raise MemoryError(
                f"Recurrent state capacity {states.slot_capacity} cannot admit one cold "
                f"request requiring {demand} slots"
            )
        maximum = min(maximum, state_maximum)
    return maximum
