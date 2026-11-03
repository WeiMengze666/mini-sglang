from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Iterator, NamedTuple, Tuple

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
    set_tp_info,
)
from minisgl.distributed.context import init_context_parallel
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.layout import ParallelLayout, configure_parallel
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend, select_moe_backend
from minisgl.precision import native_precision_scope, resolve_bf16_precision
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        configure_parallel(dcp_size=config.dcp_size, ep_size=config.ep_size)
        if config.attention_provider is not None:
            load_attention_provider(config.attention_provider)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.logical_page_size)
        self.ctx.bf16_precision = resolve_bf16_precision(config.bf16_precision)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_context_parallel(backend="nccl")
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)
            self.moe_backend.validate_precision(config.mxfp4_activation_precision)
        # Hybrid rank capacities must agree with the least-free TP device.
        init_free_memory = self._sync_get_memory()[
            0 if (config.model_config.cache_config is not None) else 1
        ]
        if config.model_config.cache_config is not None:
            self._initial_free_memory = self._local_free_memory
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

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
        if config.model_config.cache_config is not None:
            self._check_capacity(init_free_memory, config)
            self.model.load_weights(self._iter_weights(config), device=self.device)
        else:
            self.model.load_state_dict(self._load_weight_state_dict(config))

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
            max_cached_states=(
                getattr(config, "max_cached_states", 0)
                if getattr(config, "cache_type", "hybrid") != "naive"
                else 0
            ),
            unified_memory=config.enable_unified_memory,
            enable_cache_overlap=config.enable_cache_overlap,
            state_dtype=config.recurrent_state_dtype,
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
            config.attention_backend, config.model_config
        )

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
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
        self._precision_scope = native_precision_scope(
            self.ctx.bf16_precision, device=self.device, dtype=self.dtype
        )
        try:
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
                max_batch_size=min(config.max_running_req, num_tokens),
            )
        except BaseException:
            self._precision_scope.close()
            raise

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
            # Registered staging buffers can change NCCL's BF16 reduction order.
            # Strict precision uses caller buffers, matching direct NCCL collectives.
            max_bytes = (
                0
                if self.ctx.bf16_precision == "strict"
                else config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
            return tp_cpu_group
        if backend == "mnnvl":
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
        for name, tensor in load_weight(config.model_path, torch.device("cpu")):
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
                for k, v in load_weight(config.model_path, self.device)
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[
            0 if (config.model_config.cache_config is not None) else 1
        ]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.logical_page_size
            * config.cache_dtype.itemsize
            * config.model_config.num_layers
        )
        fixed_state = 0
        if config.model_config.cache_config is not None:
            layout = config.model_config.cache_config
            cache_per_page = layout.cache_bytes_per_page(
                config.cache_dtype, config.logical_page_size
            )
            state_slots = config.state_slot_budget
            fixed_state = state_slots * layout.state_bytes_per_request(
                self.dtype, tp_size=config.tp_info.size, state_dtype=config.recurrent_state_dtype
            )
            if getattr(config, "enable_unified_memory", False):
                from minisgl.kvcache.unified import align_bytes

                cache_per_page = align_bytes(cache_per_page)
                fixed_state = 0  # states and token pages draw from the same bytes
        num_pages = config.num_page_override
        if config.model_config.cache_config is not None:
            # Preserve each rank's before/after pairing. Subtracting separately
            # reduced minima can spend another rank's reserved ratio headroom.
            before = self._initial_free_memory
            available_memory = (
                int(config.memory_ratio * before) - (before - self._local_free_memory) - fixed_state
            )
            budget = torch.tensor(available_memory, device="cpu", dtype=torch.int64)
            torch.distributed.all_reduce(
                budget, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
            )
            max_pages = int(budget.item()) // cache_per_page - 1  # dummy page
            if num_pages is None:
                num_pages = max_pages
            elif num_pages > max_pages:
                raise MemoryError(
                    f"Hybrid requested {num_pages} cache pages, but the least-safe TP "
                    f"rank permits {max_pages} within its post-load memory budget"
                )
        elif num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = (
                int(config.memory_ratio * old_free_memory) - model_memory - fixed_state
            )
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.logical_page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
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
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        try:
            torch.cuda.synchronize(self.device)
            self.graph_runner.destroy_cuda_graphs()
            destroy_distributed()
            torch.distributed.destroy_process_group()
        finally:
            self._precision_scope.close()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


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
    if config.max_cached_states < 0:
        raise ValueError("max_cached_states cannot be negative")
    if config.mxfp4_backend not in ("torch", "triton", "flashinfer"):
        raise ValueError("mxfp4_backend must be torch, triton or flashinfer")
    if config.mxfp4_activation_precision not in ("bf16", "mxfp8"):
        raise ValueError("mxfp4_activation_precision must be bf16 or mxfp8")
    if config.distributed_backend not in ("auto", "torch", "pynccl", "mnnvl"):
        raise ValueError("distributed_backend must be auto, torch, pynccl or mnnvl")
    if config.distributed_backend == "mnnvl" and not config.distributed_provider:
        raise ValueError("distributed_provider is required for the mnnvl backend")
    resolve_bf16_precision(config.bf16_precision)
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
        if config.attention_backend == "auto":
            override("attention_backend", "sgl")
        backends = [
            SUPPORTED_ATTENTION_BACKENDS[name] for name in config.attention_backend.split(",")
        ]
        if any(not backend.stateful_layers for backend in backends):
            raise ValueError("Selected attention backend does not support stateful layers")
        requested_graphs = bool(config.cuda_graph_bs) or (
            config.cuda_graph_bs is None and (config.cuda_graph_max_bs or 0) > 0
        )
        if requested_graphs and not backends[-1].cuda_graphs:
            raise ValueError("Selected decode attention backend does not support CUDA graphs")
        if not requested_graphs:
            override("cuda_graph_bs", [])
            override("cuda_graph_max_bs", 0)
        if hasattr(config, "cache_type"):
            if config.cache_type != "naive":
                override("cache_type", "hybrid" if config.max_cached_states > 0 else "naive")
        logger.warning_rank0(
            f"Hybrid attention {config.attention_backend}: TP={config.tp_info.size}, "
            f"DCP={config.dcp_size}, EP={config.ep_size}; sparse recurrent checkpoints; "
            f"CUDA graphs {'requested' if requested_graphs else 'disabled'}; "
            f"cache overlap {'enabled' if config.enable_cache_overlap else 'disabled'}."
        )
        return

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")
