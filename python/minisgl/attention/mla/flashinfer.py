"""FlashInfer expanded prefill and paged compressed decode, with portable fallback."""

import torch
import torch.nn.functional as F

from .reference import (
    CausalLayout,
    MLALatentOutput,
    MLAPagedMetadata,
    mla_attention,
)


class _MLAPlan:
    def __init__(self, workspace, *, batch_size=None, max_pages=None):
        from flashinfer import BatchMLAPagedAttentionWrapper

        # Keep a native plan and all of its workspace under one owner.
        buffers = {}
        if batch_size is not None:

            def empty(size):
                return torch.empty(size, dtype=torch.int32, device=workspace.device)

            buffers = {
                "use_cuda_graph": True,
                "qo_indptr": empty(batch_size + 1),
                "kv_indptr": empty(batch_size + 1),
                "kv_indices": empty(batch_size * max_pages),
                "kv_len_arr": empty(batch_size),
            }
        self.wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2", **buffers)
        self.device = workspace.device
        self.event = torch.cuda.Event()
        self.metadata, self.options = None, None

    def prepare(self, metadata, options):
        if self.metadata is metadata and self.options == options:
            return
        from flashinfer.mla import MLAPlanMetadata

        if self.metadata is not None:
            # plan() reuses pinned host staging storage for its asynchronous copy.
            self.event.synchronize()
        from tvm_ffi import use_torch_stream

        # Native FFI streams are independent of PyTorch's current stream.
        with use_torch_stream():
            self.wrapper.plan(
                metadata=MLAPlanMetadata.csr(
                    torch.arange(len(metadata.lengths) + 1, dtype=torch.int32),
                    metadata.indptr,
                    metadata.indices,
                    metadata.lengths,
                ),
                **options,
            )
        self.event.record()
        self.metadata, self.options = metadata, options

    def run(self, **kwargs):
        from tvm_ffi import use_torch_stream

        with use_torch_stream():
            return self.wrapper.run(**kwargs)


class FlashInferMLAAdapter:
    def __init__(self, *, kv_dtype: torch.dtype | None = None):
        self.kv_dtype = kv_dtype
        self._expanded_workspace = None
        self._decode_plan = None
        self._decode_workspace = None
        self._graph_plans = {}
        self._graph_sizes = set()
        self._max_pages = 0

    def init_graphs(self, batch_sizes, max_pages, device):
        # Plans are materialized during warmup once dtype/shape capabilities are
        # known. Graphs share one decode workspace; prefill owns a separate one.
        self._graph_sizes = set(batch_sizes)
        self._max_pages = max_pages
        self._graph_plans.clear()

    def _plan(self, q, metadata, native):
        from .cutedsl import CuTeMLADecodePlan

        cls = CuTeMLADecodePlan if native else _MLAPlan
        size = metadata.capture_size
        plan = self._graph_plans.get(size) if size is not None else self._decode_plan
        if plan is not None and isinstance(plan, cls) and plan.device == q.device:
            return plan
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MLA plans must be warmed before CUDA graph capture")
        if self._decode_workspace is None or self._decode_workspace.device != q.device:
            self._decode_workspace = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
        if size is not None:
            if size not in self._graph_sizes:
                raise ValueError("No MLA graph was initialized for this batch size")
            plan = cls(self._decode_workspace, batch_size=size, max_pages=self._max_pages)
            self._graph_plans[size] = plan
        else:
            plan = self._decode_plan = cls(self._decode_workspace)
        return plan

    def prepare_replay(self, metadata, batch_size):
        plan = self._graph_plans[batch_size]
        metadata.capture_size = batch_size
        plan.prepare(metadata, plan.options)

    def decode(
        self,
        q,
        pages,
        weight,
        metadata: MLAPagedMetadata,
        *,
        nope_dim,
        value_dim,
        scale=None,
    ):
        """Attend a whole decode batch directly over the cache's physical pages.

        The wrapper is replanned once per batch and shared by compatible layers.
        Returns compressed values and LSE; the layer owns context exchange and
        value projection. Unsupported dtypes/dimensions return None.
        """
        combined = pages if isinstance(pages, torch.Tensor) else None
        latent, key = (pages[..., :512], pages[..., 512:]) if combined is not None else pages
        if (
            not q.is_cuda
            or q.dtype != torch.bfloat16
            or latent.shape[-1] != 512
            or key.shape[-1] != 64
        ):
            return None
        heads, dim = q.shape[1:]
        padded_heads = max(16, 1 << (heads - 1).bit_length())
        softmax_scale = dim**-0.5 if scale is None else scale
        native = (
            combined is not None
            and metadata.block_table is not None
            and metadata.page_size == 64
            and (nope_dim, value_dim) == (128, 128)
            and torch.cuda.get_device_capability(q.device)[0] >= 10
        )
        plan = self._plan(q, metadata, native)
        plan.prepare(
            metadata,
            {
                "num_heads": padded_heads,
                "head_dim_ckv": 512,
                "head_dim_kpe": 64,
                "page_size": metadata.page_size,
                "causal": False,
                "sm_scale": softmax_scale,
                "q_data_type": q.dtype,
                "kv_data_type": latent.dtype,
                "query_layout": "split",
                "kv_cache_layout": "split",
                "lse_mode": "basee",
            },
        )
        weights = weight.view(heads, -1, 512)
        query = torch.bmm(q[..., :nope_dim].transpose(0, 1), weights[:, :nope_dim]).transpose(0, 1)
        if native:
            output, lse = plan.run(
                (query, q[..., nope_dim:]), combined, nope_dim=nope_dim, scale=softmax_scale
            )
            return MLALatentOutput(output, lse)
        query = F.pad(query, (0, 0, 0, padded_heads - heads))
        suffix = F.pad(q[..., nope_dim:], (0, 0, 0, padded_heads - heads)).contiguous()
        output, lse = plan.run(
            query=(query, suffix),
            kv_cache=(latent, key),
            return_lse=True,
            return_lse_base_on_e=True,
        )
        output, lse = output[:, :heads], lse[:, :heads]
        if metadata.nonempty is not None:
            output = output.masked_fill(~metadata.nonempty[:, None, None], 0)
            lse = lse.masked_fill(~metadata.nonempty[:, None], -torch.inf)
        return MLALatentOutput(output, lse)

    def prepare_prefill(self, batch, page_table, cp, cu_lengths):
        from sglang.srt.environ import envs

        from .prefill import MLAPrefillMetadata

        return MLAPrefillMetadata.from_batch(
            batch,
            page_table,
            cp,
            envs.SGLANG_MAX_KV_CHUNK_CAPACITY.get(),
            cu_lengths=cu_lengths,
        )

    def _run_ragged_prefill(
        self,
        q,
        latent,
        key,
        weight,
        queries,
        keys_layout,
        *,
        nope_dim,
        value_dim,
        scale,
        causal,
        return_lse,
    ):
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
        from minisgl.layers.linear import linear_bf16

        if self._expanded_workspace is None or self._expanded_workspace.device != q.device:
            self._expanded_workspace = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
        expanded = linear_bf16(latent, weight).view(len(latent), q.shape[1], nope_dim + value_dim)
        keys = torch.cat((expanded[..., :nope_dim], key[:, None].expand(-1, q.shape[1], -1)), -1)
        query, values = q.contiguous(), expanded[..., nope_dim:]
        if self.kv_dtype == torch.float8_e4m3fn:
            query, keys, values = (part.to(self.kv_dtype) for part in (query, keys, values))
        output = q.new_empty((len(q), q.shape[1], value_dim))
        result = trtllm_ragged_attention_deepseek(
            query=query,
            key=keys,
            value=values,
            out=output,
            workspace_buffer=self._expanded_workspace,
            seq_lens=keys_layout.lengths,
            cum_seq_lens_q=queries.cu_lengths,
            cum_seq_lens_kv=keys_layout.cu_lengths,
            max_q_len=queries.max_length,
            max_kv_len=keys_layout.max_length,
            batch_size=len(keys_layout.lengths),
            window_left=-1,
            enable_pdl=False,
            bmm1_scale=q.shape[-1] ** -0.5 if scale is None else scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0 if causal else -1.0,
            is_causal=causal,
            return_lse=return_lse,
        )
        return result if return_lse else output

    def prefill_batch(
        self,
        q,
        latent,
        key,
        weight,
        q_positions,
        kv_positions,
        *,
        nope_dim,
        value_dim,
        scale=None,
        return_lse=False,
        causal_layout=None,
        metadata,
        cache,
        layer_id,
        cp,
    ):
        """One suffix launch per batch, then bounded noncausal prefix chunks."""
        if (
            not q.is_cuda
            or q.dtype != torch.bfloat16
            or len(q) == 0
            or (q.shape[-1], value_dim) not in ((128, 128), (192, 128), (256, 256))
        ):
            return None
        from tvm_ffi import use_torch_stream

        from . import MLAOutput

        has_prefix = bool(metadata.prefix_chunks)
        with use_torch_stream():
            result = self._run_ragged_prefill(
                q,
                latent,
                key,
                weight,
                metadata.queries,
                metadata.queries,
                nope_dim=nope_dim,
                value_dim=value_dim,
                scale=scale,
                causal=True,
                return_lse=has_prefix,
            )
            if not has_prefix:
                return MLAOutput(result)
            from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
            from sglang.srt.layers.attention.merge_state import merge_state

            output, lse = result
            for chunk in metadata.prefix_chunks:
                prefix_latent, prefix_key = chunk.gather(cache, layer_id, cp, q.dtype)
                prefix_output, prefix_lse = self._run_ragged_prefill(
                    q,
                    prefix_latent,
                    prefix_key,
                    weight,
                    metadata.queries,
                    chunk.layout,
                    nope_dim=nope_dim,
                    value_dim=value_dim,
                    scale=scale,
                    causal=False,
                    return_lse=True,
                )
                if chunk.has_empty:
                    fixup_zero_kv_rows(
                        prefix_output,
                        prefix_lse,
                        chunk.layout.lengths,
                        metadata.queries.cu_lengths,
                        metadata.queries.max_length,
                    )
                merged_output, merged_lse = torch.empty_like(output), torch.empty_like(lse)
                merge_state(prefix_output, prefix_lse, output, lse, merged_output, merged_lse)
                output, lse = merged_output, merged_lse
        return MLAOutput(output)

    def prefill(self, q, latent, key, weight, *, nope_dim, value_dim, scale=None):
        """The same native ragged call for an unsupported batched-layout fallback."""
        from tvm_ffi import use_torch_stream

        from .prefill import RaggedLayout

        queries = RaggedLayout.from_lengths((len(q),), q.device)
        keys = RaggedLayout.from_lengths((len(latent),), q.device)
        with use_torch_stream():
            return self._run_ragged_prefill(
                q,
                latent,
                key,
                weight,
                queries,
                keys,
                nope_dim=nope_dim,
                value_dim=value_dim,
                scale=scale,
                causal=True,
                return_lse=False,
            )

    def __call__(
        self,
        q,
        latent,
        key,
        weight,
        q_positions,
        kv_positions,
        *,
        nope_dim,
        value_dim,
        scale=None,
        return_lse=False,
        causal_layout: CausalLayout | None = None,
    ):
        if (
            not return_lse
            and causal_layout is not None
            and causal_layout.key_start == 0
            and causal_layout.key_stride == 1
            and causal_layout.query_start + len(q) == len(latent)
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and (q.shape[-1], value_dim) in ((128, 128), (192, 128), (256, 256))
            and len(q) > 0
        ):
            return self.prefill(
                q, latent, key, weight, nope_dim=nope_dim, value_dim=value_dim, scale=scale
            )
        return mla_attention(
            q,
            latent,
            key,
            weight,
            q_positions,
            kv_positions,
            nope_dim=nope_dim,
            value_dim=value_dim,
            scale=scale,
            return_lse=return_lse,
            causal_layout=causal_layout,
        )
