"""Absorbed MLA with ordinary or interleaved causal query subsequences."""

import torch
import torch.nn.functional as F

from .mla import CausalLayout, MLAPagedMetadata, causal_query_groups, mla_attention


class _MLAPlan:
    def __init__(self, device, *, batch_size=None, max_pages=None):
        from flashinfer import BatchMLAPagedAttentionWrapper

        # Keep a native plan and all of its workspace under one owner.
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
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

    def prepare(self, metadata, options, *, qo_indptr=None):
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
                    (
                        torch.arange(len(metadata.lengths) + 1, dtype=torch.int32)
                        if qo_indptr is None
                        else qo_indptr
                    ),
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


class FlashInferMLA:
    def __init__(self):
        self._prefill_plan = None
        self._decode_plan = None
        self._graph_plans = {}

    def init_graphs(self, batch_sizes, max_pages, device):
        self._graph_plans = {
            size: _MLAPlan(device, batch_size=size, max_pages=max_pages) for size in batch_sizes
        }

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
        return_lse=False,
    ):
        """Attend a whole decode batch directly over the cache's physical pages.

        The wrapper is replanned once per batch and shared by compatible layers.
        Prefill and unsupported dtypes/dimensions retain the portable path.
        """
        latent, key = pages
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
        if metadata.capture_size is not None:
            plan = self._graph_plans[metadata.capture_size]
        else:
            if self._decode_plan is None or self._decode_plan.device != q.device:
                self._decode_plan = _MLAPlan(q.device)
            plan = self._decode_plan
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
        weights = weight.view(heads, nope_dim + value_dim, 512)
        query = torch.einsum("thd,hdl->thl", q[..., :nope_dim], weights[:, :nope_dim])
        query = F.pad(query, (0, 0, 0, padded_heads - heads))
        suffix = F.pad(q[..., nope_dim:], (0, 0, 0, padded_heads - heads)).contiguous()
        output, lse = plan.run(
            query=(query, suffix),
            kv_cache=pages,
            return_lse=True,
            return_lse_base_on_e=True,
        )
        output = torch.einsum("thl,hvl->thv", output[:, :heads], weights[:, nope_dim:])
        lse = lse[:, :heads]
        if metadata.nonempty is not None:
            output = output.masked_fill(~metadata.nonempty[:, None, None], 0)
            lse = lse.masked_fill(~metadata.nonempty[:, None], -torch.inf)
        return (output.float(), lse) if return_lse else output

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
            not q.is_cuda
            or q.dtype != torch.bfloat16
            or latent.shape[1] != 512
            or key.shape[1] != 64
            or not len(latent)
        ):
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
            )
        tokens = q.shape[0]
        heads, dim = q.shape[1:]
        if causal_layout is None:
            start, first_key = int(q_positions[0]), int(kv_positions[0])
            stride = int(kv_positions[1] - kv_positions[0]) if len(latent) > 1 else 1
            regular = (
                stride > 0
                and torch.equal(q_positions, torch.arange(start, start + tokens, device=q.device))
                and torch.equal(
                    kv_positions, torch.arange(len(latent), device=q.device) * stride + first_key
                )
            )
        else:
            start, first_key, stride = causal_layout
            if stride < 1:
                raise ValueError("Causal key stride must be positive")
            regular = True
        last_key = first_key + (len(latent) - 1) * stride
        if tokens == 1 and regular and last_key <= start:
            groups, order = [([0], len(latent))], None
        else:
            complete = tokens == 1 or last_key >= start + tokens - stride
            if not regular or not complete:
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
                )
            groups = causal_query_groups(start, tokens, first_key, len(latent), stride)
            if not groups:
                output = q.new_zeros((tokens, heads, value_dim))
                lse = torch.full((tokens, heads), -torch.inf, device=q.device)
                return (output.float(), lse) if return_lse else output
            rows = [row for indices, _ in groups for row in indices]
            order = None if rows == list(range(tokens)) else rows

        padded_heads = max(16, 1 << (heads - 1).bit_length())
        softmax_scale = dim**-0.5 if scale is None else scale
        if self._prefill_plan is None or self._prefill_plan.device != q.device:
            # Prefill and decode keep separate scratch/plan lifetimes. Replan
            # variable query groups instead of retaining an LRU of native plans.
            self._prefill_plan = _MLAPlan(q.device)
        plan = self._prefill_plan
        qo, kv = [0], [0]
        for rows, visible in groups:
            qo.append(qo[-1] + len(rows))
            kv.append(kv[-1] + visible)
        metadata = MLAPagedMetadata(
            torch.tensor(kv, dtype=torch.int32),
            torch.cat(
                [torch.arange(visible, device=q.device, dtype=torch.int32) for _, visible in groups]
            ),
            torch.tensor([visible for _, visible in groups], dtype=torch.int32),
            1,
        )
        plan.prepare(
            metadata,
            {
                "num_heads": padded_heads,
                "head_dim_ckv": 512,
                "head_dim_kpe": 64,
                "page_size": 1,
                "causal": tokens > 1,
                "sm_scale": softmax_scale,
                "q_data_type": q.dtype,
                "kv_data_type": latent.dtype,
                "query_layout": "split",
                "kv_cache_layout": "split",
                "lse_mode": "basee",
            },
            qo_indptr=torch.tensor(qo, dtype=torch.int32),
        )
        order = None if order is None else torch.tensor(order, device=q.device)
        weights = weight.view(heads, nope_dim + value_dim, 512)
        if order is not None:
            q = q.index_select(0, order)
        query = torch.einsum("thd,hdl->thl", q[..., :nope_dim], weights[:, :nope_dim])
        suffix = q[..., nope_dim:]
        query = F.pad(query, (0, 0, 0, padded_heads - heads))
        suffix = F.pad(suffix, (0, 0, 0, padded_heads - heads)).contiguous()
        data = torch.cat((latent, key), -1).view(-1, 1, 576)
        output, lse = plan.run(
            query=(query, suffix),
            kv_cache=(data[..., :512], data[..., 512:]),
            return_lse=True,
            return_lse_base_on_e=True,
        )
        output = torch.einsum("thl,hvl->thv", output[:, :heads], weights[:, nope_dim:])
        lse = lse[:, :heads]
        if order is not None:
            full_output = output.new_zeros((tokens, heads, value_dim))
            full_lse = lse.new_full((tokens, heads), -torch.inf)
            full_output.index_copy_(0, order, output)
            full_lse.index_copy_(0, order, lse)
            output, lse = full_output, full_lse
        return (output.float(), lse) if return_lse else output
