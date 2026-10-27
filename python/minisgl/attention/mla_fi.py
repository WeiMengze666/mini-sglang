"""Absorbed MLA with ordinary or interleaved causal query subsequences."""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from .mla import CausalLayout, causal_query_groups, mla_attention


class FlashInferMLA:
    def __init__(self):
        self._workspace = None
        self._plans = OrderedDict()

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
            pattern = (1,)
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
            pattern = (tokens, start, first_key, stride)

        from flashinfer import BatchMLAPagedAttentionWrapper
        from flashinfer.mla import MLAPlanMetadata

        padded_heads = max(16, 1 << (heads - 1).bit_length())
        softmax_scale = dim**-0.5 if scale is None else scale
        signature = (q.device, heads, len(latent), softmax_scale, pattern)
        if self._workspace is None or self._workspace.device != q.device:
            self._plans.clear()
            self._workspace = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
        cached = self._plans.pop(signature, None)
        if cached is None:
            plan = BatchMLAPagedAttentionWrapper(self._workspace, backend="fa2")
            qo, kv = [0], [0]
            for rows, visible in groups:
                qo.append(qo[-1] + len(rows))
                kv.append(kv[-1] + visible)
            metadata = MLAPlanMetadata.csr(
                torch.tensor(qo, dtype=torch.int32),
                torch.tensor(kv, dtype=torch.int32),
                torch.cat(
                    [
                        torch.arange(visible, device=q.device, dtype=torch.int32)
                        for _, visible in groups
                    ]
                ),
                torch.tensor([visible for _, visible in groups], dtype=torch.int32),
            )
            plan.plan(
                metadata=metadata,
                num_heads=padded_heads,
                head_dim_ckv=512,
                head_dim_kpe=64,
                page_size=1,
                causal=tokens > 1,
                sm_scale=softmax_scale,
                q_data_type=q.dtype,
                kv_data_type=latent.dtype,
                query_layout="split",
                kv_cache_layout="split",
                lse_mode="basee",
            )
            order = None if order is None else torch.tensor(order, device=q.device)
            cached = (plan, order)
        self._plans[signature] = cached
        plan, order = cached
        if len(self._plans) > 16:
            self._plans.popitem(last=False)
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
