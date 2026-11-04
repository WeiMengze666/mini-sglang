"""Resource adapter for FlashInfer's native cyclic-DCP MLA decode kernel."""

import torch
import torch.nn.functional as F


class CuTeMLADecodePlan:
    def __init__(self, workspace, *, batch_size=None, max_pages=None, page_size=64):
        self.workspace = workspace
        self.device = workspace.device
        self.metadata, self.options = None, None
        self.graph = batch_size is not None
        self.page_size = page_size
        self.max_pages = max_pages
        # FlashInfer loads complete 128-token tiles from the dense page table.
        self.page_alignment = max(1, 128 // page_size)
        if self.graph:
            self.blocks = torch.zeros(
                (batch_size, self._aligned_width(max_pages)), dtype=torch.int32, device=self.device
            )
            self.local = torch.empty(batch_size, dtype=torch.int32, device=self.device)
            self.global_ = torch.empty_like(self.local)

    def _aligned_width(self, pages):
        return (pages + self.page_alignment - 1) // self.page_alignment * self.page_alignment

    def prepare(self, metadata, options):
        if self.metadata is metadata and self.options == options:
            return
        if metadata.page_size != self.page_size:
            raise ValueError("MLA plan page size does not match its cache")
        width = metadata.block_table.shape[1]
        if self.graph:
            if width > self.max_pages:
                raise ValueError("MLA graph page capacity exceeded")
            self.blocks.zero_()
            self.blocks[:, :width].copy_(metadata.block_table)
            self.local.copy_(metadata.local_lengths)
            self.global_.copy_(metadata.global_lengths)
        else:
            padding = self._aligned_width(width) - width
            self.blocks = (
                F.pad(metadata.block_table, (0, padding)) if padding else metadata.block_table
            )
            self.local = metadata.local_lengths
            self.global_ = metadata.global_lengths
        self.metadata, self.options = metadata, options

    def run(self, query, pages, *, nope_dim, scale):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
        from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
        from tvm_ffi import use_torch_stream

        metadata = self.metadata
        # The official NoPE FP8 path concatenates in producer precision, then
        # converts both query components to the KV dtype with unit scale.
        with use_torch_stream():
            query = concat_mla_absorb_q_general(*query).to(pages.dtype)
            output, lse = trtllm_batch_decode_with_kv_cache_mla(
                query=query.unsqueeze(1),
                kv_cache=pages.unsqueeze(1),
                workspace_buffer=self.workspace,
                qk_nope_head_dim=nope_dim,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=self.blocks,
                seq_lens=self.local,
                max_seq_len=(
                    self.max_pages * metadata.page_size
                    if self.graph
                    else max(
                        1,
                        (metadata.max_seq_len + metadata.cp_size - 1 - metadata.cp_rank)
                        // metadata.cp_size,
                    )
                ),
                bmm1_scale=scale,
                backend="cute-dsl",
                enable_dcp=metadata.cp_size > 1,
                cp_world=metadata.cp_size,
                cp_rank=metadata.cp_rank,
                causal_seqlens_kv_global=self.global_ if metadata.cp_size > 1 else None,
                return_lse=True,
                return_lse_base="basee",
            )
        output = output.reshape(len(query), query.shape[1], 512)
        lse = lse.reshape(len(query), query.shape[1])
        # Empty cyclic shards are neutral elements of the distributed merge.
        empty = self.local == 0
        return output.masked_fill(empty[:, None, None], 0), lse.masked_fill(
            empty[:, None], -torch.inf
        )
