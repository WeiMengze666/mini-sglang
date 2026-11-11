from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import CLIP_MAX_NEW_TOKENS, PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    chunk_alignment: int = 1
    chunk_budget: int | None = None
    compute_page_size: int = 1

    def _chunk_size(self, req: PendingReq, handle: BaseCacheHandle, cached_len: int) -> int:
        budget = self.token_budget if self.chunk_budget is None else self.chunk_budget
        remaining = req.input_len - cached_len
        page = self.compute_page_size
        if self.chunk_budget is not None and -(-remaining // page) * page > budget:
            end = cached_len + budget // page * page
        else:
            end = min(cached_len + budget, req.input_len)
        replay_until = getattr(handle, "replay_until", 0)
        if replay_until > cached_len:
            end = min(end, replay_until)
        if end < req.input_len:
            end = end // self.chunk_alignment * self.chunk_alignment
        return max(0, end - cached_len)

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        if self._chunk_size(req, handle, cached_len) == 0:
            return None
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + min(req.output_len, CLIP_MAX_NEW_TOKENS)

        if estimated_len + self.reserved_size >= self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size >= self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        try:
            self.cache_manager.restore_state(table_idx, req.uid, handle)
        except MemoryError:
            self.table_manager.free(table_idx)
            self.cache_manager.unlock(handle)
            return None
        if estimated_len + self.reserved_size >= self.cache_manager.available_size:
            self.cache_manager.release_state(table_idx, req.uid)
            self.table_manager.free(table_idx)
            self.cache_manager.unlock(handle)
            return None
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req | None:
        remain_len = pending_req.input_len - cached_len
        chunk_size = self._chunk_size(pending_req, cache_handle, cached_len)
        if chunk_size == 0:
            return None
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        charge = -(-chunk_size // self.compute_page_size) * self.compute_page_size
        self.token_budget -= charge
        if self.chunk_budget is not None:
            self.chunk_budget -= charge
        # Reserve the actual new KV pages of this forward. Unfinished chunks
        # cannot decode yet; their future output is billed when their final chunk runs.
        page = getattr(self.cache_manager, "page_size", self.compute_page_size)
        first_page = (cached_len + page - 1) // page
        last_page = (cached_len + chunk_size + page - 1) // page
        new_pages = last_page - first_page
        future = 0 if is_chunked else min(pending_req.output_len, CLIP_MAX_NEW_TOKENS)
        self.reserved_size += new_pages * page + future
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        cached_prompt_tokens = getattr(pending_req, "cached_prompt_tokens", None)
        if cached_prompt_tokens is None:
            cached_prompt_tokens = (
                pending_req.chunked_req.cached_prompt_tokens
                if pending_req.chunked_req is not None
                else cached_len
            )
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            cached_prompt_tokens=cached_prompt_tokens,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0 or (
            self.chunk_budget is not None and self.chunk_budget <= 0
        ):
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=self.cache_manager.current_handle(chunked_req),
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    chunked_prefill_size: int | None = None
    compute_page_size: int | None = None

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def requeue(self, req: Req) -> None:
        """Replay complete host history after a completed-boundary decode retraction."""
        remaining = req.max_device_len - len(req.input_ids)
        if remaining <= 0:
            raise ValueError("Only unfinished requests can be requeued")
        self.pending_list.append(
            PendingReq(
                req.uid,
                req.input_ids,
                replace(req.sampling_params, max_tokens=remaining),
                cached_prompt_tokens=req.cached_prompt_tokens,
            )
        )

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        page_size = self.compute_page_size or self.cache_manager.page_size
        chunk_budget = (
            prefill_budget if self.chunked_prefill_size is None else self.chunked_prefill_size
        )
        adder = PrefillAdder(
            token_budget=prefill_budget,
            chunk_budget=chunk_budget,
            compute_page_size=page_size,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            # Keep intermediate frontiers page-aligned for KV and state reuse.
            # A configured budget below one page must still make progress.
            chunk_alignment=(
                page_size if chunk_budget >= page_size else 1
            ),
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
