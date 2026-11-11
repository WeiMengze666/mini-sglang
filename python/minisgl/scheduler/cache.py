from __future__ import annotations

from contextlib import contextmanager
from math import lcm
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        pool=None,
        *,
        cpu_group=None,
        checkpoint_quantum=None,
    ):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.pool = pool
        self.arena = getattr(pool, "arena", None)
        self._handles = {}
        self._tracking = {}
        states = getattr(pool, "recurrent", None)
        self.tracks_decode = states is not None and bool(states.max_staged_states)
        self._tracking_slots = 2 if getattr(pool, "supports_overlap", False) else 1
        self._checkpoint_cpu_group = cpu_group
        self._checkpoint_quantum = checkpoint_quantum

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    def restore_state(self, table_idx: int, uid: int, handle: BaseCacheHandle) -> None:
        self.prefix_cache.restore_state(table_idx, uid, handle)
        if getattr(self.pool, "recurrent", None) is not None and handle.cached_len == 0:
            if self.arena is not None:
                self.prefix_cache.release_checkpoint_memory(self.arena.state_bytes)
            self.pool.recurrent.reserve_request(table_idx, uid)
        self._handles[table_idx, uid] = handle

    def release_state(self, table_idx, uid):
        if self.arena is not None:
            self.arena._check_mutable()
        self._handles.pop((table_idx, uid), None)
        for checkpoint in self._tracking.pop((table_idx, uid), ()):
            self.pool.release_staged(checkpoint)
        if self.pool is not None and hasattr(self.pool, "release_request"):
            self.pool.release_request(table_idx, uid)

    def current_handle(self, req):
        # Chunked prefill may already have built its next Req while the previous
        # result updates the shared prefix. The manager owns the latest lease.
        return self._handles.get((req.table_idx, req.uid), req.cache_handle)

    def should_cache(self, is_prefill, length):
        interval = self.prefix_cache.checkpoint_interval
        return is_prefill or (interval is not None and length % interval == 0)

    def coordinates_checkpoints(self, batch):
        return self._checkpoint_quantum is not None and batch.is_prefill

    def _checkpoint_exchange(self, value):
        gathered = [value]
        if self._checkpoint_cpu_group is not None:
            gathered = [None] * torch.distributed.get_world_size(self._checkpoint_cpu_group)
            torch.distributed.all_gather_object(gathered, value, group=self._checkpoint_cpu_group)
        return gathered

    def plan_checkpoints(self, batch, backend):
        """Reserve optional local leases, then intersect full-TP logical row masks."""
        coordinated = self.coordinates_checkpoints(batch)
        old_plans, plans, signature, error = tuple(batch.checkpoint_plan), [], [], None
        try:
            if old_plans:
                raise RuntimeError("Batch already owns a checkpoint plan")
            capability = getattr(backend, "checkpoint_quantum", None)
            quantum = capability(batch) if capability is not None else None
            states = getattr(self.pool, "recurrent", None)
            if not coordinated and (not batch.is_prefill or quantum is None or states is None):
                return ()
            grid_quantum = self._checkpoint_quantum if coordinated else quantum
            if type(grid_quantum) is not int or grid_quantum <= 0:
                raise ValueError("Checkpoint quantum must be positive")
            grid = lcm(self.page_size, grid_quantum)
            for req in batch.reqs:
                frontier = req.device_len // grid * grid
                fork = getattr(self.current_handle(req), "replay_until", 0)
                if req.cached_len < fork < frontier and fork % grid == 0:
                    if (fork - req.cached_len) % grid_quantum == 0:
                        frontier = fork
                signature.append((req.table_idx, req.uid, req.cached_len, req.device_len, frontier))
                eligible = (
                    quantum == grid_quantum
                    and states is not None
                    and req.cached_len < frontier < req.device_len
                    and (frontier - req.cached_len) % grid_quantum == 0
                )
                plan = None
                if eligible:
                    staged = bool(states.max_staged_states)
                    plan = states.reserve_checkpoint(
                        req.table_idx, req.uid, frontier, end_len=req.device_len, staged=staged
                    )
                    if (
                        plan is None
                        and not staged
                        and states.available_checkpoint_slots == 0
                        and self.prefix_cache.evict_checkpoint()
                    ):
                        plan = states.reserve_checkpoint(
                            req.table_idx, req.uid, frontier, end_len=req.device_len
                        )
                plans.append(plan)
        except BaseException as exc:
            error = exc
        if coordinated:
            packet = (
                tuple(signature),
                tuple(p is not None for p in plans),
                None if error is None else f"{type(error).__name__}: {error}",
            )
            peers = self._checkpoint_exchange(packet)
            errors = [item[2] for item in peers if item[2] is not None]
            if errors or any(item[0] != packet[0] for item in peers):
                error = RuntimeError(
                    f"Checkpoint plan agreement failed: {errors or 'row signature'}"
                )
            if error is None:
                for row, plan in enumerate(plans):
                    if plan is not None and not all(item[1][row] for item in peers):
                        try:
                            self.release_checkpoint_plan((plan,))
                            plans[row] = None
                        except BaseException as exc:
                            error = exc
                errors = self._checkpoint_exchange(
                    None if error is None else f"{type(error).__name__}: {error}"
                )
                if any(item is not None for item in errors):
                    error = RuntimeError(f"Checkpoint reservation cleanup failed: {errors}")
        if error is not None:
            self.release_checkpoint_plan([p for p in plans if p is not None], error=error)
            if coordinated:
                self.release_checkpoint_plan(old_plans, error=error)
                batch.checkpoint_plan = ()
            raise error
        batch.checkpoint_plan = tuple(p for p in plans if p is not None)
        return batch.checkpoint_plan

    def checkpoint_metadata_ready(self, batch, error=None):
        if self.coordinates_checkpoints(batch):
            errors = self._checkpoint_exchange(
                None if error is None else f"{type(error).__name__}: {error}"
            )
            if any(item is not None for item in errors):
                error = RuntimeError(f"Checkpoint metadata agreement failed: {errors}")
        if error is not None:
            self.release_checkpoint_plan(batch.checkpoint_plan, error=error)
            batch.checkpoint_plan = ()
            raise error

    @contextmanager
    def checkpoint_writes(self, batch, backend):
        """Reserve/abort scope; only HybridCache seals a complete all-layer forward."""
        plans = self.plan_checkpoints(batch, backend)
        try:
            yield plans
        except BaseException as error:
            self.release_checkpoint_plan(plans, error=error)
            raise

    def release_checkpoint_plan(self, plans, *, error=None):
        failure = None
        for plan in plans:
            try:
                self.pool.recurrent.abort_checkpoint(plan)
            except BaseException as exc:
                if error is not None:
                    error.add_note(f"Checkpoint abort failed: {exc}")
                elif failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    @property
    def has_tracking(self):
        return any(self._tracking.values())

    def finished_checkpoint(self, req, length):
        """Name only the latest completed track, or its previous overlap track."""
        tracking = self._tracking.get((req.table_idx, req.uid), ())
        if not tracking:
            return None
        latest = tracking[-1]
        if latest.length <= length:
            return latest
        if len(tracking) == 2 and tracking[-2].length <= length:
            return tracking[-2]
        return None

    def stage_states(self, reqs, lengths, *, is_prefill, excluded=()):
        staged = {}
        if self.pool is None:
            return staged
        try:
            for req, length in zip(reqs, lengths):
                if (
                    req.uid in excluded
                    or length % self.page_size
                    or (not is_prefill and not self.tracks_decode)
                ):
                    continue
                tracking = None
                if not is_prefill:
                    tracking = self._tracking.setdefault((req.table_idx, req.uid), [])
                    if tracking and tracking[-1].length == length:
                        continue
                    # Reuse the bounded private budget before allocating its replacement.
                    if len(tracking) == self._tracking_slots:
                        self.pool.release_staged(tracking[0])
                        tracking.pop(0)
                try:
                    checkpoint = self.pool.stage_state(req.table_idx, req.uid, length)
                except MemoryError:
                    continue  # Optional reuse must not prevent the admitted forward.
                if checkpoint is not None:
                    if tracking is None:
                        staged[req.uid] = checkpoint
                    else:
                        tracking.append(checkpoint)
        except BaseException as error:
            # A failed copy may still own device resources: the worker exits first.
            if staged or self.has_tracking:
                self.pool.recurrent.fail_device(error)
            raise
        return staged

    def release_staged(self, staged):
        for checkpoint in staged.values():
            self.pool.release_staged(checkpoint)

    @property
    def available_size(self) -> int:
        free_pages = len(self.free_slots)
        if self.arena is not None:
            snapshots = self.prefix_cache.reclaimable_state_bytes
            free_pages = min(
                free_pages, (self.arena.free_bytes + snapshots) // self.arena.page_bytes
            )
        return self.prefix_cache.size_info.evictable_size + free_pages * self.page_size

    def lock(self, handle: BaseCacheHandle, *, for_restore: bool = True) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False, for_restore=for_restore)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def reclaim_capacity(self, required_tokens: int) -> bool:
        """Reclaim unpinned cache pages before considering live-request retraction."""
        required_pages = div_ceil(required_tokens, self.page_size)
        if self.arena is not None:
            self.prefix_cache.release_checkpoint_memory(required_pages * self.arena.page_bytes)
        free_pages = len(self.free_slots)
        if self.arena is not None:
            free_pages = min(free_pages, self.arena.free_pages)
        shortage = max(0, required_pages - free_pages) * self.page_size
        reclaim = min(shortage, self.prefix_cache.size_info.evictable_size)
        if reclaim:
            self._free(self.prefix_cache.evict(reclaim))
        free_pages = len(self.free_slots)
        if self.arena is not None:
            free_pages = min(free_pages, self.arena.free_pages)
        return free_pages >= required_pages

    def discard_req(self, req: Req) -> None:
        """Drop private completed KV and state without publishing a new prefix."""
        handle = self._handles.get((req.table_idx, req.uid))
        if handle is None:
            raise RuntimeError("Cannot retract a request without its current cache ownership")
        end = div_ceil(req.cached_len, self.page_size) * self.page_size
        self._free(self.page_table[req.table_idx, handle.cached_len : end])
        self.unlock(handle)
        self.release_state(req.table_idx, req.uid)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool, frontier=None, checkpoint=None) -> None:
        if self.arena is not None:
            self.arena._check_mutable()
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        length = req.cached_len if frontier is None else frontier
        insert_ids = req.input_ids[:length]
        page_indices = self.page_table[req.table_idx, :length]
        old_handle = self.current_handle(req)
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # An explicit old frontier needs its staged state. It cannot be rebuilt
        # from a working slot that the next forward may already have advanced.
        if checkpoint is not None or frontier is None:
            self.prefix_cache.cache_state(req, new_handle, checkpoint=checkpoint, cached_len=length)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len].clone())
        if finished:  # this tail part should be freed
            self._free(self.page_table[req.table_idx, new_handle.cached_len : req.cached_len])
            self._handles.pop((req.table_idx, req.uid), None)
        else:  # keep the tail part, update the handle
            new_handle = self.prefix_cache.continue_handle(old_handle, new_handle, length)
            if new_handle.cached_len:
                # Replay may join an existing radix edge. Point the continuing
                # request at its shared KV pages before freed duplicates are reused.
                page_indices[: new_handle.cached_len].copy_(new_handle.get_matched_indices())
            req.cache_handle = new_handle
            self._handles[req.table_idx, req.uid] = new_handle
            # A continuing request already owns private recurrent state.
            self.lock(new_handle, for_restore=False)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        failed = False
        try:
            self._free = lazy_free
            yield
        except BaseException as error:
            states = getattr(self.pool, "recurrent", None)
            if states is not None and states.device_abort is not None:
                # An owned worker must exit before deferred resources can be recycled.
                failed = True
                states.fail_device(error)
            raise
        finally:
            del self._free
            if not failed:
                if self.arena is not None:
                    for pages in lazy_free_list:
                        self.arena.free_pages_by_id((pages // self.page_size).tolist())
                self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if self.arena is not None:
            self.arena._check_mutable()
        free_pages = len(self.free_slots)
        if self.arena is not None:
            self.prefix_cache.release_checkpoint_memory(needed_pages * self.arena.page_bytes)
            free_pages = min(free_pages, self.arena.free_pages)
        if needed_pages > free_pages:
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            if self.arena is not None:
                self.arena.free_pages_by_id((evicted[:: self.page_size] // self.page_size).tolist())
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        if self.arena is not None:
            self.arena.allocate_pages((allocated // self.page_size).tolist())
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            if self.arena is not None:
                self.arena.free_pages_by_id((indices[:: self.page_size] // self.page_size).tolist())
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
