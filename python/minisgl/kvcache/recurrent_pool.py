"""Request-owned recurrent and convolution state; stale releases cannot free a new owner."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from math import prod
from typing import NoReturn

import torch


@dataclass(frozen=True)
class StateCheckpoint:
    slot: int
    length: int


@dataclass(frozen=True)
class CheckpointWrite:
    """A pending write lease; its origin never restricts later prefix reuse."""

    checkpoint: StateCheckpoint
    table_idx: int
    uid: int
    start_len: int
    end_len: int

    @property
    def generation(self) -> int:
        # Checkpoint slots are monotonic, including after request-slot reuse.
        return self.checkpoint.slot


@dataclass
class _PendingCheckpoint:
    plan: CheckpointWrite
    written: set[int] = field(default_factory=set)
    borrowed: set[int] = field(default_factory=set)
    sealed: bool = False


class RecurrentStatePool:
    def __init__(
        self,
        layer_ids,
        capacity,
        heads,
        dim,
        conv_kernel_size,
        dtype,
        device,
        state_slot_capacity=None,
        arena=None,
        tracking_buffers=0,
        *,
        state_dtype=torch.float32,
        device_abort: Callable[[BaseException], NoReturn] | None = None,
    ):
        if state_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Recurrent state storage supports float32 or bfloat16")
        if capacity < 1 or tracking_buffers not in (0, 1, 2):
            raise ValueError("Invalid recurrent request or tracking capacity")
        self._layers = {layer: i for i, layer in enumerate(layer_ids)}
        self.capacity = capacity  # request-table capacity, independent of physical slots
        self.slot_capacity = capacity if state_slot_capacity is None else state_slot_capacity
        if self.slot_capacity < 1:
            raise ValueError("Recurrent physical slot capacity must be positive")
        self.checkpoint_enabled = bool(self._layers) and state_slot_capacity is not None
        self.tracking_buffers = tracking_buffers if self.checkpoint_enabled else 0
        self.device_abort, self.arena = device_abort, arena
        self._recurrent_shape = (len(self._layers), heads, dim, dim)
        self._conv_shape = (len(self._layers), 3, heads * dim, conv_kernel_size - 1)
        self._conv_dtype, self._state_dtype = dtype, state_dtype
        # One physical slot axis for working, tracking and immutable checkpoints.
        # Logical views retain the existing kernel-facing channel/history and K,V order.
        self._recurrent = (
            None
            if arena is not None
            else torch.empty(
                (len(self._layers), self.slot_capacity, heads, dim, dim),
                dtype=state_dtype,
                device=device,
            ).transpose(-1, -2)
        )
        self._conv = (
            None
            if arena is not None
            else torch.empty(
                (len(self._layers), self.slot_capacity, conv_kernel_size - 1, 3, heads * dim),
                dtype=dtype,
                device=device,
            ).permute(0, 1, 3, 4, 2)
        )
        self._owners = [None] * capacity
        self._processed_lens = [0] * capacity
        self._active = [False] * capacity
        self._working_slots = [None] * capacity
        self._tracking_slots = [[] for _ in range(capacity)]
        self._free_state_slots = list(reversed(range(self.slot_capacity)))
        self._checkpoints = {}
        self._checkpoint_lengths = {}
        self._next_checkpoint = 0
        self._staged = set()
        self._pending = {}
        self._checkpoint_storage = {}
        self._request_checkpoints = {}
        self._deferred_checkpoint_storage = {}
        self._arena_deferred_slots = set()

    def _state_key(self, slot):
        return ("state_slot", self, slot)

    def _slot_views(self, slot):
        if self.arena is not None:
            return self._views(self._state_key(slot))
        return self._conv[:, slot], self._recurrent[:, slot]

    def _drain_arena_releases(self):
        # The arena owns physical deferral until its last global borrower exits.
        # Reconcile metadata only after it has actually removed the common key.
        for slot in tuple(self._arena_deferred_slots):
            if not self.arena.has_state(self._state_key(slot)):
                self._arena_deferred_slots.remove(slot)
                self._free_state_slots.append(slot)

    def _allocate_slots(self, count):
        self._drain_arena_releases()
        if len(self._free_state_slots) < count:
            raise MemoryError("Recurrent shared state slots exhausted")
        selected = list(reversed(self._free_state_slots[-count:])) if count else []
        allocated = []
        try:
            if self.arena is not None:
                self.arena._check_mutable()
                for slot in selected:
                    self.arena.allocate_state(self._state_key(slot))
                    allocated.append(slot)
        except MemoryError:
            # These are the last appended arena rows; reverse rollback moves no owner.
            for slot in reversed(allocated):
                self.arena.free_state(self._state_key(slot))
            raise
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            raise
        for slot in selected:
            assert self._free_state_slots.pop() == slot
        return selected

    def _release_slot(self, slot):
        if slot in self._free_state_slots or slot in self._arena_deferred_slots:
            raise RuntimeError("Recurrent physical slot released twice")
        if self.arena is not None and self.arena.has_state(self._state_key(slot)):
            try:
                if self.arena._view_borrows:
                    self.arena._free_state_after_borrow(self._state_key(slot))
                    self._arena_deferred_slots.add(slot)
                    return
                self.arena.free_state(self._state_key(slot))
            except BaseException as error:
                if self.device_abort is not None:
                    self.fail_device(error)
                raise
        self._free_state_slots.append(slot)

    def _reserved_tracking(self, slot):
        return any(slot in slots for slots in self._tracking_slots)

    def _release_unreserved(self, slot):
        if not self._reserved_tracking(slot):
            self._release_slot(slot)

    def request_slot_demand(self, table_idx, uid):
        self._slot(table_idx)
        if self._owners[table_idx] == uid:
            return 0
        if self._owners[table_idx] is not None:
            raise RuntimeError("Recurrent request table belongs to another owner")
        return 1 + self.tracking_buffers

    def _allocate_request_slots(self, table_idx, uid):
        need = self.request_slot_demand(table_idx, uid)
        if not need:
            return False
        slots = self._allocate_slots(need)  # atomic working + both overlap destinations
        self._working_slots[table_idx] = slots[0]
        self._tracking_slots[table_idx] = slots[1:]
        return True

    def _discard_new_request_slots(self, table_idx):
        slots = [self._working_slots[table_idx], *self._tracking_slots[table_idx]]
        self._working_slots[table_idx] = None
        self._tracking_slots[table_idx] = []
        # These rows were appended together. Reverse rollback does not move
        # another arena owner and performs no device tensor cleanup.
        for slot in reversed(slots):
            self._release_slot(slot)

    def _new_checkpoint(self, table_idx, uid, length, staged):
        busy = set(self._checkpoint_storage.values()) | set(
            self._deferred_checkpoint_storage.values()
        )
        physical = next(
            (slot for slot in self._tracking_slots[table_idx] if slot not in busy), None
        )
        if physical is None:
            physical = self._allocate_slots(1)[0]
        try:
            views = None if self.arena is not None else self._slot_views(physical)
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            if physical not in self._tracking_slots[table_idx]:
                self._release_slot(physical)
            raise
        generation = self._next_checkpoint
        self._next_checkpoint += 1
        checkpoint = StateCheckpoint(generation, length)
        self._checkpoint_storage[generation] = physical
        self._checkpoint_lengths[generation] = length
        self._checkpoints[generation] = views
        self._request_checkpoints[generation] = (table_idx, uid)
        if staged:
            self._staged.add(generation)
        return checkpoint

    def _validate_checkpoint(self, checkpoint):
        if self._checkpoint_lengths.get(checkpoint.slot) != checkpoint.length:
            raise RuntimeError("Recurrent checkpoint is stale, released, or has another frontier")

    @property
    def state_usage(self):
        self._drain_arena_releases()
        working = {slot for slot in self._working_slots if slot is not None}
        tracking = {slot for slots in self._tracking_slots for slot in slots}
        private = {self._checkpoint_storage[g] for g in self._request_checkpoints}
        tree = {
            slot
            for g, slot in self._checkpoint_storage.items()
            if g not in self._request_checkpoints
        }
        deferred = set(self._deferred_checkpoint_storage.values()) | self._arena_deferred_slots
        deferred -= working | tracking | private | tree
        private -= tracking
        owned = working | tracking | private | tree | deferred
        free = set(self._free_state_slots)
        if owned & free or len(owned) + len(free) != self.slot_capacity:
            raise RuntimeError("Recurrent shared slot ownership ledger is inconsistent")
        return {
            "total": self.slot_capacity,
            "free": len(free),
            "working": len(working),
            "tracking": len(tracking),
            "private_checkpoint": len(private),
            "tree_checkpoint": len(tree),
            "deferred_release": len(deferred),
            "pending_writes": len(self._pending),
        }

    @property
    def available_checkpoint_slots(self):
        self._drain_arena_releases()
        return len(self._free_state_slots)

    @property
    def available_tracking_slots(self):
        self._drain_arena_releases()
        busy = set(self._checkpoint_storage.values()) | set(
            self._deferred_checkpoint_storage.values()
        )
        return len(self._free_state_slots) + sum(
            slot not in busy for slots in self._tracking_slots for slot in slots
        )

    @property
    def supports_checkpoint_overlap(self):
        return (
            self.arena is None
            and self.checkpoint_enabled
            and self.tracking_buffers == 2
            and self.device_abort is not None
        )

    @property
    def allocated_bytes(self):
        if self.arena is not None:
            return 0
        return sum(t.numel() * t.element_size() for t in (self._conv, self._recurrent))

    @property
    def conv_dtype(self):
        return self._conv_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    def _views(self, key):
        return self._decode_views(self.arena.state_view(key))

    def _decode_views(self, data):
        recurrent_bytes = prod(self._recurrent_shape) * self.state_dtype.itemsize
        # Public state remains K,V; native recurrent kernels consume its V,K view.
        recurrent = data[:recurrent_bytes].view(self.state_dtype).view(self._recurrent_shape)
        recurrent = recurrent.transpose(-1, -2)
        conv_bytes = prod(self._conv_shape) * self.conv_dtype.itemsize
        # Both slot ends are aligned, including mixed dtypes and empty history.
        conv = data[data.numel() - conv_bytes :].view(self.conv_dtype)
        layers, parts, channels, history = self._conv_shape
        conv = conv.view(layers, history, parts, channels).permute(0, 2, 3, 1)
        return conv, recurrent

    def _working(self, table_idx):
        physical = self._working_slots[table_idx]
        if physical is None:
            raise RuntimeError("Request has no working state lease")
        return self._slot_views(physical)

    def reserve_request(self, table_idx, uid, *, evict=None):
        """Reserve working state, initialize it, and then publish ownership."""
        self._slot(table_idx)
        if self._owners[table_idx] == uid:
            return
        while True:
            try:
                self._allocate_request_slots(table_idx, uid)
                break
            except MemoryError:
                if evict is None or not evict():
                    raise
        try:
            conv, recurrent = self._working(table_idx)
            recurrent.zero_()
            conv.zero_()
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            self._discard_new_request_slots(table_idx)
            raise
        self._owners[table_idx] = uid
        self._processed_lens[table_idx] = 0

    def fail_device(self, error: BaseException) -> NoReturn:
        """A borrowed worker owner must exit before resource contexts unwind."""
        if self.device_abort is None:
            raise error
        self.device_abort(error)

    def reserve_checkpoint(self, table_idx, uid, length, *, end_len, staged=False):
        self._slot(table_idx)
        if not self.checkpoint_enabled or self._owners[table_idx] is None:
            return None
        if self._owners[table_idx] != uid or self._active[table_idx]:
            raise RuntimeError("Checkpoint reservation requires an idle request owner")
        start = self._processed_lens[table_idx]
        if not start < length <= end_len:
            raise ValueError("Checkpoint frontier must be inside the next forward")
        try:
            checkpoint = self._new_checkpoint(table_idx, uid, length, staged)
        except MemoryError:
            return None
        plan = CheckpointWrite(checkpoint, table_idx, uid, start, end_len)
        self._pending[checkpoint.slot] = _PendingCheckpoint(plan)
        return plan

    def _pending_checkpoint(self, plan):
        pending = self._pending.get(plan.generation)
        if pending is None or pending.plan != plan:
            raise RuntimeError("Checkpoint write lease is stale or has another owner")
        if self._owners[plan.table_idx] != plan.uid:
            raise RuntimeError("Checkpoint request owner changed")
        return pending

    def validate_checkpoint_write(self, plan):
        """Validate a fresh lease before preparing native tracking metadata."""
        pending = self._pending_checkpoint(plan)
        if pending.sealed or pending.written or pending.borrowed:
            raise ValueError("Checkpoint preparation requires an unwritten lease")
        if self._processed_lens[plan.table_idx] != plan.start_len:
            raise ValueError("Checkpoint preparation requires its original forward start")

    @contextmanager
    def borrow_checkpoint(self, plan, layer_id):
        """Borrow one layer's paired conv/SSM destination for a successful write."""
        pending = self._pending_checkpoint(plan)
        if pending.sealed or not self._active[plan.table_idx]:
            raise RuntimeError("Checkpoint writes require an active unsealed forward")
        if self._processed_lens[plan.table_idx] != plan.start_len:
            raise RuntimeError("Checkpoint belongs to another forward frontier")
        if layer_id not in self._layers:
            raise KeyError(layer_id)
        if layer_id in pending.written or layer_id in pending.borrowed:
            raise RuntimeError("Checkpoint layer was already borrowed or written")
        saved = self._checkpoints[plan.generation]
        views = (
            nullcontext(saved)
            if self.arena is None
            else self.arena.borrow_state(self._state_key(self._checkpoint_storage[plan.generation]))
        )
        layer = self._layers[layer_id]
        pending.borrowed.add(layer_id)
        try:
            with views as data:
                conv, recurrent = data if self.arena is None else self._decode_views(data)
                yield conv[layer], recurrent[layer]
                if self._pending_checkpoint(plan) is not pending:
                    raise RuntimeError("Checkpoint write lease changed during a layer")
                if (
                    not self._active[plan.table_idx]
                    or self._processed_lens[plan.table_idx] != plan.start_len
                ):
                    raise RuntimeError("Checkpoint forward changed during a layer")
                pending.written.add(layer_id)
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            self.abort_checkpoint(plan)
            raise
        finally:
            pending.borrowed.discard(layer_id)
            if not pending.borrowed and plan.generation in self._deferred_checkpoint_storage:
                self._release_unreserved(self._deferred_checkpoint_storage.pop(plan.generation))

    def seal_checkpoint(self, plan):
        """Seal only after the complete model forward and every state layer succeed."""
        pending = self._pending_checkpoint(plan)
        if self._active[plan.table_idx] or self._processed_lens[plan.table_idx] != plan.end_len:
            raise RuntimeError("Checkpoint sealing requires its completed forward")
        if pending.borrowed or pending.written != set(self._layers):
            raise RuntimeError("Checkpoint is missing completed state layers")
        pending.sealed = True
        return plan.checkpoint

    def abort_checkpoint(self, plan):
        """Discard only this caller's pending/sealed lease, never a donated slot."""
        pending = self._pending.get(plan.generation)
        if pending is not None and pending.plan == plan:
            self._forget_checkpoint(plan.checkpoint)

    def snapshot(self, table_idx, uid, *, staged=False, expected_len=None):
        """Copy a committed frontier into a request-owned tracking destination."""
        self._slot(table_idx)
        if self._owners[table_idx] != uid or self._active[table_idx]:
            raise RuntimeError("Snapshot requires committed request state")
        if self._processed_lens[table_idx] <= 0:
            raise ValueError("Cannot snapshot an empty recurrent state")
        if expected_len is not None and self._processed_lens[table_idx] != expected_len:
            raise RuntimeError("Snapshot does not match the completed forward frontier")
        if not self.checkpoint_enabled:
            raise MemoryError("Recurrent checkpoint storage is not configured")
        checkpoint = self._new_checkpoint(table_idx, uid, self._processed_lens[table_idx], staged)
        try:
            conv, recurrent = self._working(table_idx)
            target_conv, target_recurrent = self._slot_views(
                self._checkpoint_storage[checkpoint.slot]
            )
            target_conv.copy_(conv)
            target_recurrent.copy_(recurrent)
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            self.drop_checkpoint(checkpoint)
            raise
        return checkpoint

    def adopt_staged(self, checkpoint, *, finished=False):
        self._validate_checkpoint(checkpoint)
        if self.arena is not None:
            self.arena._check_mutable()
        pending = self._pending.get(checkpoint.slot)
        if pending is not None and (pending.plan.checkpoint != checkpoint or not pending.sealed):
            raise RuntimeError("Cannot publish an unsealed checkpoint")
        owner = self._request_checkpoints.get(checkpoint.slot)
        if owner is not None:
            table_idx, uid = owner
            if self._owners[table_idx] != uid:
                raise RuntimeError("Checkpoint request owner changed before donation")
            physical = self._checkpoint_storage[checkpoint.slot]
            tracks = self._tracking_slots[table_idx]
            if physical in tracks:
                index = tracks.index(physical)
                if finished:
                    tracks.pop(index)
                else:
                    # Allocate before changing ANY donor owner or generation.
                    replacement = self._allocate_slots(1)[0]
                    tracks[index] = replacement
            self._request_checkpoints.pop(checkpoint.slot)
        self._pending.pop(checkpoint.slot, None)
        self._staged.discard(checkpoint.slot)

    def release_staged(self, checkpoint):
        if (
            checkpoint.slot in self._staged
            or checkpoint.slot in self._pending
            or checkpoint.slot in self._request_checkpoints
        ):
            self.drop_checkpoint(checkpoint)

    def restore(self, table_idx, uid, checkpoint):
        self._slot(table_idx)
        self._validate_checkpoint(checkpoint)
        if self._active[table_idx]:
            raise RuntimeError("Cannot restore during an unfinished forward")
        if self._owners[table_idx] not in (None, uid):
            raise RuntimeError("Cannot restore over another request's working state")
        pending = self._pending.get(checkpoint.slot)
        if pending is not None and not pending.sealed:
            raise RuntimeError("Cannot restore an unsealed checkpoint")
        fresh = self._allocate_request_slots(table_idx, uid)
        try:
            # Arena mutations precede all view acquisition.
            conv, recurrent = self._slot_views(self._checkpoint_storage[checkpoint.slot])
            target_conv, target_recurrent = self._working(table_idx)
            target_conv.copy_(conv)
            target_recurrent.copy_(recurrent)
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            if fresh:
                self._discard_new_request_slots(table_idx)
            raise
        self._owners[table_idx] = uid
        self._processed_lens[table_idx] = checkpoint.length

    def _forget_checkpoint(self, checkpoint):
        self._validate_checkpoint(checkpoint)
        pending = self._pending.get(checkpoint.slot)
        self._checkpoints.pop(checkpoint.slot)
        self._checkpoint_lengths.pop(checkpoint.slot)
        self._request_checkpoints.pop(checkpoint.slot, None)
        self._staged.discard(checkpoint.slot)
        self._pending.pop(checkpoint.slot, None)
        physical = self._checkpoint_storage.pop(checkpoint.slot)
        if pending is not None and pending.borrowed:
            self._deferred_checkpoint_storage[checkpoint.slot] = physical
        else:
            self._release_unreserved(physical)

    def drop_checkpoint(self, checkpoint):
        if checkpoint.slot in self._checkpoints:
            if self.arena is not None:
                self.arena._check_mutable()
            self._forget_checkpoint(checkpoint)

    def _slot(self, table_idx: int) -> None:
        if not 0 <= table_idx < self.capacity:
            raise IndexError("Hybrid request-state slot is out of range")

    def begin_request(self, table_idx, uid, cached_len):
        self._slot(table_idx)
        if cached_len < 0:
            raise ValueError("cached_len cannot be negative")
        if self._active[table_idx]:
            raise RuntimeError("Hybrid request already has an unfinished forward")
        if self._owners[table_idx] != uid:
            if cached_len != 0:
                raise RuntimeError("Hybrid cannot restore recurrent state from a token prefix")
            self.reserve_request(table_idx, uid)
        if self._processed_lens[table_idx] != cached_len:
            raise RuntimeError("Hybrid recurrent state does not match request cached_len")
        self._active[table_idx] = True

    def finish_request(self, table_idx: int, uid: int, processed_len: int) -> None:
        self._slot(table_idx)
        if self._owners[table_idx] != uid or not self._active[table_idx]:
            raise RuntimeError("Hybrid request-state ownership mismatch")
        if processed_len <= self._processed_lens[table_idx]:
            raise ValueError("Hybrid forward must process at least one token")
        self._processed_lens[table_idx] = processed_len
        self._active[table_idx] = False

    def release_request(self, table_idx, uid):
        self._slot(table_idx)
        if self._owners[table_idx] != uid:
            return
        if self.arena is not None:
            self.arena._check_mutable()
        for generation, owner in list(self._request_checkpoints.items()):
            if owner == (table_idx, uid):
                self.drop_checkpoint(
                    StateCheckpoint(generation, self._checkpoint_lengths[generation])
                )
        working = self._working_slots[table_idx]
        if working is not None:
            try:
                conv, recurrent = self._working(table_idx)
                recurrent.zero_()
                conv.zero_()
            except BaseException as error:
                if self.device_abort is not None:
                    self.fail_device(error)
                raise
        tracks = self._tracking_slots[table_idx]
        self._tracking_slots[table_idx] = []
        self._working_slots[table_idx] = None
        deferred = set(self._deferred_checkpoint_storage.values())
        for slot in ([working] if working is not None else []) + tracks:
            if slot not in deferred:
                self._release_slot(slot)
        self._owners[table_idx] = None
        self._processed_lens[table_idx] = 0
        self._active[table_idx] = False

    def reset(self):
        for table_idx, uid in enumerate(self._owners):
            if uid is not None:
                self.release_request(table_idx, uid)
        for generation in list(self._checkpoints):
            self.drop_checkpoint(StateCheckpoint(generation, self._checkpoint_lengths[generation]))

    def get_kda_state(self, layer_id: int, table_idx: int):
        from minisgl.attention.kda import KDAState

        self._slot(table_idx)
        if self._owners[table_idx] is None or not self._active[table_idx]:
            raise RuntimeError("begin_request must precede KDA state access")
        layer = self._layers[layer_id]
        conv, recurrent = self._working(table_idx)
        return KDAState(conv=conv[layer], recurrent=recurrent[layer])

    def set_kda_state(self, layer_id: int, table_idx: int, state) -> None:
        target = self.get_kda_state(layer_id, table_idx)
        if state.conv.shape != target.conv.shape or state.recurrent.shape != target.recurrent.shape:
            raise ValueError("KDA state shape mismatch")
        target.conv.copy_(state.conv)
        target.recurrent.copy_(state.recurrent)

    def batch_indices(self, table_indices):
        if len(set(table_indices)) != len(table_indices):
            raise ValueError("Recurrent batch cannot update one request twice")
        physical = []
        for index in table_indices:
            self._slot(index)
            if self._owners[index] is None or not self._active[index]:
                raise RuntimeError("begin_request must precede batched state access")
            slot = self._working_slots[index]
            physical.append(
                self.arena.state_index(self._state_key(slot)) if self.arena is not None else slot
            )
        device = self.arena.buffer.device if self.arena is not None else self._conv.device
        return torch.tensor(physical, dtype=torch.long, device=device)

    @property
    def is_idle(self):
        return all(owner is None for owner in self._owners)

    def _batched_layer(self, layer_id):
        layer = self._layers[layer_id]
        if self.arena is None:
            return self._conv[layer], self._recurrent[layer]
        # The row stride includes alignment and both kinds of state. Physical
        # rows may move, but the arena base and these views remain stable.
        rows = self.arena.buffer.numel() // self.arena.state_bytes
        views = self._decode_views(self.arena.buffer[: self.arena.state_bytes])
        return tuple(
            part.as_strided(
                (rows, *part.shape),
                (self.arena.state_bytes // part.element_size(), *part.stride()),
            )[:, layer]
            for part in views
        )

    def indexed_kda_state(self, layer_id):
        """Writable pool views for kernels using active request indices.

        Convolution is [slots, channels, history], with contiguous channels;
        recurrence is [slots, heads, V, K]. Neither view gathers or copies state.
        Owners must first be resolved by ``batch_indices`` for this forward.
        """
        conv, recurrent = self._batched_layer(layer_id)
        return conv.flatten(1, 2), recurrent.transpose(-1, -2)

    def get_kda_batch(self, layer_id, indices):
        from minisgl.attention.kda import KDAState

        return KDAState(*(part.index_select(0, indices) for part in self._batched_layer(layer_id)))

    def set_kda_batch(self, layer_id, indices, state):
        for target, value in zip(self._batched_layer(layer_id), state):
            target.index_copy_(0, indices, value.to(dtype=target.dtype))
