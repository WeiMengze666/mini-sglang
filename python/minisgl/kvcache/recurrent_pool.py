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
        max_cached_states=0,
        arena=None,
        max_staged_states=0,
        *,
        state_dtype=torch.float32,
        device_abort: Callable[[BaseException], NoReturn] | None = None,
    ):
        if state_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Recurrent state storage supports float32 or bfloat16")
        self._layers = {layer: i for i, layer in enumerate(layer_ids)}
        self.capacity = capacity
        self.device_abort = device_abort
        self.arena = arena
        self._recurrent_shape = (len(self._layers), heads, dim, dim)
        self._conv_shape = (len(self._layers), 3, heads * dim, conv_kernel_size - 1)
        self._conv_dtype = dtype
        self._state_dtype = state_dtype
        # Logical K,V and channel/history views over native-friendly storage.
        # Kernel adapters can update indexed rows without repacking the pool.
        self._recurrent = (
            None
            if arena is not None
            else torch.zeros(
                (len(self._layers), capacity, heads, dim, dim),
                dtype=state_dtype,
                device=device,
            ).transpose(-1, -2)
        )
        self._conv = (
            None
            if arena is not None
            else torch.zeros(
                (
                    len(self._layers),
                    capacity,
                    conv_kernel_size - 1,
                    3,
                    heads * dim,
                ),
                dtype=dtype,
                device=device,
            ).permute(0, 1, 3, 4, 2)
        )
        self._owners: list[int | None] = [None] * capacity
        self._processed_lens = [0] * capacity
        self._active = [False] * capacity
        self.max_cached_states = max_cached_states
        self._checkpoints: dict[int, tuple[torch.Tensor, torch.Tensor] | None] = {}
        self._next_checkpoint = 0
        self.max_staged_states = max_staged_states
        self._staged: set[int] = set()
        self._pending: dict[int, _PendingCheckpoint] = {}
        # Physical storage is bounded at startup; generation IDs never alias
        # an old lease even when a released physical row is reused.
        slots = max_cached_states + max_staged_states
        self._checkpoint_conv = (
            None
            if arena is not None
            else torch.empty(
                (slots, len(self._layers), conv_kernel_size - 1, 3, heads * dim),
                dtype=dtype,
                device=device,
            ).permute(0, 1, 3, 4, 2)
        )
        self._checkpoint_recurrent = (
            None
            if arena is not None
            else torch.empty(
                (slots, len(self._layers), heads, dim, dim),
                dtype=state_dtype,
                device=device,
            ).transpose(-1, -2)
        )
        self._free_checkpoint_slots = list(reversed(range(slots)))
        self._checkpoint_storage: dict[int, int] = {}
        self._deferred_checkpoint_storage: dict[int, int] = {}

    @property
    def available_checkpoint_slots(self):
        return self.max_cached_states - (len(self._checkpoints) - len(self._staged))

    @property
    def available_tracking_slots(self):
        return self.max_staged_states - len(self._staged)

    @property
    def supports_checkpoint_overlap(self):
        return (
            self.arena is None
            and self.max_staged_states >= 2 * self.capacity
            and self.device_abort is not None
        )

    @property
    def allocated_bytes(self):
        if self.arena is not None:
            return 0  # The shared arena owner accounts for its whole buffer.
        return sum(
            t.numel() * t.element_size()
            for t in (
                self._conv,
                self._recurrent,
                self._checkpoint_conv,
                self._checkpoint_recurrent,
            )
        )

    def _allocate_checkpoint_storage(self, generation):
        if self.arena is not None:
            self.arena.allocate_state(("checkpoint", generation))
            return None
        if not self._free_checkpoint_slots:
            raise MemoryError("Recurrent physical checkpoint capacity exhausted")
        slot = self._free_checkpoint_slots.pop()
        self._checkpoint_storage[generation] = slot
        return self._checkpoint_conv[slot], self._checkpoint_recurrent[slot]

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
        if self.arena is not None:
            return self._views(("request", table_idx))
        return self._conv[:, table_idx], self._recurrent[:, table_idx]

    def _allocate_working(self, table_idx):
        if self.arena is not None and not self.arena.has_state(("request", table_idx)):
            self.arena.allocate_state(("request", table_idx))

    def reserve_request(self, table_idx, uid):
        """Reserve an empty working slot at admission, before budgeting KV pages."""
        self.begin_request(table_idx, uid, 0)
        self._active[table_idx] = False

    def fail_device(self, error: BaseException) -> NoReturn:
        """A borrowed worker owner must exit before resource contexts unwind."""
        if self.device_abort is None:
            raise error
        self.device_abort(error)

    def reserve_checkpoint(self, table_idx, uid, length, *, end_len, staged=False):
        """Reserve a cached or tracking lease without copying working state.

        An unadmitted request is skipped without reserving a working slot.
        Optional reuse may be skipped under pressure. Existing tree checkpoints
        and their restore locks are never evicted by this operation.
        """
        self._slot(table_idx)
        if self._owners[table_idx] is None:
            return None
        if self._owners[table_idx] != uid or self._active[table_idx]:
            raise RuntimeError("Checkpoint reservation requires an idle request owner")
        start = self._processed_lens[table_idx]
        if not start < length <= end_len:
            raise ValueError("Checkpoint frontier must be inside the next forward")
        available = self.available_tracking_slots if staged else self.available_checkpoint_slots
        if available <= 0:
            return None
        if self.arena is not None:
            self.arena._check_mutable()
        slot = self._next_checkpoint
        self._next_checkpoint += 1
        try:
            saved = self._allocate_checkpoint_storage(slot)
        except MemoryError:
            return None
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            raise
        checkpoint = StateCheckpoint(slot, length)
        plan = CheckpointWrite(checkpoint, table_idx, uid, start, end_len)
        self._checkpoints[slot] = saved
        self._pending[slot] = _PendingCheckpoint(plan)
        if staged:
            self._staged.add(slot)
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
            else self.arena.borrow_state(("checkpoint", plan.generation))
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
                self._free_checkpoint_slots.append(
                    self._deferred_checkpoint_storage.pop(plan.generation)
                )

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
            if self.arena is not None:
                self.arena._free_state_after_borrow(("checkpoint", plan.generation))
            self._forget_checkpoint(plan.checkpoint)

    def snapshot(
        self, table_idx: int, uid: int, *, staged=False, expected_len=None
    ) -> StateCheckpoint:
        """Copy committed state on the forward stream into a caller-owned slot.

        The request keeps mutating its working slot. Donate transfers only this
        checkpoint's index to the tree; there is no second state copy.
        """
        self._slot(table_idx)
        if self._owners[table_idx] != uid or self._active[table_idx]:
            raise RuntimeError("Snapshot requires committed request state")
        if self._processed_lens[table_idx] <= 0:
            raise ValueError("Cannot snapshot an empty recurrent state")
        if expected_len is not None and self._processed_lens[table_idx] != expected_len:
            raise RuntimeError("Snapshot does not match the completed forward frontier")
        used = len(self._staged) if staged else len(self._checkpoints) - len(self._staged)
        capacity = self.max_staged_states if staged else self.max_cached_states
        if used >= capacity:
            raise MemoryError("Recurrent checkpoint capacity exhausted")
        slot = self._next_checkpoint
        self._next_checkpoint += 1
        conv, recurrent = self._working(table_idx)
        saved = self._allocate_checkpoint_storage(slot)
        self._checkpoints[slot] = saved
        try:
            target_conv, target_recurrent = (
                saved if self.arena is None else self._views(("checkpoint", slot))
            )
            target_conv.copy_(conv)
            target_recurrent.copy_(recurrent)
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            self.drop_checkpoint(StateCheckpoint(slot, self._processed_lens[table_idx]))
            raise
        if staged:
            self._staged.add(slot)
        return StateCheckpoint(slot, self._processed_lens[table_idx])

    def adopt_staged(self, checkpoint):
        if checkpoint.slot not in self._checkpoints:
            raise RuntimeError("Cannot publish a released checkpoint")
        pending = self._pending.get(checkpoint.slot)
        if pending is not None:
            if pending.plan.checkpoint != checkpoint or not pending.sealed:
                raise RuntimeError("Cannot publish an unsealed checkpoint")
        if checkpoint.slot in self._staged and self.available_checkpoint_slots <= 0:
            raise MemoryError("Recurrent tree checkpoint capacity exhausted")
        self._pending.pop(checkpoint.slot, None)
        self._staged.discard(checkpoint.slot)

    def release_staged(self, checkpoint):
        if checkpoint.slot in self._staged or checkpoint.slot in self._pending:
            self.drop_checkpoint(checkpoint)

    def restore(self, table_idx: int, uid: int, checkpoint: StateCheckpoint) -> None:
        """Copy shared state into a private working slot before any mutation."""
        self._slot(table_idx)
        if self._active[table_idx]:
            raise RuntimeError("Cannot restore during an unfinished forward")
        if self._owners[table_idx] not in (None, uid):
            raise RuntimeError("Cannot restore over another request's working state")
        pending = self._pending.get(checkpoint.slot)
        if pending is not None and not pending.sealed:
            raise RuntimeError("Cannot restore an unsealed checkpoint")
        saved = self._checkpoints[checkpoint.slot]
        self._allocate_working(table_idx)
        conv, recurrent = (
            saved if self.arena is None else self._views(("checkpoint", checkpoint.slot))
        )
        target_conv, target_recurrent = self._working(table_idx)
        try:
            target_conv.copy_(conv)
            target_recurrent.copy_(recurrent)
        except BaseException as error:
            if self.device_abort is not None:
                self.fail_device(error)
            raise
        self._owners[table_idx] = uid
        self._processed_lens[table_idx] = checkpoint.length

    def _forget_checkpoint(self, checkpoint):
        pending = self._pending.get(checkpoint.slot)
        self._checkpoints.pop(checkpoint.slot)
        self._staged.discard(checkpoint.slot)
        self._pending.pop(checkpoint.slot, None)
        physical = self._checkpoint_storage.pop(checkpoint.slot, None)
        if physical is not None:
            if pending is not None and pending.borrowed:
                self._deferred_checkpoint_storage[checkpoint.slot] = physical
            else:
                self._free_checkpoint_slots.append(physical)

    def drop_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        if checkpoint.slot in self._checkpoints:
            if self.arena is not None:
                self.arena.free_state(("checkpoint", checkpoint.slot))
            self._forget_checkpoint(checkpoint)

    def _slot(self, table_idx: int) -> None:
        if not 0 <= table_idx < self.capacity:
            raise IndexError("Hybrid request-state slot is out of range")

    def begin_request(self, table_idx: int, uid: int, cached_len: int) -> None:
        self._slot(table_idx)
        if cached_len < 0:
            raise ValueError("cached_len cannot be negative")
        if self._active[table_idx]:
            raise RuntimeError("Hybrid request already has an unfinished forward")
        if self._owners[table_idx] != uid:
            if cached_len != 0:
                raise RuntimeError("Hybrid cannot restore recurrent state from a token prefix")
            self._allocate_working(table_idx)
            conv, recurrent = self._working(table_idx)
            try:
                recurrent.zero_()
                conv.zero_()
            except BaseException as error:
                if self.device_abort is not None:
                    self.fail_device(error)
                raise
            self._owners[table_idx] = uid
            self._processed_lens[table_idx] = 0
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

    def release_request(self, table_idx: int, uid: int) -> None:
        self._slot(table_idx)
        # A late completion must never release the next request using this slot.
        if self._owners[table_idx] != uid:
            return
        if self.arena is not None:
            self.arena._check_mutable()
        for pending in list(self._pending.values()):
            if pending.plan.table_idx == table_idx and pending.plan.uid == uid:
                self.abort_checkpoint(pending.plan)
        if self.arena is None:
            self._recurrent[:, table_idx].zero_()
            self._conv[:, table_idx].zero_()
        else:
            self.arena.free_state(("request", table_idx))
        self._owners[table_idx] = None
        self._processed_lens[table_idx] = 0
        self._active[table_idx] = False

    def reset(self) -> None:
        for slot in list(self._checkpoints):
            self.drop_checkpoint(StateCheckpoint(slot, 0))
        for table_idx, uid in enumerate(self._owners):
            if uid is not None:
                self.release_request(table_idx, uid)

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
        """Resolve active owners to physical rows once for a numerical forward."""
        if len(set(table_indices)) != len(table_indices):
            raise ValueError("Recurrent batch cannot update one request twice")
        for index in table_indices:
            self._slot(index)
            if self._owners[index] is None or not self._active[index]:
                raise RuntimeError("begin_request must precede batched state access")
        if self.arena is not None:
            table_indices = [self.arena.state_index(("request", index)) for index in table_indices]
        device = self.arena.buffer.device if self.arena is not None else self._conv.device
        return torch.tensor(table_indices, dtype=torch.long, device=device)

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
