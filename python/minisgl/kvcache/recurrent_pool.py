"""Request-owned recurrent and convolution state; stale releases cannot free a new owner."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class StateCheckpoint:
    slot: int
    length: int


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
    ):
        self._layers = {layer: i for i, layer in enumerate(layer_ids)}
        self.capacity = capacity
        self.arena = arena
        self._recurrent_shape = (len(self._layers), heads, dim, dim)
        self._conv_shape = (len(self._layers), 3, heads * dim, conv_kernel_size - 1)
        self._dtype = dtype
        self._recurrent = (
            None
            if arena is not None
            else torch.zeros(
                (len(self._layers), capacity, heads, dim, dim),
                dtype=torch.float32,
                device=device,
            )
        )
        self._conv = (
            None
            if arena is not None
            else torch.zeros(
                (
                    len(self._layers),
                    capacity,
                    3,
                    heads * dim,
                    conv_kernel_size - 1,
                ),
                dtype=dtype,
                device=device,
            )
        )
        self._owners: list[int | None] = [None] * capacity
        self._processed_lens = [0] * capacity
        self._active = [False] * capacity
        self.max_cached_states = max_cached_states
        self._checkpoints: dict[int, tuple[torch.Tensor, torch.Tensor] | None] = {}
        self._next_checkpoint = 0
        self.max_staged_states = max_staged_states
        self._staged: set[int] = set()

    def _views(self, key):
        return self._decode_views(self.arena.state_view(key))

    def _decode_views(self, data):
        count = prod(self._recurrent_shape)
        recurrent = data[: count * 4].view(torch.float32).view(self._recurrent_shape)
        conv_bytes = prod(self._conv_shape) * self._dtype.itemsize
        conv = data[count * 4 : count * 4 + conv_bytes].view(self._dtype).view(self._conv_shape)
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
        if self.arena is None:
            self._checkpoints[slot] = (conv.clone(), recurrent.clone())
        else:
            self.arena.allocate_state(("checkpoint", slot))
            target_conv, target_recurrent = self._views(("checkpoint", slot))
            target_conv.copy_(conv)
            target_recurrent.copy_(recurrent)
            self._checkpoints[slot] = None
        if staged:
            self._staged.add(slot)
        return StateCheckpoint(slot, self._processed_lens[table_idx])

    def adopt_staged(self, checkpoint):
        if checkpoint.slot in self._staged:
            if len(self._checkpoints) - len(self._staged) >= self.max_cached_states:
                raise MemoryError("Recurrent tree checkpoint capacity exhausted")
            self._staged.remove(checkpoint.slot)

    def release_staged(self, checkpoint):
        if checkpoint.slot in self._staged:
            self.drop_checkpoint(checkpoint)

    def restore(self, table_idx: int, uid: int, checkpoint: StateCheckpoint) -> None:
        """Copy shared state into a private working slot before any mutation."""
        self._slot(table_idx)
        if self._active[table_idx]:
            raise RuntimeError("Cannot restore during an unfinished forward")
        if self._owners[table_idx] not in (None, uid):
            raise RuntimeError("Cannot restore over another request's working state")
        saved = self._checkpoints[checkpoint.slot]
        self._allocate_working(table_idx)
        conv, recurrent = (
            saved if self.arena is None else self._views(("checkpoint", checkpoint.slot))
        )
        target_conv, target_recurrent = self._working(table_idx)
        target_conv.copy_(conv)
        target_recurrent.copy_(recurrent)
        self._owners[table_idx] = uid
        self._processed_lens[table_idx] = checkpoint.length

    def drop_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        if checkpoint.slot in self._checkpoints:
            self._checkpoints.pop(checkpoint.slot)
            self._staged.discard(checkpoint.slot)
            if self.arena is not None:
                self.arena.free_state(("checkpoint", checkpoint.slot))

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
            recurrent.zero_()
            conv.zero_()
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

    def get_kda_batch(self, layer_id, indices):
        from minisgl.attention.kda import KDAState

        return KDAState(*(part.index_select(0, indices) for part in self._batched_layer(layer_id)))

    def set_kda_batch(self, layer_id, indices, state):
        for target, value in zip(self._batched_layer(layer_id), state):
            target.index_copy_(0, indices, value)
