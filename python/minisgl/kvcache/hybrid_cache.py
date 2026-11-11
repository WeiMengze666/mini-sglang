"""Sparse recurrent checkpoints over the existing token radix tree.

KV pages can match only as far as a restorable recurrent checkpoint. When a
match splits an edge, replay stops at the aligned fork and checkpoints it.
Working state, snapshots, and donated tree state have separate ownership.
"""

from dataclasses import dataclass

from .base import MatchResult
from .radix_cache import RadixCacheHandle, RadixPrefixCache, RadixTreeNode
from .recurrent_pool import RecurrentStatePool, StateCheckpoint


@dataclass(frozen=True)
class HybridCacheHandle(RadixCacheHandle):
    replay_until: int = 0


class HybridPrefixCache(RadixPrefixCache):
    # Decode boundaries remain request-private until the completed turn is published.
    checkpoint_interval = None

    @property
    def reclaimable_state_bytes(self):
        return (
            len(self.checkpoints) * self.states.arena.state_bytes
            if self.states.arena is not None
            else 0
        )

    def __init__(self, device, states: RecurrentStatePool, max_states_per_path: int | None = None):
        super().__init__(device)
        if (
            max_states_per_path is not None and max_states_per_path < 1
        ) or states.max_cached_states < 1:
            raise ValueError("Hybrid prefix caching requires positive checkpoint capacity")
        self.states = states
        self.max_states_per_path = max_states_per_path
        self.checkpoints: dict[int, tuple[RadixTreeNode, StateCheckpoint]] = {}
        self._pending_restores: dict[int, tuple[int, int]] = {}

    def lock_handle(self, handle, unlock=False, *, for_restore=True):
        super().lock_handle(handle, unlock)
        if isinstance(handle, HybridCacheHandle) and handle.cached_len:
            if unlock:
                self._unpin_state(handle)
            elif for_restore:
                count = self._pending_restores.get(id(handle), (0, 0))[1]
                self._pending_restores[id(handle)] = (handle.node.uuid, count + 1)

    def _unpin_state(self, handle):
        entry = self._pending_restores.pop(id(handle), None)
        if entry is not None and entry[1] > 1:
            self._pending_restores[id(handle)] = (entry[0], entry[1] - 1)

    def match_prefix(self, input_ids):
        node, matched = self._tree_walk(input_ids)
        length = matched
        while not node.is_root() and node.uuid not in self.checkpoints:
            length -= node.length
            node = node.parent
        # No state at a newly split node: replay from an ancestor and place one
        # at the fork, rather than pretending append-only KV restores an SSM.
        return MatchResult(HybridCacheHandle(length, node, matched if matched > length else 0))

    def continue_handle(self, old_handle, new_handle, processed_len):
        fork = getattr(old_handle, "replay_until", 0)
        return HybridCacheHandle(
            new_handle.cached_len, new_handle.node, fork if processed_len < fork else 0
        )

    def restore_state(self, table_idx, uid, handle):
        if handle.cached_len:
            checkpoint = self.checkpoints[handle.node.uuid][1]
            if checkpoint.length != handle.cached_len:
                raise RuntimeError("Recurrent checkpoint and radix prefix lengths differ")
            arena = self.states.arena
            if arena is not None:
                self._release_checkpoint_memory(arena.state_bytes, exclude=handle.node.uuid)
            self.states.restore(table_idx, uid, checkpoint)
            self._unpin_state(handle)

    def release_checkpoint_memory(self, required_bytes):
        self._release_checkpoint_memory(required_bytes)

    def _release_checkpoint_memory(self, required_bytes, exclude=None):
        arena = self.states.arena
        if arena is None:
            return
        while arena.free_bytes < required_bytes and self._evict_checkpoint(exclude):
            pass

    def evict_checkpoint(self) -> bool:
        return self._evict_checkpoint()

    def _evict_checkpoint(self, exclude=None):
        pinned = {node_id for node_id, _ in self._pending_restores.values()}
        node = min(
            (
                node
                for node, _ in self.checkpoints.values()
                if node.uuid != exclude and node.uuid not in pinned
            ),
            key=lambda n: n.timestamp,
            default=None,
        )
        if node is None:
            return False
        # Restored readers own private state; only a queued restore pins this slot.
        self._drop(node)
        return True

    def _drop(self, node):
        entry = self.checkpoints.get(node.uuid)
        if entry is not None:
            self.states.drop_checkpoint(entry[1])
            self.checkpoints.pop(node.uuid)

    def cache_state(self, req, handle, *, checkpoint=None, cached_len=None):
        # A state cannot be rewound to an earlier page boundary. Checkpoint only
        # exactly aligned committed states and replay the remaining tail later.
        length = req.cached_len if cached_len is None else cached_len
        if handle.cached_len != length or not length:
            return
        if checkpoint is not None and checkpoint.length != length:
            raise RuntimeError("Staged checkpoint and radix frontier lengths differ")
        node = handle.node
        if node.uuid in self.checkpoints:
            return
        # KV readers already own a private restored state. Only an outstanding
        # restore must keep its tree checkpoint alive until the copy is queued.
        pinned = {node_id for node_id, _ in self._pending_restores.values()}
        path, parent = [], node
        while not parent.is_root():
            if parent.uuid in self.checkpoints:
                path.append(parent)
            parent = parent.parent
        if self.max_states_per_path is not None and len(path) >= self.max_states_per_path:
            candidates = [n for n in path if n.uuid not in pinned]
            if not candidates:
                return
            self._drop(min(candidates, key=lambda n: n.timestamp))
        if len(self.checkpoints) >= self.states.max_cached_states:
            if not self.evict_checkpoint():
                return
        try:
            snapshot = checkpoint or self.states.snapshot(
                req.table_idx, req.uid, expected_len=length
            )
            self.states.adopt_staged(snapshot)
        except MemoryError:
            # KV and working state take precedence over an optional checkpoint.
            return
        self.checkpoints[node.uuid] = (node, snapshot)  # donate: transfer the slot, no copy

    def _remove_node(self, node):
        self._drop(node)
        super()._remove_node(node)
