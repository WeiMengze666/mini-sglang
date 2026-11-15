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
        if self.states.arena is None:
            return 0
        pinned = {handle.node.uuid for handle, _ in self._state_locks.values()}
        return (
            sum(node_id not in pinned for node_id in self.checkpoints)
            * self.states.arena.state_bytes
        )

    def __init__(self, device, states: RecurrentStatePool, max_states_per_path: int | None = None):
        super().__init__(device)
        if (
            max_states_per_path is not None and max_states_per_path < 1
        ) or not states.checkpoint_enabled:
            raise ValueError("Hybrid prefix caching requires positive checkpoint capacity")
        self.states = states
        self.max_states_per_path = max_states_per_path
        # Insertion order is the state-only LRU, independent of KV walk timestamps.
        self.checkpoints: dict[int, tuple[RadixTreeNode, StateCheckpoint]] = {}
        self._state_locks: dict[int, tuple[RadixCacheHandle, int]] = {}

    def lock_handle(self, handle, unlock=False, *, for_restore=True):
        anchored = isinstance(handle, RadixCacheHandle) and bool(handle.cached_len)
        entry = self._state_locks.get(id(handle)) if anchored else None
        if unlock and anchored and (entry is None or entry[0] is not handle):
            raise RuntimeError("Recurrent state unlock requires its matching handle receipt")
        super().lock_handle(handle, unlock)
        if not anchored:
            return
        if unlock:
            if entry[1] == 1:
                self._state_locks.pop(id(handle))
            else:
                self._state_locks[id(handle)] = (handle, entry[1] - 1)
        else:
            self._state_locks[id(handle)] = (handle, 1 if entry is None else entry[1] + 1)

    def match_prefix(self, input_ids):
        node, matched = self._tree_walk(input_ids)
        length = matched
        while not node.is_root() and node.uuid not in self.checkpoints:
            length -= node.length
            node = node.parent
        # No state at a newly split node: replay from an ancestor and place one
        # at the fork, rather than pretending append-only KV restores an SSM.
        self._touch_checkpoint(node)
        return MatchResult(HybridCacheHandle(length, node, matched if matched > length else 0))

    def continue_handle(self, old_handle, new_handle, processed_len):
        fork = getattr(old_handle, "replay_until", 0)
        return HybridCacheHandle(
            new_handle.cached_len, new_handle.node, fork if processed_len < fork else 0
        )

    def restore_state(self, table_idx, uid, handle):
        if handle.cached_len:
            entry = self._state_locks.get(id(handle))
            if entry is None or entry[0] is not handle:
                raise RuntimeError("Recurrent restore requires a live source lock receipt")
            checkpoint = self.checkpoints[handle.node.uuid][1]
            if checkpoint.length != handle.cached_len:
                raise RuntimeError("Recurrent checkpoint and radix prefix lengths differ")
            while True:
                try:
                    self.states.restore(table_idx, uid, checkpoint)
                    break
                except MemoryError:
                    # Atomic admission leaves the source untouched. Its receipt
                    # stays held through COW and the continuing request lifetime.
                    if not self._evict_checkpoint(exclude=handle.node.uuid):
                        raise

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

    def _touch_checkpoint(self, node):
        if node.uuid in self.checkpoints:
            self.checkpoints[node.uuid] = self.checkpoints.pop(node.uuid)

    def _evict_checkpoint(self, exclude=None):
        pinned = {handle.node.uuid for handle, _ in self._state_locks.values()}
        node = next(
            (
                node
                for node, _ in self.checkpoints.values()
                if node.uuid != exclude and node.uuid not in pinned
            ),
            None,
        )
        if node is None:
            return False
        # Active state-component receipts protect their anchor even after COW.
        self._drop(node)
        return True

    def _drop(self, node):
        entry = self.checkpoints.get(node.uuid)
        if entry is not None:
            self.states.drop_checkpoint(entry[1])
            self.checkpoints.pop(node.uuid)

    def cache_state(self, req, handle, *, checkpoint=None, cached_len=None, finished=False):
        length = req.cached_len if cached_len is None else cached_len
        if handle.cached_len != length or not length:
            return
        if checkpoint is not None and checkpoint.length != length:
            raise RuntimeError("Staged checkpoint and radix frontier lengths differ")
        node = handle.node
        if node.uuid in self.checkpoints:
            self._touch_checkpoint(node)
            return  # caller retains the unused track until its normal release
        pinned = {anchor.node.uuid for anchor, _ in self._state_locks.values()}
        path, parent = [], node
        while not parent.is_root():
            if parent.uuid in self.checkpoints:
                path.append(parent)
            parent = parent.parent
        if self.max_states_per_path is not None and len(path) >= self.max_states_per_path:
            candidates = [
                n for n, _ in self.checkpoints.values() if n in path and n.uuid not in pinned
            ]
            if not candidates:
                return
            self._drop(candidates[0])
        snapshot = checkpoint
        while True:
            try:
                if snapshot is None:
                    snapshot = self.states.snapshot(req.table_idx, req.uid, expected_len=length)
                # Ongoing requests reserve a replacement before donating their
                # tracking row. Finished requests need no replacement.
                self.states.adopt_staged(snapshot, finished=finished)
                break
            except MemoryError:
                if self.evict_checkpoint():
                    continue
                if checkpoint is None and snapshot is not None:
                    self.states.release_staged(snapshot)
                return
        self.checkpoints[node.uuid] = (node, snapshot)

    def _remove_node(self, node):
        self._drop(node)
        super()._remove_node(node)
