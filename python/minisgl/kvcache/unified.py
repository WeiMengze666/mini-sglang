"""Two-ended byte arena with stable logical handles and compact physical blocks.

Recurrent state grows from the left, token pages from the right. Removing a
block moves the last block of that kind into its hole, then updates indirection.
Views must be acquired after allocations/frees; mutations run on the forward
stream so earlier kernels finish using an old view before a move overwrites it.
"""

import torch


def align_bytes(size: int) -> int:
    return (size + 255) // 256 * 256


class UnifiedMemory:
    def __init__(self, num_pages, page_bytes, state_bytes, device):
        self.page_bytes, self.state_bytes = align_bytes(page_bytes), align_bytes(state_bytes)
        if min(num_pages, self.page_bytes, self.state_bytes) <= 0:
            raise ValueError("Unified memory requires positive capacities")
        self.buffer = torch.empty(num_pages * self.page_bytes, dtype=torch.uint8, device=device)
        self.page_map = torch.full((num_pages,), -1, dtype=torch.int64, device=device)
        self._states: dict[object, int] = {}
        self._pages: dict[int, int] = {}
        self._state_ids: list[object] = []
        self._page_ids: list[int] = []

    @property
    def free_bytes(self):
        return (
            self.buffer.numel()
            - len(self._states) * self.state_bytes
            - len(self._pages) * self.page_bytes
        )

    @property
    def free_pages(self):
        return self.free_bytes // self.page_bytes

    def state_view(self, key):
        start = self._states[key] * self.state_bytes
        return self.buffer[start : start + self.state_bytes]

    def state_index(self, key):
        """Current physical row; resolve again after any allocation or free."""
        return self._states[key]

    def has_state(self, key):
        return key in self._states

    def allocate_state(self, key):
        if key in self._states:
            raise ValueError("State handle is already allocated")
        if self.free_bytes < self.state_bytes:
            raise MemoryError("Unified memory has insufficient space for recurrent state")
        self._states[key] = len(self._state_ids)
        self._state_ids.append(key)
        return self.state_view(key)

    def free_state(self, key):
        index = self._states.pop(key)
        last = self._state_ids.pop()
        if last != key:
            target = self.buffer[index * self.state_bytes : (index + 1) * self.state_bytes]
            target.copy_(self.state_view(last))
            self._states[last] = index
            self._state_ids[index] = last

    def _page_view(self, index):
        stop = self.buffer.numel() - index * self.page_bytes
        return self.buffer[stop - self.page_bytes : stop]

    def allocate_pages(self, keys):
        keys = [int(key) for key in keys]
        if len(set(keys)) != len(keys) or any(key in self._pages for key in keys):
            raise ValueError("Token page handle is already allocated")
        if any(not 0 <= key < len(self.page_map) for key in keys):
            raise IndexError("Logical token page is out of range")
        if len(keys) > self.free_pages:
            raise MemoryError("Unified memory has insufficient space for token pages")
        physical = []
        for key in keys:
            self._pages[key] = index = len(self._page_ids)
            self._page_ids.append(key)
            self._page_view(index).zero_()
            physical.append(len(self.page_map) - 1 - index)
        self.page_map[keys] = torch.tensor(physical, device=self.buffer.device)

    def free_pages_by_id(self, keys):
        for key in keys:
            key = int(key)
            index = self._pages.pop(key)
            last = self._page_ids.pop()
            if last != key:
                self._page_view(index).copy_(self._page_view(self._pages[last]))
                self._pages[last] = index
                self._page_ids[index] = last
                self.page_map[last] = len(self.page_map) - 1 - index
            self.page_map[key] = -1

    def page_storage(self, dtype):
        return self.buffer.view(dtype).view(len(self.page_map), self.page_bytes // dtype.itemsize)

    def check_integrity(self):
        assert self.free_bytes >= 0
        assert len(self._state_ids) == len(self._states)
        assert len(self._page_ids) == len(self._pages)
        assert all(self._states[key] == i for i, key in enumerate(self._state_ids))
        assert all(self._pages[key] == i for i, key in enumerate(self._page_ids))
