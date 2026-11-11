from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req

from .utils import CLIP_MAX_NEW_TOKENS


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
    _prompt_lens: dict[int, int] = field(default_factory=dict)

    @property
    def next_page_tokens(self) -> int:
        return sum(
            (
                (req.device_len + self.page_size - 1) // self.page_size
                - (req.cached_len + self.page_size - 1) // self.page_size
            )
            * self.page_size
            for req in self.running_reqs
        )

    def retraction_candidate(self) -> Req:
        # Native "length" policy retracts least generated, then longest prompt.
        return min(
            self.running_reqs,
            key=lambda req: (
                len(req.input_ids) - self._prompt_lens[req.uid],
                -self._prompt_lens[req.uid],
                -req.uid,
            ),
        )

    def retract_req(self, req: Req) -> None:
        self.running_reqs.remove(req)  # Preserve original prompt length on requeue.

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}
        for req in self.running_reqs:
            self._prompt_lens.setdefault(req.uid, req.max_device_len - req.output_len)

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        self._prompt_lens.pop(req.uid, None)

    def abort_req(self, uid: int) -> Req | None:
        self._prompt_lens.pop(uid, None)
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return (
            sum(min(req.remain_len, CLIP_MAX_NEW_TOKENS) for req in self.running_reqs)
            + tokens_reserved
        )

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
