"""Independent context and expert layouts inside a tensor-parallel world."""

from dataclasses import dataclass

from .parallel import local_size, tp_info
from .partition import Partition


@dataclass(frozen=True)
class ParallelLayout:
    rank: int = 0
    size: int = 1
    dcp_size: int = 1
    ep_size: int = 1

    def __post_init__(self):
        if not 0 <= self.rank < self.size:
            raise ValueError("Parallel rank must belong to the TP world")
        local_size(self.size, self.dcp_size, "TP size for DCP")
        local_size(self.size, self.ep_size, "TP size for EP")

    @property
    def dcp_rank(self):
        return self.rank % self.dcp_size

    @property
    def attention_rank(self):
        return self.rank // self.dcp_size

    @property
    def attention_size(self):
        return self.size // self.dcp_size

    @property
    def expert_rank(self):
        return self.rank % self.ep_size

    @property
    def expert_tensor_rank(self):
        return self.rank // self.ep_size

    @property
    def expert_tensor_size(self):
        return self.size // self.ep_size

    def expert_range(self, num_experts):
        return self.experts.range(num_experts, "experts")

    @property
    def tensor(self):
        return Partition(self.rank, self.size)

    @property
    def attention(self):
        return Partition(self.attention_rank, self.attention_size)

    @property
    def experts(self):
        return Partition(self.expert_rank, self.ep_size)

    @property
    def expert_tensor(self):
        return Partition(self.expert_tensor_rank, self.expert_tensor_size)


_DCP_SIZE, _EP_SIZE = 1, 1


def get_parallel_layout():
    rank, size = tp_info()
    return ParallelLayout(rank, size, _DCP_SIZE, _EP_SIZE)


def configure_parallel(*, dcp_size=1, ep_size=1):
    global _DCP_SIZE, _EP_SIZE
    ParallelLayout(*tp_info(), dcp_size, ep_size)
    _DCP_SIZE, _EP_SIZE = dcp_size, ep_size
