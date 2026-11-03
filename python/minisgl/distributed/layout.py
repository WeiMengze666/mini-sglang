"""Map a TP group to independent DCP and EP coordinates."""

from dataclasses import dataclass

from .info import DistributedInfo, local_size, try_get_tp_info


@dataclass(frozen=True)
class ParallelLayout:
    tensor: DistributedInfo = DistributedInfo(0, 1)
    dcp_size: int = 1
    ep_size: int = 1

    def __post_init__(self):
        local_size(self.tensor.size, self.dcp_size, "TP size for DCP")
        local_size(self.tensor.size, self.ep_size, "TP size for EP")

    @property
    def context(self):
        return DistributedInfo(self.tensor.rank % self.dcp_size, self.dcp_size)

    @property
    def mla_decode(self):
        """Expanded query-head partition for MLA decode; KDA keeps full TP."""
        return DistributedInfo(self.tensor.rank // self.dcp_size, self.tensor.size // self.dcp_size)

    @property
    def experts(self):
        return DistributedInfo(self.tensor.rank % self.ep_size, self.ep_size)

    @property
    def expert_tensor(self):
        return DistributedInfo(self.tensor.rank // self.ep_size, self.tensor.size // self.ep_size)

    def expert_range(self, num_experts):
        return self.experts.range(num_experts, "experts")


_DCP_SIZE, _EP_SIZE = 1, 1


def get_parallel_layout():
    tensor = try_get_tp_info() or DistributedInfo(0, 1)
    return ParallelLayout(tensor, _DCP_SIZE, _EP_SIZE)


def configure_parallel(*, dcp_size=1, ep_size=1):
    global _DCP_SIZE, _EP_SIZE
    ParallelLayout(try_get_tp_info() or DistributedInfo(0, 1), dcp_size, ep_size)
    _DCP_SIZE, _EP_SIZE = dcp_size, ep_size
