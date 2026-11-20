"""Topology checks for native FlashInfer exchanges."""

from minisgl.utils import is_sm100_supported


def supports_flashinfer_alltoall(*, dcp_size, tp_size, pp_size, nnodes):
    # Both callers construct a single-node group. Each CP group must fit on the node.
    return nnodes == pp_size == 1 and is_sm100_supported() and tp_size % dcp_size == 0
