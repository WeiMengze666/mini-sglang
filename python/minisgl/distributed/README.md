# Parallel metadata and communication

`DistributedInfo(rank, size)` is the single logical group descriptor. It also
derives contiguous ranges, tensor shard specs, and local expert IDs. It holds no
process-group or communication resources.

`ParallelLayout(tensor, dcp_size, ep_size)` composes the TP descriptor with
independent context/expert layout rules. Its `context`, `attention`, `experts`
and `expert_tensor` views are derived `DistributedInfo` objects. It does not
store another rank/size pair or parallel copies of the derived coordinates.

For TP rank 5 of 8, DCP=2 and EP=4:

| View | Local rank | Size |
| --- | ---: | ---: |
| tensor | 5 | 8 |
| context | 1 | 2 |
| attention | 2 | 4 |
| experts | 1 | 4 |
| expert_tensor | 1 | 2 |

These are logical coordinates; they are not global member lists.
`init_context_parallel` creates the actual subgroup handles in a consistent order.

`ShardSpec(axis, start, stop)` in `shard.py` describes a tensor slice, independent
of group topology. Layers and checkpoint readers derive it from a group view.

`TensorParallel` and `ContextParallel` retain a group descriptor and implement
operations; `DistributedImpl` backends own the underlying TP communication.
No model name is needed for the layout, shard or transport decisions.
