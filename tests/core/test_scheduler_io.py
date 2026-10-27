"""Reliable first-message delivery and idle waiting across real TP control ranks."""

import os
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
import zmq
from minisgl.distributed import DistributedInfo
from minisgl.message import AbortBackendMsg, BaseBackendMsg
from minisgl.scheduler.io import SchedulerIOMixin
from minisgl.utils import ZmqPushQueue


class _Scheduler(SchedulerIOMixin):
    def run_when_idle(self):
        pass


def _worker(rank, world, folder):
    base = Path(folder)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(base / "init"),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=3))
    queues, thread = [], None
    try:
        # Publishing before the last receiver exists must not lose its wakeup.
        if rank == world - 1:
            time.sleep(0.4)
        config = SimpleNamespace(
            tp_info=DistributedInfo(rank, world),
            offline_mode=False,
            zmq_backend_addr=f"ipc://{base}/backend",
            zmq_detokenizer_addr=f"ipc://{base}/detokenizer",
            zmq_scheduler_broadcast_addr=f"ipc://{base}/control",
            backend_create_detokenizer_link=True,
        )
        scheduler = _Scheduler(config, control)
        if rank == 0:
            queues.extend((scheduler._recv_from_tokenizer, scheduler._send_into_tokenizer))
            targets = scheduler._send_into_ranks
            queues.extend(targets if isinstance(targets, list) else [targets])
            sender = ZmqPushQueue(
                config.zmq_backend_addr, create=False, encoder=BaseBackendMsg.encoder
            )
            queues.append(sender)
            for uid in (11, 12, 13):
                sender.put(AbortBackendMsg(uid))
        else:
            queues.append(scheduler._recv_from_rank0)
        for queue in queues:
            queue.socket.setsockopt(zmq.LINGER, 0)
            queue.socket.setsockopt(zmq.RCVTIMEO, 8000)
            queue.socket.setsockopt(zmq.SNDTIMEO, 8000)

        received = scheduler.receive_msg(blocking=True)
        deadline = time.monotonic() + 5
        while len(received) < 3 and time.monotonic() < deadline:
            received.extend(scheduler.receive_msg(blocking=False))
        assert [msg.uid for msg in received] == [11, 12, 13]
        control.barrier().wait()

        if rank == 0:

            def delayed_request():
                # Each ZMQ socket stays on the thread that created it.
                delayed = ZmqPushQueue(
                    config.zmq_backend_addr, create=False, encoder=BaseBackendMsg.encoder
                )
                time.sleep(4)
                delayed.put(AbortBackendMsg(14))
                delayed.stop()

            thread = threading.Thread(target=delayed_request)
            thread.start()
        tick = time.monotonic()
        received = scheduler.receive_msg(blocking=True)
        assert time.monotonic() - tick >= 3.5
        assert [msg.uid for msg in received] == [14]
        control.barrier().wait()
    finally:
        if thread is not None:
            thread.join(timeout=10)
        for queue in queues:
            queue.stop()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_GLOO_TESTS") != "1", reason="opt-in multiprocess CPU gate"
)
def test_delayed_receiver_and_idle_longer_than_collective_timeout():
    # Short paths also stay below the platform's Unix-domain socket path limit.
    with tempfile.TemporaryDirectory(prefix="minisgl-io-") as folder:
        mp.spawn(_worker, args=(4, folder), nprocs=4, join=True)
