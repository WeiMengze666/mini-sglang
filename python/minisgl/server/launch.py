from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from contextlib import ExitStack
from dataclasses import replace
from queue import Empty
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _wait_for_workers(ack_queue, workers, count):
    """Read readiness messages while detecting a failed worker on any rank."""
    while count:
        exited = [
            (worker.name, worker.exitcode) for worker in workers if worker.exitcode is not None
        ]
        if exited:
            raise RuntimeError(f"Workers exited during startup: {exited}")
        try:
            message = ack_queue.get(timeout=0.25)
        except Empty:
            continue
        yield message
        count -= 1


def _stop_workers(workers):
    """Clean up only processes created by this initializer."""
    started = [worker for worker in workers if worker.pid is not None]
    for worker in started:
        if worker.is_alive():
            worker.terminate()
    for worker in started:
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        workers = []
        with ExitStack() as startup:
            startup.callback(ack_queue.close)
            startup.callback(_stop_workers, workers)

            def start(**kwargs):
                worker = mp.Process(daemon=False, **kwargs)
                workers.append(worker)
                worker.start()

            for i in range(world_size):
                new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
                start(
                    target=_run_scheduler,
                    args=(new_args, ack_queue),
                    name=f"minisgl-TP{i}-scheduler",
                )

            num_tokenizers = server_args.num_tokenizer
            start(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_detokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": num_tokenizers,
                    "ack_queue": ack_queue,
                },
                name="minisgl-detokenizer-0",
            )
            for i in range(num_tokenizers):
                start(
                    target=tokenize_worker,
                    kwargs={
                        "tokenizer_path": server_args.model_path,
                        "addr": server_args.zmq_tokenizer_addr,
                        "backend_addr": server_args.zmq_backend_addr,
                        "frontend_addr": server_args.zmq_frontend_addr,
                        "local_bs": 1,
                        "create": server_args.tokenizer_create_addr,
                        "tokenizer_id": i,
                        "ack_queue": ack_queue,
                    },
                    name=f"minisgl-tokenizer-{i}",
                )

            # Only the primary scheduler acknowledges the synchronized TP group.
            for message in _wait_for_workers(ack_queue, workers, num_tokenizers + 2):
                logger.info(message)
            ack_queue.close()
            startup.pop_all()

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
