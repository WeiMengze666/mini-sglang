from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                            error=msg.error,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                requests: List[BaseBackendMsg] = []
                for msg in tokenize_msg:
                    try:
                        (input_ids,) = tokenize_manager.tokenize([msg])
                    except (ValueError, TypeError, OverflowError) as exc:
                        logger.warning("Rejecting request %d: %s", msg.uid, exc)
                        send_frontend.put(UserReply(msg.uid, "", True, error=str(exc)))
                        continue
                    requests.append(UserMsg(msg.uid, input_ids, msg.sampling_params))
                if requests:
                    send_backend.put(
                        requests[0] if len(requests) == 1 else BatchBackendMsg(data=requests)
                    )
            if len(abort_msg) > 0:
                for msg in abort_msg:
                    detokenize_manager.abort(msg.uid)
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
