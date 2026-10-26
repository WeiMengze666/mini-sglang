"""CPU request-pipeline regression tests; no GPU or model weights are used."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.engine import ForwardOutput
from minisgl.llm.llm import LLM
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager
from minisgl.tokenizer.detokenize import DetokenizeManager


def tokenizer():
    pieces = {1: b"hello", 2: b" world", 3: b"\xe4", 4: b"\xbd", 5: b"\xa0", 99: b"STOP"}

    def decode(ids):
        return b"".join(pieces[i] for i in ids).decode("utf-8", errors="replace")

    return SimpleNamespace(
        eos_token_id=99,
        stop_token_ids={99},
        batch_decode=lambda rows: [decode(ids) for ids in rows],
        decode=decode,
    )


def test_detokenization_is_invariant_to_batch_partitioning():
    messages = [
        DetokenizeMsg(7, 1, False),
        DetokenizeMsg(8, 3, False),
        DetokenizeMsg(7, 2, False),
        DetokenizeMsg(8, 4, False),
        DetokenizeMsg(8, 5, False),
        DetokenizeMsg(7, 99, True),
        DetokenizeMsg(8, 99, True),
    ]
    expected = ["hello", "", " world", "", "你", "", ""]
    for partition in range(1 << (len(messages) - 1)):
        manager = DetokenizeManager(tokenizer())
        output, start = [], 0
        for end in range(1, len(messages) + 1):
            if end == len(messages) or partition & (1 << (end - 1)):
                output.extend(manager.detokenize(messages[start:end]))
                start = end
        assert output == expected
        assert manager.decode_map == {}


def test_abort_clears_state_and_backend_terminal_clears_late_tokens():
    manager = DetokenizeManager(tokenizer())
    manager.detokenize([DetokenizeMsg(7, 1, False)])
    manager.abort(7)
    manager.abort(7)  # Repeated cancellation is harmless.
    assert manager.decode_map == {}
    manager.detokenize([DetokenizeMsg(7, 2, False), DetokenizeMsg(7, None, True)])
    assert manager.decode_map == {}
    assert manager.detokenize([DetokenizeMsg(7, 1, True)]) == ["hello"]


@pytest.mark.parametrize(
    "message,base",
    [
        (DetokenizeMsg(7, None, True), BaseTokenizerMsg),
        (DetokenizeMsg(7, None, True, error="invalid prompt"), BaseTokenizerMsg),
        (UserReply(7, "", True, error="invalid prompt"), BaseFrontendMsg),
    ],
)
def test_tokenless_terminal_and_error_messages_round_trip(message, base):
    assert base.decoder(base.encoder(message)) == message


@pytest.fixture
def scheduler(monkeypatch):
    # Replace only accelerator allocation/compute. Request admission, chunking,
    # table allocation, page allocation, decode, and cleanup use real managers.
    import minisgl.distributed.info as distributed_info

    monkeypatch.setattr(distributed_info, "_TP_INFO", distributed_info.DistributedInfo(0, 1))
    original_empty = torch.empty
    original_tensor = torch.tensor

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    scheduler = object.__new__(Scheduler)
    scheduler._has_recurrent_state = True
    scheduler.vocab_size = 100
    scheduler.device = torch.device("cpu")
    scheduler.finished_reqs = set()
    scheduler.aborted_reqs = set()
    scheduler.stop_token_ids = {99}
    scheduler.prefill_budget = 8
    scheduler.replies = []
    scheduler.released = []
    page_table = torch.full((3, 8), -1, dtype=torch.int32)
    scheduler.table_manager = TableManager(2, page_table)
    scheduler.cache_manager = CacheManager(32, 1, page_table, "naive")
    scheduler.decode_manager = DecodeManager(1)
    scheduler.prefill_manager = PrefillManager(
        scheduler.cache_manager, scheduler.table_manager, scheduler.decode_manager
    )
    scheduler.token_pool = scheduler.table_manager.token_pool
    scheduler.send_result = scheduler.replies.extend
    scheduler.receive_msg = lambda blocking=False: []

    def forward(batch, sampling_args):
        for req in batch.reqs:
            req.complete_one()
        next_tokens = torch.ones(len(batch.reqs), dtype=torch.int32)
        return ForwardOutput(next_tokens, next_tokens, SimpleNamespace(synchronize=lambda: None))

    scheduler.engine = SimpleNamespace(
        max_seq_len=8,
        page_table=page_table,
        graph_runner=SimpleNamespace(
            pad_batch=lambda batch: setattr(batch, "padded_reqs", batch.reqs)
        ),
        sampler=SimpleNamespace(prepare=lambda batch: None),
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
        kv_cache=SimpleNamespace(
            release_request=lambda table_idx, uid: scheduler.released.append((table_idx, uid))
        ),
        forward_batch=forward,
    )
    return scheduler


def request(uid=7, tokens=(1, 2), max_tokens=2):
    return UserMsg(
        uid, torch.tensor(tokens, dtype=torch.int32), SamplingParams(max_tokens=max_tokens)
    )


def assert_idle(scheduler):
    assert not scheduler.prefill_manager.runnable
    assert not scheduler.decode_manager.runnable
    assert scheduler.table_manager.available_size == 2
    assert len(set(scheduler.table_manager._free_slots)) == 2
    assert scheduler.cache_manager.available_size == 32
    scheduler.cache_manager.check_integrity()


@pytest.mark.parametrize(
    "tokens,max_tokens,error",
    [
        ([], 1, "at least one token"),
        (list(range(8)), 1, "maximum sequence length"),
        (list(range(9)), 0, "maximum sequence length"),
        ([1], -1, "non-negative integer"),
        ([1], 1.5, "non-negative integer"),
        ([1], True, "non-negative integer"),
        ([-1], 1, "token IDs in [0, 100)"),
        ([100], 1, "token IDs in [0, 100)"),
        ([1, 2**31 - 1], 1, "token IDs in [0, 100)"),
        ([1], 0, None),
        (list(range(8)), 0, None),
    ],
)
def test_invalid_and_zero_output_requests_are_terminal_without_allocations(
    scheduler, tokens, max_tokens, error
):
    scheduler._process_one_msg(request(tokens=tokens, max_tokens=max_tokens))
    assert len(scheduler.replies) == 1
    reply = scheduler.replies[0]
    assert reply.finished and reply.next_token is None
    assert (reply.error is None) if error is None else error in reply.error
    assert scheduler.released == []
    assert_idle(scheduler)
    # Rejection cannot poison a following valid request or the batch loop.
    scheduler._process_one_msg(request(uid=8, max_tokens=2))
    scheduler.normal_loop()
    scheduler.normal_loop()
    assert [m.next_token for m in scheduler.replies if m.uid == 8] == [1, 1]
    assert scheduler.replies[-1].finished
    assert_idle(scheduler)


@pytest.mark.parametrize("input_ids", [torch.tensor(1), torch.ones(2, 2), torch.ones(2)])
def test_malformed_input_tensor_is_rejected(scheduler, input_ids):
    scheduler._process_one_msg(UserMsg(7, input_ids, SamplingParams()))
    assert "one-dimensional CPU integer tensor" in scheduler.replies[0].error
    assert_idle(scheduler)


def test_clipping_does_not_mutate_shared_sampling_params(scheduler):
    params = SamplingParams(max_tokens=4)
    scheduler._process_one_msg(
        BatchBackendMsg(
            [
                UserMsg(7, torch.ones(7, dtype=torch.int32), params),
                UserMsg(8, torch.ones(2, dtype=torch.int32), params),
            ]
        )
    )
    pending = scheduler.prefill_manager.pending_list
    assert [req.output_len for req in pending] == [1, 4]
    assert pending[0].sampling_params is not pending[1].sampling_params
    assert params.max_tokens == 4
    for _ in range(10):
        scheduler.normal_loop()
        if not (scheduler.prefill_manager.runnable or scheduler.decode_manager.runnable):
            break
    assert len([reply for reply in scheduler.replies if reply.uid == 7]) == 1
    assert len([reply for reply in scheduler.replies if reply.uid == 8]) == 4
    assert_idle(scheduler)


@pytest.mark.parametrize("stage", ["pending", "chunked", "decoding", "inflight"])
def test_abort_reclaims_real_scheduler_resources_and_sends_terminal(scheduler, stage):
    scheduler.prefill_budget = 2 if stage == "chunked" else 8
    scheduler._process_one_msg(request(tokens=(1, 2, 1, 2), max_tokens=3))
    data = None
    if stage in ("chunked", "decoding"):
        scheduler.normal_loop()
    elif stage == "inflight":
        inputs = scheduler._schedule_next_batch()
        data = (inputs, scheduler._forward(inputs))
    scheduler._process_one_msg(AbortBackendMsg(7))
    terminal = scheduler.replies[-1]
    assert terminal == DetokenizeMsg(7, None, True)
    count = len(scheduler.replies)
    scheduler._process_last_data(data)
    assert len(scheduler.replies) == count
    assert len(scheduler.released) == (stage != "pending")
    assert_idle(scheduler)


def test_offline_errors_are_reported_after_valid_requests_are_drained(scheduler):
    # Exercise LLM.generate with the same real scheduler managers and fake compute.
    scheduler.__class__ = LLM
    scheduler.tokenizer = tokenizer()
    scheduler.receive_msg = scheduler.offline_receive_msg
    scheduler.send_result = scheduler.offline_send_result

    def run():
        for _ in range(10):
            scheduler.normal_loop()
        pytest.fail("Offline generation did not terminate")

    scheduler.run_forever = run
    with pytest.raises(ValueError, match="at least one token"):
        scheduler.generate([[], [1, 2]], SamplingParams(max_tokens=2))
    assert scheduler.status_map[1].output_ids == [1, 1]
    assert_idle(scheduler)
    assert scheduler.generate([[1]], SamplingParams(max_tokens=0)) == [
        {"text": "", "token_ids": []}
    ]
    assert_idle(scheduler)


def frontend():
    from minisgl.server.api_server import FrontendManager

    sent = []

    async def put(msg):
        sent.append(msg)

    state = FrontendManager(None, SimpleNamespace(put=put), None, initialized=True)
    state.sent = sent
    return state


def enqueue(state, uid, *replies):
    state.ack_map[uid].extend(replies)
    state.event_map[uid].set()


def test_frontend_removes_terminal_state_before_consumer_breaks():
    async def run():
        state = frontend()
        uid = state.new_user()
        enqueue(state, uid, UserReply(uid, "", True, error="invalid prompt"))
        replies = state.wait_for_ack(uid)
        assert (await anext(replies)).error == "invalid prompt"
        assert state.ack_map == state.event_map == {}
        await replies.aclose()
        assert not state.sent

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_frontend_early_close_or_cancel_aborts_and_clears_state(cancel):
    async def run():
        state = frontend()
        uid = state.new_user()
        stream = state.stream_chat_completions(uid)
        if cancel:
            task = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            enqueue(state, uid, UserReply(uid, "hello", False))
            await anext(stream)
        await stream.aclose()
        assert state.ack_map == state.event_map == {}
        assert state.sent == [AbortMsg(uid)]

    asyncio.run(run())


@pytest.mark.parametrize("method", ["stream_generate", "stream_chat_completions"])
def test_frontend_streams_terminal_errors_without_hanging(method):
    async def run():
        state = frontend()
        uid = state.new_user()
        enqueue(state, uid, UserReply(uid, "", True, error="prompt too long"))
        output = b"".join([chunk async for chunk in getattr(state, method)(uid)])
        assert b'"type": "invalid_request_error"' in output
        assert b"prompt too long" in output and b"[DONE]" in output
        assert b'"finish_reason": "stop"' not in output
        assert state.ack_map == state.event_map == {}
        assert not state.sent

    asyncio.run(run())


def test_valid_stream_preserves_each_increment_and_finishes_once():
    async def run():
        state = frontend()
        uid = state.new_user()
        enqueue(
            state,
            uid,
            UserReply(uid, "hello", False),
            UserReply(uid, " world", False),
            UserReply(uid, "", True),
        )
        chunks = [chunk async for chunk in state.stream_chat_completions(uid)]
        assert chunks[-1] == b"data: [DONE]\n\n"
        payloads = [json.loads(chunk[6:]) for chunk in chunks[:-1]]
        choices = [payload["choices"][0] for payload in payloads]
        assert "".join(choice["delta"].get("content", "") for choice in choices) == "hello world"
        assert choices[0]["delta"]["role"] == "assistant"
        assert [choice["finish_reason"] for choice in choices].count("stop") == 1
        assert state.ack_map == state.event_map == {}
        assert not state.sent

    asyncio.run(run())


def test_disconnect_wrapper_closes_inner_stream_and_aborts():
    async def run():
        state = frontend()
        uid = state.new_user()
        enqueue(state, uid, UserReply(uid, "hello", False))

        async def disconnected():
            return True

        stream = state.stream_with_cancellation(
            state.stream_generate(uid), SimpleNamespace(is_disconnected=disconnected), uid
        )
        with pytest.raises(asyncio.CancelledError):
            await anext(stream)
        assert state.ack_map == state.event_map == {}
        assert state.sent == [AbortMsg(uid)]

    asyncio.run(run())


def test_nonstreaming_api_surfaces_scheduler_errors(monkeypatch):
    from fastapi import HTTPException
    from minisgl.server import api_server

    async def run():
        state = frontend()

        async def send(msg):
            enqueue(state, msg.uid, UserReply(msg.uid, "", True, error="prompt too long"))

        state.send_one = send
        monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
        with pytest.raises(HTTPException) as error:
            await api_server.v1_completions(
                api_server.OpenAICompletionRequest(model="tiny-kimi", prompt="x"), None
            )
        assert error.value.status_code == 400
        assert error.value.detail == "prompt too long"
        assert state.ack_map == state.event_map == {}

    asyncio.run(run())


def test_missing_prompt_is_http_error_without_creating_request(monkeypatch):
    from fastapi import HTTPException
    from minisgl.server import api_server

    async def run():
        state = frontend()
        monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
        with pytest.raises(HTTPException) as error:
            await api_server.v1_completions(
                api_server.OpenAICompletionRequest(model="tiny-kimi"), None
            )
        assert error.value.status_code == 400
        assert state.ack_map == state.event_map == {}
        assert not state.sent

    asyncio.run(run())


def test_tokenizer_worker_rejects_only_bad_request_and_cleans_abort(monkeypatch):
    import minisgl.tokenizer.server as server

    sent = {"backend": [], "frontend": []}
    tok = tokenizer()

    def encode(text, **kwargs):
        if text == "invalid":
            raise ValueError("invalid prompt")
        return torch.tensor([[1]])

    tok.encode = encode
    monkeypatch.setattr(server, "load_tokenizer", lambda path: tok)
    monkeypatch.setattr(
        server, "ZmqPushQueue", lambda addr, **kwargs: SimpleNamespace(put=sent[addr].append)
    )
    batches = [
        DetokenizeMsg(7, 1, False),
        AbortMsg(7),
        TokenizeMsg(8, "invalid", SamplingParams()),
        TokenizeMsg(9, "valid", SamplingParams()),
    ]

    def get():
        if not batches:
            raise KeyboardInterrupt
        return batches.pop(0)

    monkeypatch.setattr(
        server, "ZmqPullQueue", lambda *args, **kwargs: SimpleNamespace(get=get, empty=lambda: True)
    )
    instances = []
    original_init = DetokenizeManager.__init__

    def init(manager, tok):
        original_init(manager, tok)
        instances.append(manager)

    monkeypatch.setattr(DetokenizeManager, "__init__", init)
    server.tokenize_worker(
        tokenizer_path="unused",
        addr="unused",
        create=False,
        backend_addr="backend",
        frontend_addr="frontend",
        local_bs=1,
    )
    assert instances[0].decode_map == {}
    assert sent["frontend"][1] == UserReply(8, "", True, error="invalid prompt")
    assert sent["backend"][0] == AbortBackendMsg(7)
    assert isinstance(sent["backend"][1], UserMsg)
    assert sent["backend"][1].uid == 9
