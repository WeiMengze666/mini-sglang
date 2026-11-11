"""The public reporting switches control existing API output."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from minisgl.message import UserReply
from minisgl.server import api_server as api


@pytest.mark.parametrize("enabled", [False, True])
def test_metrics_flag_controls_endpoint(monkeypatch, enabled):
    state = SimpleNamespace(
        config=SimpleNamespace(enable_metrics=enabled),
        uid_counter=1, completed_requests=1, failed_requests=0, cancelled_requests=0,
        completed_prompt_tokens=3, completed_output_tokens=2,
        completed_cached_prompt_tokens=1, event_map={},
    )
    monkeypatch.setattr(api, "_GLOBAL_STATE", state)
    if not enabled:
        with pytest.raises(HTTPException) as error:
            asyncio.run(api.metrics())
        assert error.value.status_code == 404
    else:
        assert "minisgl:completed_cached_prompt_tokens_total 1" in asyncio.run(api.metrics()).body.decode()


@pytest.mark.parametrize("enabled", [False, True])
def test_cache_report_flag_controls_usage_without_changing_internal_accounting(enabled):
    async def run():
        manager = object.__new__(api.FrontendManager)
        manager.config = SimpleNamespace(enable_cache_report=enabled)
        event = asyncio.Event()
        event.set()
        original = UserReply(7, "ok", True, prompt_tokens=3, completion_tokens=2, cached_tokens=1)
        manager.event_map = {7: event}
        manager.ack_map = {7: [original]}
        manager.completed_requests = manager.completed_prompt_tokens = 0
        manager.completed_output_tokens = manager.completed_cached_prompt_tokens = 0
        manager.failed_requests = 0
        replies = [ack async for ack in manager.wait_for_ack(7)]
        assert manager.completed_cached_prompt_tokens == 1 and original.cached_tokens == 1
        assert ("prompt_tokens_details" in replies[0].usage) == enabled
    asyncio.run(run())
