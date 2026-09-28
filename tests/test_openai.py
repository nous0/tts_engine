"""OpenAI provider tests against a mocked HTTP transport (no network, no key).

The real SDK runs on top of ``httpx.MockTransport``, so these tests exercise its
actual retry behaviour and our translation of its errors into ``ProviderError``.
Error responses carry ``retry-after-ms: 1`` so the SDK's backoff doesn't slow the
suite down.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.providers.base import ProviderError, SynthOpts
from app.core.providers.openai import OpenAIProvider

FAST_RETRY = {"retry-after-ms": "1"}


def _provider(responses, max_retries: int = 2) -> tuple[OpenAIProvider, list[httpx.Request]]:
    """Provider whose HTTP calls are answered from ``responses`` in order.

    Each item is an ``httpx.Response`` or an exception to raise.
    """
    calls: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        api_key="sk-test", max_retries=max_retries, timeout=5.0, http_client=client
    )
    return provider, calls


def _error(status: int, message: str = "boom", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": message, "type": "test"}},
        headers={**FAST_RETRY, **(headers or {})},
    )


OK = httpx.Response(200, content=b"AUDIO", headers={"content-type": "audio/wav"})


async def test_success_returns_audio():
    provider, calls = _provider([OK])
    assert await provider.synthesize("hi", "alloy", SynthOpts()) == b"AUDIO"
    assert len(calls) == 1


async def test_server_errors_are_retried_then_succeed():
    provider, calls = _provider([_error(500), _error(502), OK])
    assert await provider.synthesize("hi", "alloy", SynthOpts()) == b"AUDIO"
    assert len(calls) == 3


async def test_persistent_server_errors_become_upstream_after_retries():
    provider, calls = _provider([_error(500)])
    with pytest.raises(ProviderError) as info:
        await provider.synthesize("hi", "alloy", SynthOpts())
    assert info.value.kind == "upstream"
    assert len(calls) == 3  # first try + 2 retries


async def test_bad_request_is_not_retried():
    provider, calls = _provider([_error(400, "Invalid voice")])
    with pytest.raises(ProviderError) as info:
        await provider.synthesize("hi", "nope", SynthOpts())
    assert info.value.kind == "bad_request"
    assert "Invalid voice" in str(info.value)
    assert len(calls) == 1


async def test_bad_api_key_is_auth_and_not_retried():
    provider, calls = _provider([_error(401, "Incorrect API key")])
    with pytest.raises(ProviderError) as info:
        await provider.synthesize("hi", "alloy", SynthOpts())
    assert info.value.kind == "auth"
    assert len(calls) == 1


async def test_rate_limit_carries_retry_after():
    provider, calls = _provider([_error(429, "slow down", {"retry-after": "7"})])
    with pytest.raises(ProviderError) as info:
        await provider.synthesize("hi", "alloy", SynthOpts())
    assert info.value.kind == "rate_limited"
    assert info.value.retry_after == 7.0
    assert len(calls) == 3  # 429 is retried by the SDK first


async def test_timeout_becomes_timeout_kind():
    provider, _ = _provider([httpx.ReadTimeout("too slow")], max_retries=0)
    with pytest.raises(ProviderError) as info:
        await provider.synthesize("hi", "alloy", SynthOpts())
    assert info.value.kind == "timeout"


async def test_connection_error_becomes_upstream():
    provider, _ = _provider([httpx.ConnectError("no route")], max_retries=0)
    with pytest.raises(ProviderError) as info:
        await provider.synthesize("hi", "alloy", SynthOpts())
    assert info.value.kind == "upstream"


async def test_stream_errors_are_translated_too():
    provider, _ = _provider([_error(401)])
    with pytest.raises(ProviderError) as info:
        async for _ in provider.synthesize_stream("hi", "alloy", SynthOpts()):
            pass
    assert info.value.kind == "auth"


async def test_stream_yields_audio_on_success():
    provider, _ = _provider([OK])
    chunks = [c async for c in provider.synthesize_stream("hi", "alloy", SynthOpts())]
    assert b"".join(chunks) == b"AUDIO"


def test_default_voice_is_configurable():
    assert OpenAIProvider(api_key="sk-test").default_voice == "alloy"
    assert OpenAIProvider(api_key="sk-test", default_voice="nova").default_voice == "nova"
