"""WebSocket /v1/speech/ws protocol tests (fake provider, no model)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import get_settings
from app.core.engine import TTSEngine
from app.main import app
from tests.test_session import FakeStreamProvider

URL = "/v1/speech/ws"
START = {"type": "start"}


def _install(provider: FakeStreamProvider) -> None:
    app.state.engine = TTSEngine(
        {"fake": provider}, default_provider="fake", default_voice="v1"
    )


def _recv(ws) -> tuple[str, object]:
    """('json', dict) | ('bytes', bytes) | ('close', code)."""
    message = ws.receive()
    if message["type"] == "websocket.close":
        return "close", message.get("code")
    if message.get("text") is not None:
        return "json", json.loads(message["text"])
    return "bytes", message["bytes"]


def _recv_until(ws, predicate, limit: int = 200) -> list[tuple[str, object]]:
    out = []
    for _ in range(limit):
        item = _recv(ws)
        out.append(item)
        if predicate(item):
            return out
    raise AssertionError(f"condition not met in {limit} messages: {out[-5:]}")


def _is(kind: str):
    return lambda item: item[0] == "json" and item[1]["type"] == kind


def test_full_turn_streams_audio_before_the_flush():
    provider = FakeStreamProvider()
    with TestClient(app) as client:
        _install(provider)
        with client.websocket_connect(URL) as ws:
            ws.send_json(START)
            kind, ready = _recv(ws)
            assert kind == "json" and ready["type"] == "ready"
            assert (ready["sample_rate"], ready["channels"], ready["sample_width"]) == (24000, 1, 2)
            assert (ready["format"], ready["provider"], ready["voice"]) == ("pcm", "fake", "v1")

            ws.send_json({"type": "text", "text": "Hello there. How"})
            first = _recv_until(ws, lambda item: item == ("bytes", b"Hello there.#1"))
            assert first[0] == ("json", {"type": "sentence", "index": 0, "turn": 0,
                                         "chars": 12, "cached": False})

            ws.send_json({"type": "text", "text": " are you?"})
            ws.send_json({"type": "flush"})
            rest = _recv_until(ws, _is("done"))
    assert rest[-1] == ("json", {"type": "done", "turn": 0})
    assert ("bytes", b"How are you?#1") in rest


def test_unknown_voice_at_start_is_refused():
    with TestClient(app) as client:
        _install(FakeStreamProvider())
        with client.websocket_connect(URL) as ws:
            ws.send_json({"type": "start", "voice": "C:/evil.pt"})
            kind, error = _recv(ws)
            assert error["type"] == "error" and error["status"] == 400
            assert "Unknown voice" in error["detail"]
            assert _recv(ws) == ("close", 1008)


@pytest.mark.parametrize(
    "first",
    [{"type": "text", "text": "hi"}, {"type": "start", "speed": 99}, {"no": "type"}],
)
def test_a_bad_first_message_is_refused(first):
    with TestClient(app) as client:
        _install(FakeStreamProvider())
        with client.websocket_connect(URL) as ws:
            ws.send_json(first)
            kind, error = _recv(ws)
            assert error["type"] == "error" and error["status"] == 400
            assert _recv(ws) == ("close", 1008)


def test_bad_messages_after_start_get_an_error_but_keep_the_session():
    with TestClient(app) as client:
        _install(FakeStreamProvider())
        with client.websocket_connect(URL) as ws:
            ws.send_json(START)
            _recv(ws)  # ready
            for bad in ("not json", '{"type": "dance"}', '{"type": "text", "text": 5}',
                        '{"type": "start"}'):
                ws.send_text(bad)
                kind, error = _recv(ws)
                assert error["type"] == "error" and error["status"] == 400
            ws.send_bytes(b"\x00\x01")
            assert _recv(ws)[1]["type"] == "error"

            ws.send_json({"type": "text", "text": "Still working."})
            ws.send_json({"type": "flush"})
            assert _recv_until(ws, _is("done"))[-1][1]["turn"] == 0


def test_cancel_stops_speech_and_the_next_turn_works():
    provider = FakeStreamProvider(chunks=50, delay=0.02)
    with TestClient(app) as client:
        _install(provider)
        with client.websocket_connect(URL) as ws:
            ws.send_json(START)
            _recv(ws)
            ws.send_json({"type": "text", "text": "One. Two. Three."})
            ws.send_json({"type": "flush"})
            _recv_until(ws, lambda item: item[0] == "bytes")
            ws.send_json({"type": "cancel"})
            before_cancel = _recv_until(ws, _is("cancelled"))
            assert not any(item[0] == "json" and item[1]["type"] == "done"
                           for item in before_cancel)

            provider.chunks, provider.delay = 1, 0.0
            ws.send_json({"type": "text", "text": "Four."})
            ws.send_json({"type": "flush"})
            after = _recv_until(ws, _is("done"))
    assert [item[1] for item in after if item[0] == "bytes"] == [b"Four.#0"]
    assert provider.started == ["One.", "Four."]


def test_provider_failure_reports_an_error_for_the_turn():
    with TestClient(app) as client:
        _install(FakeStreamProvider(fail_on="Bad"))
        with client.websocket_connect(URL) as ws:
            ws.send_json(START)
            _recv(ws)
            ws.send_json({"type": "text", "text": "Bad sentence."})
            ws.send_json({"type": "flush"})
            error = _recv_until(ws, _is("error"))[-1][1]
            assert (error["turn"], error["status"], error["kind"]) == (0, 504, "timeout")

            ws.send_json({"type": "text", "text": "Good one."})
            ws.send_json({"type": "flush"})
            assert _recv_until(ws, _is("done"))[-1][1]["turn"] == 1


def test_overlong_turn_gets_an_error(monkeypatch):
    monkeypatch.setenv("WS_MAX_TURN_CHARS", "100")
    get_settings.cache_clear()
    with TestClient(app) as client:
        _install(FakeStreamProvider())
        with client.websocket_connect(URL) as ws:
            ws.send_json(START)
            _recv(ws)
            ws.send_json({"type": "text", "text": "x" * 150})
            kind, error = _recv(ws)
            assert error["type"] == "error" and "longer than 100" in error["detail"]


def test_browser_origins_are_refused_unless_allowed(monkeypatch):
    with TestClient(app) as client:
        _install(FakeStreamProvider())
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(URL, headers={"origin": "https://evil.example"}):
                pass
        assert refused.value.code == 1008

    monkeypatch.setenv("WS_ALLOWED_ORIGINS", "http://localhost:7860, https://app.example")
    get_settings.cache_clear()
    with TestClient(app) as client:
        _install(FakeStreamProvider())
        with client.websocket_connect(URL, headers={"origin": "http://localhost:7860"}) as ws:
            ws.send_json(START)
            assert _recv(ws)[1]["type"] == "ready"


def test_cached_sentences_are_flagged(tmp_path):
    from app.core.cache import FileCache

    provider = FakeStreamProvider()
    with TestClient(app) as client:
        app.state.engine = TTSEngine({"fake": provider}, default_provider="fake",
                                     default_voice="v1",
                                     cache=FileCache(tmp_path / "cache", 10_000_000))
        with client.websocket_connect(URL) as ws:
            ws.send_json(START)
            _recv(ws)
            flags = []
            for _ in range(2):
                ws.send_json({"type": "text", "text": "Hello there."})
                ws.send_json({"type": "flush"})
                items = _recv_until(ws, _is("done"))
                flags += [i[1]["cached"] for i in items if _is("sentence")(i)]
    assert flags == [False, True]
    assert provider.started == ["Hello there."]
