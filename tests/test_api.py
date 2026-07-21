"""API smoke tests for Phase 1. No real provider calls — a fake provider is injected."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.engine import TTSEngine
from app.core.providers.base import SynthOpts, Voice
from app.main import app


class FakeProvider:
    name = "fake"

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        return b"FAKE_AUDIO"

    async def synthesize_stream(self, text: str, voice: str, opts: SynthOpts):
        # Yield the text length as a marker plus a fixed tail, in two chunks,
        # so tests can assert that multiple chunks arrive per sentence.
        marker = f"[{len(text)}]".encode()
        yield marker
        yield b"_AUDIO"

    async def list_voices(self) -> list[Voice]:
        return [Voice(id="fake-1", name="Fake One", provider=self.name)]


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_speech_returns_audio():
    with TestClient(app) as client:
        app.state.engine = TTSEngine(
            {"fake": FakeProvider()}, default_provider="fake", default_voice="fake-1"
        )
        resp = client.post("/v1/speech", json={"text": "Hello world"})
    assert resp.status_code == 200
    assert resp.content == b"FAKE_AUDIO"
    assert resp.headers["content-type"] == "audio/mpeg"


def test_speech_unconfigured_provider_returns_400():
    with TestClient(app) as client:
        app.state.engine = TTSEngine({}, default_provider="openai", default_voice="alloy")
        resp = client.post("/v1/speech", json={"text": "Hello", "provider": "openai"})
    assert resp.status_code == 400


def test_speech_validation_rejects_empty_text():
    with TestClient(app) as client:
        resp = client.post("/v1/speech", json={"text": ""})
    assert resp.status_code == 422


def _install_fake_engine() -> None:
    app.state.engine = TTSEngine(
        {"fake": FakeProvider()}, default_provider="fake", default_voice="fake-1"
    )


def test_speech_stream_returns_chunked_audio():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post(
            "/v1/speech/stream",
            json={"text": "Hello world. How are you?"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    # Two sentences -> two synthesized chunks, each emitting a marker + tail.
    body = resp.content
    assert body.count(b"_AUDIO") == 2
    assert body.startswith(b"[")
    # The marker carries the byte length of each sentence chunk.
    assert b"[13]" in body or b"[12]" in body


def test_speech_stream_unconfigured_provider_returns_400():
    with TestClient(app) as client:
        app.state.engine = TTSEngine({}, default_provider="openai", default_voice="alloy")
        resp = client.post(
            "/v1/speech/stream", json={"text": "Hi", "provider": "openai"}
        )
    assert resp.status_code == 400


def test_speech_stream_rejects_empty_text():
    with TestClient(app) as client:
        resp = client.post("/v1/speech/stream", json={"text": ""})
    assert resp.status_code == 422
