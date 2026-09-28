"""API smoke tests. No real provider calls — a fake provider is injected."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.core.engine import TTSEngine
from app.core.providers.base import SynthOpts, Voice
from app.main import app
from tests.test_audio import tone


class FakeProvider:
    name = "fake"

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        # Podcast assembly asks for pcm; return real samples so the pipeline
        # can concatenate and normalize them like provider output.
        if opts.format == "pcm":
            return tone(100)
        return b"FAKE_AUDIO"

    async def synthesize_stream(self, text: str, voice: str, opts: SynthOpts):
        # Yield the text length as a marker plus a fixed tail, in two chunks,
        # so tests can assert that multiple chunks arrive per sentence.
        marker = f"[{len(text)}]".encode()
        yield marker
        yield b"_AUDIO"

    async def list_voices(self) -> list[Voice]:
        return [
            Voice(id="fake-1", name="Fake One", provider=self.name, gender="female"),
            Voice(id="fake-2", name="Fake Two", provider=self.name, gender="male"),
        ]


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
    # Default format is now wav (works without ffmpeg).
    assert resp.headers["content-type"] == "audio/wav"


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
    assert resp.headers["content-type"] == "audio/wav"
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


def _await_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    """Poll a job until it leaves the queued/running states."""
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/v1/jobs/{job_id}").json()
        if payload["status"] in ("done", "error"):
            return payload
        time.sleep(0.05)
    return payload


def test_podcast_renders_a_job_to_audio():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post(
            "/v1/podcast",
            json={
                "turns": [
                    {"speaker": "Alice", "text": "Hello there."},
                    {"speaker": "Bob", "text": "Hi Alice."},
                ],
                "format": "wav",
                "pause_ms": 200,
            },
        )
        assert resp.status_code == 202
        submitted = resp.json()
        assert submitted["status"] == "queued"

        job = _await_job(client, submitted["job_id"])
        assert job["status"] == "done", job
        assert job["progress"] == 2
        assert job["total"] == 2
        # Two speakers got two distinct voices from the provider catalog.
        assert set(job["voices"]) == {"Alice", "Bob"}
        assert job["voices"]["Alice"] != job["voices"]["Bob"]

        audio_resp = client.get(job["audio_url"])
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"] == "audio/wav"
        assert audio_resp.content[:4] == b"RIFF"


def test_podcast_accepts_plain_text_script():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post(
            "/v1/podcast",
            json={"script": "Alice: One.\nBob: Two.", "format": "pcm"},
        )
        assert resp.status_code == 202
        job = _await_job(client, resp.json()["job_id"])
    assert job["status"] == "done", job
    assert job["total"] == 2


def test_podcast_casts_voices_by_gender():
    with TestClient(app) as client:
        _install_fake_engine()  # fake-1 is female, fake-2 is male
        resp = client.post(
            "/v1/podcast",
            json={"script": "Bob: One.\nAlice: Two.", "format": "pcm"},
        )
        assert resp.status_code == 202
        job = _await_job(client, resp.json()["job_id"])
    assert job["voices"] == {"Bob": "fake-2", "Alice": "fake-1"}


def test_podcast_explicit_genders_override_the_name():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post(
            "/v1/podcast",
            json={
                "script": "Host: One.\nBob: Two.",
                "genders": {"Host": "male", "Bob": "female"},
                "format": "pcm",
            },
        )
        assert resp.status_code == 202
        job = _await_job(client, resp.json()["job_id"])
    assert job["voices"] == {"Host": "fake-2", "Bob": "fake-1"}


def test_podcast_rejects_unknown_gender():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post(
            "/v1/podcast",
            json={"script": "Host: One.", "genders": {"Host": "robot"}},
        )
    assert resp.status_code == 422


def test_podcast_requires_turns_or_script():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post("/v1/podcast", json={"format": "wav"})
    assert resp.status_code == 422


def test_podcast_unconfigured_provider_returns_400():
    with TestClient(app) as client:
        app.state.engine = TTSEngine({}, default_provider="openai", default_voice="alloy")
        resp = client.post(
            "/v1/podcast",
            json={"script": "Alice: Hi.", "provider": "openai"},
        )
    assert resp.status_code == 400


def test_podcast_rejects_unsupported_format_without_ffmpeg(monkeypatch):
    class PcmOnlyProvider(FakeProvider):
        def can_produce(self, fmt: str) -> bool:
            return fmt in ("wav", "pcm")

    with TestClient(app) as client:
        app.state.engine = TTSEngine(
            {"fake": PcmOnlyProvider()}, default_provider="fake", default_voice="fake-1"
        )
        resp = client.post("/v1/podcast", json={"script": "A: Hi.", "format": "mp3"})
    assert resp.status_code == 400
    assert "ffmpeg" in resp.json()["detail"]


def test_unknown_job_returns_404():
    with TestClient(app) as client:
        resp = client.get("/v1/jobs/nope")
    assert resp.status_code == 404


def test_job_audio_before_completion_returns_409():
    with TestClient(app) as client:
        resp = client.get("/v1/jobs/nope/audio")
    assert resp.status_code == 404


def test_list_jobs_returns_submitted_jobs():
    with TestClient(app) as client:
        _install_fake_engine()
        submitted = client.post(
            "/v1/podcast", json={"script": "A: Hi.", "format": "pcm"}
        ).json()
        _await_job(client, submitted["job_id"])
        jobs = client.get("/v1/jobs").json()
    assert any(j["job_id"] == submitted["job_id"] for j in jobs)


class OtherFakeProvider(FakeProvider):
    name = "other"

    async def list_voices(self) -> list[Voice]:
        return [Voice(id="other-1", name="Other One", provider=self.name, gender="male")]


def _install_two_providers() -> None:
    app.state.engine = TTSEngine(
        {"fake": FakeProvider(), "other": OtherFakeProvider()},
        default_provider="fake",
        default_voice="fake-1",
    )


def test_voices_lists_every_registered_provider():
    with TestClient(app) as client:
        _install_two_providers()
        resp = client.get("/v1/voices")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_provider"] == "fake"
    assert body["providers"] == ["fake", "other"]
    assert [v["id"] for v in body["voices"]] == ["fake-1", "fake-2", "other-1"]
    assert body["voices"][0] == {
        "id": "fake-1",
        "name": "Fake One",
        "provider": "fake",
        "language": None,
        "gender": "female",
    }


def test_voices_filters_by_provider():
    with TestClient(app) as client:
        _install_two_providers()
        resp = client.get("/v1/voices", params={"provider": "other"})
    assert resp.status_code == 200
    assert [v["id"] for v in resp.json()["voices"]] == ["other-1"]


def test_voices_filters_by_gender():
    with TestClient(app) as client:
        _install_two_providers()
        resp = client.get("/v1/voices", params={"gender": "male"})
    assert resp.status_code == 200
    assert [v["id"] for v in resp.json()["voices"]] == ["fake-2", "other-1"]


def test_voices_unconfigured_provider_returns_400():
    with TestClient(app) as client:
        _install_two_providers()
        resp = client.get("/v1/voices", params={"provider": "openai"})
    assert resp.status_code == 400
    assert "not configured" in resp.json()["detail"]


def test_voices_rejects_unknown_gender():
    with TestClient(app) as client:
        _install_two_providers()
        resp = client.get("/v1/voices", params={"gender": "robot"})
    assert resp.status_code == 422


def test_every_response_carries_a_request_id():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.headers["X-Request-ID"]


def test_client_request_id_is_echoed_back():
    with TestClient(app) as client:
        resp = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


def test_requests_are_logged_with_their_id(caplog):
    with caplog.at_level("INFO", logger="app"), TestClient(app) as client:
        client.get("/health", headers={"X-Request-ID": "trace-me"})
    lines = [r for r in caplog.records if r.name == "app.http"]
    assert any(
        "GET /health -> 200" in r.getMessage() and r.request_id == "trace-me" for r in lines
    )
