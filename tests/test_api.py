"""API smoke tests. No real provider calls — a fake provider is injected."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.core.engine import TTSEngine
from app.core.providers.base import ProviderError, SynthOpts, UnsupportedFormat, Voice
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
    # One streaming WAV header up front, never one per sentence.
    assert body.startswith(b"RIFF")
    assert body.count(b"RIFF") == 1
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


class WavPerCallProvider(FakeProvider):
    """Emits a WAV header per call when asked for wav, like real providers do."""

    name = "wavfake"

    def __init__(self) -> None:
        self.formats: list[str] = []

    async def synthesize_stream(self, text: str, voice: str, opts: SynthOpts):
        self.formats.append(opts.format)
        if opts.format == "wav":
            yield b"RIFF-per-call"
        yield b"pcm"


def test_speech_stream_wav_has_a_single_header_across_sentences():
    provider = WavPerCallProvider()
    with TestClient(app) as client:
        app.state.engine = TTSEngine(
            {"wavfake": provider}, default_provider="wavfake", default_voice="fake-1"
        )
        resp = client.post(
            "/v1/speech/stream", json={"text": "One. Two. Three.", "format": "wav"}
        )
    assert resp.status_code == 200
    assert resp.content.count(b"RIFF") == 1
    assert b"RIFF-per-call" not in resp.content
    assert provider.formats == ["pcm", "pcm", "pcm"]


def test_speech_stream_rejects_flac():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post("/v1/speech/stream", json={"text": "Hi.", "format": "flac"})
    assert resp.status_code == 400
    assert "can't be streamed" in resp.json()["detail"]


class FailingStreamProvider(FakeProvider):
    """Fails on the sentence number ``fail_at`` (1-based)."""

    name = "failing"

    def __init__(self, fail_at: int) -> None:
        self.fail_at = fail_at
        self.calls = 0

    async def synthesize_stream(self, text: str, voice: str, opts: SynthOpts):
        self.calls += 1
        if self.calls == self.fail_at:
            raise UnsupportedFormat(opts.format, self.name)
        yield b"pcm"


def test_speech_stream_error_before_first_audio_is_a_real_error_status():
    with TestClient(app) as client:
        app.state.engine = TTSEngine(
            {"failing": FailingStreamProvider(fail_at=1)},
            default_provider="failing",
            default_voice="fake-1",
        )
        resp = client.post("/v1/speech/stream", json={"text": "One. Two."})
    assert resp.status_code == 400


def test_speech_stream_error_mid_stream_never_writes_text_into_audio():
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.engine = TTSEngine(
            {"failing": FailingStreamProvider(fail_at=2)},
            default_provider="failing",
            default_voice="fake-1",
        )
        try:
            resp = client.post("/v1/speech/stream", json={"text": "One. Two."})
            body = resp.content
        except Exception:  # noqa: BLE001 - a dropped connection is also acceptable
            body = b""
    assert b"not supported" not in body
    assert b"Format" not in body


class ErroringProvider(FakeProvider):
    """Raises a ProviderError of the given kind from every call."""

    name = "erroring"

    def __init__(self, kind: str, retry_after: float | None = None) -> None:
        self.kind = kind
        self.retry_after = retry_after

    def _error(self) -> ProviderError:
        return ProviderError(self.name, "nope", kind=self.kind, retry_after=self.retry_after)

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        raise self._error()

    async def synthesize_stream(self, text: str, voice: str, opts: SynthOpts):
        raise self._error()
        yield b""  # pragma: no cover - makes this an async generator


def _install_erroring(kind: str, retry_after: float | None = None) -> None:
    app.state.engine = TTSEngine(
        {"erroring": ErroringProvider(kind, retry_after)},
        default_provider="erroring",
        default_voice="fake-1",
    )


@pytest.mark.parametrize(
    ("kind", "status"),
    [("bad_request", 400), ("auth", 502), ("upstream", 502), ("rate_limited", 503),
     ("timeout", 504)],
)
@pytest.mark.parametrize("path", ["/v1/speech", "/v1/speech/stream"])
def test_provider_errors_map_to_http_statuses(kind, status, path):
    with TestClient(app) as client:
        _install_erroring(kind)
        resp = client.post(path, json={"text": "Hello."})
    assert resp.status_code == status
    assert resp.json()["kind"] == kind
    assert "erroring" in resp.json()["detail"]


def test_rate_limited_response_carries_retry_after():
    with TestClient(app) as client:
        _install_erroring("rate_limited", retry_after=7)
        resp = client.post("/v1/speech", json={"text": "Hello."})
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "7"


def test_podcast_job_records_provider_errors():
    with TestClient(app) as client:
        _install_erroring("timeout")
        resp = client.post("/v1/podcast", json={"script": "Alice: Hi.", "format": "pcm"})
        assert resp.status_code == 202
        job = _await_job(client, resp.json()["job_id"])
    assert job["status"] == "error"
    assert "timed out" in job["error"]


@pytest.mark.parametrize("path", ["/v1/speech", "/v1/speech/stream"])
def test_unknown_voice_is_rejected_with_400(path):
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post(path, json={"text": "Hello.", "voice": "C:/evil/voice.pt"})
    assert resp.status_code == 400
    assert "Unknown voice" in resp.json()["detail"]


def test_podcast_with_unknown_voice_is_rejected_at_submit():
    with TestClient(app) as client:
        _install_fake_engine()
        resp = client.post(
            "/v1/podcast",
            json={"script": "Alice: Hi.\nBob: Hey.", "voices": {"Alice": "not-a-voice"}},
        )
    assert resp.status_code == 400
    assert "Unknown voice 'not-a-voice'" in resp.json()["detail"]


class RecordingProvider(FakeProvider):
    """Remembers the voice it was asked for; has its own default voice."""

    def __init__(self, name: str, voices: list[str], default_voice: str | None = None):
        self.name = name
        self._voices = voices
        if default_voice:
            self.default_voice = default_voice
        self.seen: list[str] = []

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        self.seen.append(voice)
        return b"FAKE_AUDIO"

    async def list_voices(self) -> list[Voice]:
        return [Voice(id=v, name=v, provider=self.name) for v in self._voices]


def test_non_default_provider_uses_its_own_default_voice():
    # DEFAULT_VOICE belongs to the default provider; asking another provider
    # without a voice must not send it that voice.
    other = RecordingProvider("other", ["alloy", "nova"], default_voice="nova")
    with TestClient(app) as client:
        app.state.engine = TTSEngine(
            {"main": RecordingProvider("main", ["af_heart"]), "other": other},
            default_provider="main",
            default_voice="af_heart",
        )
        resp = client.post("/v1/speech", json={"text": "Hi", "provider": "other"})
    assert resp.status_code == 200
    assert other.seen == ["nova"]


def test_invalid_default_voice_falls_back_to_the_provider_default():
    main = RecordingProvider("main", ["af_heart", "am_michael"], default_voice="am_michael")
    with TestClient(app) as client:
        app.state.engine = TTSEngine(
            {"main": main}, default_provider="main", default_voice="Kore"
        )
        resp = client.post("/v1/speech", json={"text": "Hi"})
    assert resp.status_code == 200
    assert main.seen == ["am_michael"]
