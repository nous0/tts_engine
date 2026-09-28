"""Tests for the on-disk audio cache and its use by the engine and /v1/speech."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from app.core import cache as cache_mod
from app.core.cache import FileCache
from app.core.engine import TTSEngine
from app.core.providers.base import SynthOpts, Voice
from app.main import app


class CountingProvider:
    name = "counting"

    def __init__(self, model: str = "m1") -> None:
        self.calls = 0
        self.model = model

    def cache_tag(self, opts: SynthOpts) -> str:
        return f"counting:{opts.model or self.model}"

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        self.calls += 1
        return f"{text}|{voice}|{opts.format}|{self.calls}".encode()

    async def synthesize_stream(self, text: str, voice: str, opts: SynthOpts):
        yield b"x"

    async def list_voices(self) -> list[Voice]:
        return [Voice(id=v, name=v, provider=self.name) for v in ("v1", "v2")]


def _engine(tmp_path, provider=None, max_bytes: int = 10_000_000) -> tuple[TTSEngine, object]:
    provider = provider or CountingProvider()
    engine = TTSEngine(
        {"counting": provider},
        default_provider="counting",
        default_voice="v1",
        cache=FileCache(tmp_path / "cache", max_bytes),
    )
    return engine, provider


async def test_second_identical_request_is_served_from_cache(tmp_path):
    engine, provider = _engine(tmp_path)
    first, hit1 = await engine.synthesize_cached("Hello", opts=SynthOpts(format="wav"))
    second, hit2 = await engine.synthesize_cached("Hello", opts=SynthOpts(format="wav"))
    assert (hit1, hit2) == (False, True)
    assert first == second
    assert provider.calls == 1


@pytest.mark.parametrize(
    "change",
    [
        {"text": "Hello!"},
        {"voice": "v2"},
        {"opts": SynthOpts(format="pcm")},
        {"opts": SynthOpts(format="wav", speed=1.5)},
        {"opts": SynthOpts(format="wav", instructions="cheerful")},
        {"opts": SynthOpts(format="wav", model="m2")},
    ],
)
async def test_anything_that_shapes_the_audio_changes_the_key(tmp_path, change):
    engine, provider = _engine(tmp_path)
    base = {"text": "Hello", "voice": "v1", "opts": SynthOpts(format="wav")}
    await engine.synthesize_cached(**base)
    _, hit = await engine.synthesize_cached(**{**base, **change})
    assert hit is False
    assert provider.calls == 2


async def test_changing_the_providers_default_model_misses(tmp_path):
    engine, _ = _engine(tmp_path, CountingProvider(model="m1"))
    await engine.synthesize_cached("Hello")
    # Same inputs, but the server now runs another model (e.g. OPENAI_TTS_MODEL changed).
    engine2, provider2 = _engine(tmp_path, CountingProvider(model="m2"))
    _, hit = await engine2.synthesize_cached("Hello")
    assert hit is False and provider2.calls == 1


async def test_default_voice_is_resolved_before_keying(tmp_path):
    engine, provider = _engine(tmp_path)
    await engine.synthesize_cached("Hello")  # no voice -> default v1
    _, hit = await engine.synthesize_cached("Hello", voice="v1")
    assert hit is True and provider.calls == 1


async def test_cache_survives_a_new_engine_like_a_restart(tmp_path):
    engine, _ = _engine(tmp_path)
    await engine.synthesize_cached("Hello")
    engine2, provider2 = _engine(tmp_path)
    _, hit = await engine2.synthesize_cached("Hello")
    assert hit is True and provider2.calls == 0


async def test_leftover_temp_files_are_never_served(tmp_path):
    engine, provider = _engine(tmp_path)
    await engine.synthesize_cached("Hello")
    cache_dir = tmp_path / "cache"
    (entry,) = list(cache_dir.iterdir())
    # Simulate a crash mid-write of *another* version: only a truncated .tmp exists.
    entry.rename(cache_dir / (entry.name + ".abc.tmp"))
    _, hit = await engine.synthesize_cached("Hello")
    assert hit is False and provider.calls == 2


async def test_prune_evicts_least_recently_used_first(tmp_path):
    cache = FileCache(tmp_path / "cache", max_bytes=1000)
    for i, key in enumerate(["old", "mid", "new"]):
        await cache.put(key, "wav", b"0123456789")
        os.utime(tmp_path / "cache" / f"{key}.wav", (1000 + i, 1000 + i))
    assert await cache.get("old", "wav") is not None  # a hit refreshes "old"

    cache._max_bytes = 25  # now only two of the three 10-byte entries fit
    await cache.prune()
    remaining = sorted(p.name for p in (tmp_path / "cache").iterdir())
    assert remaining == ["new.wav", "old.wav"]


async def test_prune_removes_stale_temp_files(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale = cache_dir / "k.wav.x.tmp"
    stale.write_bytes(b"partial")
    two_hours_ago = time.time() - 7200
    os.utime(stale, (two_hours_ago, two_hours_ago))
    await FileCache(cache_dir, 1000).prune()
    assert not stale.exists()


async def test_put_over_the_limit_triggers_prune(tmp_path):
    cache = FileCache(tmp_path / "cache", max_bytes=15)
    await cache.put("a", "wav", b"0123456789")
    os.utime(tmp_path / "cache" / "a.wav", (1000, 1000))
    await cache.put("b", "wav", b"0123456789")
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["b.wav"]


async def test_replace_refused_by_windows_does_not_fail_the_request(tmp_path, monkeypatch):
    def locked(src, dst):
        raise PermissionError("file is being used by another process")

    monkeypatch.setattr(cache_mod.os, "replace", locked)
    engine, provider = _engine(tmp_path)
    audio, hit = await engine.synthesize_cached("Hello")
    assert audio and hit is False
    assert not any((tmp_path / "cache").glob("*.tmp"))  # temp file cleaned up


async def test_unwritable_cache_dir_still_serves_requests(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the cache dir should be")
    provider = CountingProvider()
    engine = TTSEngine(
        {"counting": provider},
        default_provider="counting",
        default_voice="v1",
        cache=FileCache(blocker / "cache", 1_000_000),
    )
    for _ in range(2):
        audio, hit = await engine.synthesize_cached("Hello")
        assert audio and hit is False
    assert provider.calls == 2


async def test_no_cache_always_calls_the_provider(tmp_path):
    provider = CountingProvider()
    engine = TTSEngine({"counting": provider}, default_provider="counting", default_voice="v1")
    for _ in range(2):
        _, hit = await engine.synthesize_cached("Hello")
        assert hit is False
    assert provider.calls == 2


def test_speech_endpoint_reports_cache_hits(tmp_path):
    with TestClient(app) as client:
        app.state.engine, provider = _engine(tmp_path)
        first = client.post("/v1/speech", json={"text": "Hello"})
        second = client.post("/v1/speech", json={"text": "Hello"})
    assert (first.headers["X-Cache"], second.headers["X-Cache"]) == ("miss", "hit")
    assert first.content == second.content
    assert provider.calls == 1
