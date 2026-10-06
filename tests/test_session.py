"""Tests for SpeechSession: incremental text in, sentence-by-sentence audio out."""

from __future__ import annotations

import asyncio

import pytest

from app.core.cache import FileCache
from app.core.engine import InvalidRequest, TTSEngine
from app.core.providers.base import ProviderError, SynthOpts, Voice
from app.core.session import (
    Audio,
    Cancelled,
    SentenceStarted,
    SpeechSession,
    TurnDone,
    TurnFailed,
)

PCM = SynthOpts(format="pcm")


class FakeStreamProvider:
    """Streams ``chunks`` pieces per sentence; records what started/finished/closed."""

    name = "fake"

    def __init__(self, chunks: int = 2, delay: float = 0.0, fail_on: str | None = None):
        self.chunks = chunks
        self.delay = delay
        self.fail_on = fail_on
        self.started: list[str] = []
        self.completed: list[str] = []
        self.closed: list[str] = []

    async def synthesize(self, text, voice, opts):
        return b"".join([c async for c in self.synthesize_stream(text, voice, opts)])

    async def synthesize_stream(self, text, voice, opts):
        self.started.append(text)
        try:
            if self.fail_on and self.fail_on in text:
                raise ProviderError(self.name, "boom", kind="timeout")
            for i in range(self.chunks):
                await asyncio.sleep(self.delay)
                yield f"{text}#{i}".encode()
            self.completed.append(text)
        finally:
            self.closed.append(text)

    async def list_voices(self):
        return [Voice(id="v1", name="v1", provider=self.name)]


def _session(provider, tmp_path=None, **kwargs) -> SpeechSession:
    cache = FileCache(tmp_path / "cache", 10_000_000) if tmp_path else None
    engine = TTSEngine({"fake": provider}, default_provider="fake", default_voice="v1",
                       cache=cache)
    session = SpeechSession(engine, provider, "v1", PCM, **kwargs)
    session.start()
    return session


async def _collect_until(session: SpeechSession, stop, timeout: float = 5.0) -> list:
    """Collect events until one matches ``stop`` (a type or a predicate)."""
    events: list = []

    def done(event) -> bool:
        return isinstance(event, stop) if isinstance(stop, type) else stop(event)

    async def run() -> None:
        async for event in session.events():
            events.append(event)
            if done(event):
                return

    await asyncio.wait_for(run(), timeout)
    return events


async def test_sentences_are_spoken_before_the_turn_is_flushed():
    provider = FakeStreamProvider()
    session = _session(provider)
    session.feed("Hello there. How are")
    events = await _collect_until(session, lambda e: e == Audio(b"Hello there.#1"))
    assert provider.started == ["Hello there."]  # "How are" waits for more text
    assert not any(isinstance(e, TurnDone) for e in events)
    await session.close()


async def test_flush_speaks_the_rest_and_ends_the_turn():
    provider = FakeStreamProvider()
    session = _session(provider)
    session.feed("Hello there. How are")
    session.feed(" you")
    assert session.flush() == 0
    events = await _collect_until(session, TurnDone)
    assert events == [
        SentenceStarted(0, 0, len("Hello there."), False),
        Audio(b"Hello there.#0"),
        Audio(b"Hello there.#1"),
        SentenceStarted(1, 0, len("How are you"), False),
        Audio(b"How are you#0"),
        Audio(b"How are you#1"),
        TurnDone(0),
    ]
    await session.close()


async def test_turns_are_numbered():
    session = _session(FakeStreamProvider())
    for expected in range(3):
        session.feed("Hi.")
        assert session.flush() == expected
        assert (await _collect_until(session, TurnDone))[-1] == TurnDone(expected)
    await session.close()


async def test_fast_start_ends_the_first_chunk_at_a_comma():
    provider = FakeStreamProvider()
    session = _session(provider)  # fast_start is on by default for sessions
    session.feed("Here is a first clause long enough, and the rest of it.")
    session.flush()
    await _collect_until(session, TurnDone)
    assert provider.started == ["Here is a first clause long enough,", "and the rest of it."]
    await session.close()


async def test_cancel_stops_the_current_sentence_and_drops_queued_ones():
    provider = FakeStreamProvider(chunks=50, delay=0.02)
    session = _session(provider)
    session.feed("One. Two. Three.")
    session.flush()
    await _collect_until(session, Audio)  # "One." is being spoken
    await session.cancel()

    events = await _collect_until(session, Cancelled)
    assert not any(isinstance(e, TurnDone) for e in events)
    assert "One." in provider.closed and "One." not in provider.completed
    assert provider.started == ["One."]  # Two and Three never started

    # The session keeps working after a cancel.
    session.feed("Four.")
    turn = session.flush()
    events = await _collect_until(session, TurnDone)
    assert events[-1] == TurnDone(turn)
    assert all(e.pcm.startswith(b"Four.") for e in events if isinstance(e, Audio))
    await session.close()


async def test_cancel_drops_audio_that_was_not_sent_yet():
    provider = FakeStreamProvider(chunks=3)
    session = _session(provider)
    session.feed("One. Two.")
    session.flush()
    await asyncio.sleep(0.1)  # everything is synthesized, nothing consumed
    await session.cancel()
    events = await _collect_until(session, Cancelled)
    assert not any(isinstance(e, Audio | SentenceStarted) for e in events)
    await session.close()


async def test_cancel_discards_buffered_text():
    provider = FakeStreamProvider()
    session = _session(provider)
    session.feed("This half sentence")
    await session.cancel()
    session.feed("Fresh start.")
    session.flush()
    await _collect_until(session, TurnDone)
    assert provider.started == ["Fresh start."]
    await session.close()


async def test_a_failing_sentence_ends_its_turn_but_not_the_session():
    provider = FakeStreamProvider(fail_on="Bad")
    session = _session(provider)
    session.feed("Good one. Bad one. Skipped one.")
    session.flush()
    events = await _collect_until(session, TurnFailed)
    failed = events[-1]
    assert failed.turn == 0 and failed.error.kind == "timeout"
    assert "Skipped one." not in provider.started

    session.feed("Next.")
    turn = session.flush()
    events = await _collect_until(session, TurnDone)
    assert events[-1] == TurnDone(turn)
    assert not any(isinstance(e, TurnDone) and e.turn == 0 for e in events)
    await session.close()


async def test_turn_length_is_limited():
    session = _session(FakeStreamProvider(), max_turn_chars=20)
    session.feed("a" * 15)
    with pytest.raises(InvalidRequest, match="longer than 20"):
        session.feed("b" * 10)
    session.flush()
    session.feed("c" * 15)  # a new turn has a fresh budget
    await session.close()


async def test_repeated_sentences_come_from_the_cache(tmp_path):
    provider = FakeStreamProvider()
    session = _session(provider, tmp_path)
    for _ in range(2):
        session.feed("Hello there.")
        session.flush()
        events = await _collect_until(session, TurnDone)
    started = [e for e in events if isinstance(e, SentenceStarted)]
    assert started[0].cached is True
    assert provider.started == ["Hello there."]  # synthesized once
    await session.close()


async def test_close_stops_speaking():
    provider = FakeStreamProvider(chunks=100, delay=0.02)
    session = _session(provider)
    session.feed("A long sentence.")
    session.flush()
    await _collect_until(session, Audio)
    await session.close()
    assert "A long sentence." in provider.closed
    assert "A long sentence." not in provider.completed


def test_session_requires_pcm():
    provider = FakeStreamProvider()
    engine = TTSEngine({"fake": provider}, default_provider="fake", default_voice="v1")
    with pytest.raises(ValueError, match="PCM"):
        SpeechSession(engine, provider, "v1", SynthOpts(format="wav"))
