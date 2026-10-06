"""Tests for SpeechSession: incremental text in, sentence-by-sentence audio out."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.cache import FileCache
from app.core.engine import InvalidRequest, TTSEngine
from app.core.providers.base import ProviderError, SynthOpts, Voice
from app.core.session import (
    FIRST_CHUNK_MAX_CHARS,
    Audio,
    Cancelled,
    LatencyModel,
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
        SentenceStarted(0, 0, len("Hello there."), False, "Hello there.", "sentence"),
        Audio(b"Hello there.#0"),
        Audio(b"Hello there.#1"),
        SentenceStarted(1, 0, len("How are you"), False, "How are you", "end"),
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


async def test_text_is_normalized_and_empty_chunks_are_skipped():
    provider = FakeStreamProvider(chunks=1)
    session = _session(provider)
    session.feed("**Hello** there, see https://example.com now.\n\U0001F600\n")
    session.flush()
    events = await _collect_until(session, TurnDone)
    assert provider.started == ["Hello there, see link now."]  # the emoji-only line is gone
    assert [e for e in events if isinstance(e, SentenceStarted)][0].chars == len(
        "Hello there, see link now."
    )
    await session.close()


async def test_normalization_can_be_turned_off():
    provider = FakeStreamProvider(chunks=1)
    session = _session(provider, normalize=False)
    session.feed("**Hello** there.")
    session.flush()
    await _collect_until(session, TurnDone)
    assert provider.started == ["**Hello** there."]
    await session.close()


async def test_decimals_and_abbreviations_are_not_split():
    provider = FakeStreamProvider(chunks=1)
    session = _session(provider, chunking="sentence")
    for token in "It takes 3.5 seconds, e.g. on Dr. Smith's laptop. Done.".split(" "):
        session.feed(token + " ")
    session.flush()
    await _collect_until(session, TurnDone)
    assert provider.started == ["It takes 3.5 seconds, e.g. on Dr. Smith's laptop.", "Done."]
    await session.close()


async def test_shaped_session_trims_padding_and_adds_boundary_pauses():
    from app.core import audio
    from tests.test_audio import PaddedPcmProvider

    provider = PaddedPcmProvider()
    engine = TTSEngine({"padded": provider}, default_provider="padded", default_voice="v1",
                       shape_audio=True)
    session = SpeechSession(engine, provider, "v1", PCM, chunking="sentence")
    assert session._shape_audio is True  # follows the engine by default
    session.start()
    session.feed("One. Two.")
    session.flush()
    events = await _collect_until(session, TurnDone)
    pcm = b"".join(e.pcm for e in events if isinstance(e, Audio))
    # "One." ends a sentence (300 ms pause); "Two." ends the turn (no pause).
    expected = 2 * 400 + audio.PAUSE_AFTER_MS["sentence"]
    assert expected <= audio.duration_ms(pcm) <= expected + 4 * audio.EDGE_KEEP_MS + 5
    boundaries = [e.boundary for e in events if isinstance(e, SentenceStarted)]
    assert boundaries == ["sentence", "end"]
    await session.close()


# --- adaptive chunking -------------------------------------------------------------


class TimedProvider:
    """Takes ``a + b * chars`` seconds per chunk; its audio lasts ``ms_per_char`` each.

    The defaults make short chunks slower to produce than to play (like Kokoro's fixed
    cost per chunk) and long ones faster, which is where chunking decides stalls.
    """

    name = "timed"

    def __init__(self, a: float = 0.15, b: float = 0.004, ms_per_char: float = 10.0):
        self.a, self.b, self.ms_per_char = a, b, ms_per_char
        self.latency_prior = (a, b)
        self.chunks: list[str] = []

    async def synthesize(self, text, voice, opts):
        return b"".join([c async for c in self.synthesize_stream(text, voice, opts)])

    async def synthesize_stream(self, text, voice, opts):
        self.chunks.append(text)
        await asyncio.sleep(self.a + self.b * len(text))
        frames = int(24_000 * len(text) * self.ms_per_char / 1000)
        yield b"\x10\x27" * frames  # constant, clearly audible samples

    async def list_voices(self):
        return [Voice(id="v1", name="v1", provider=self.name)]


PARAGRAPH = (
    "Hi! I can definitely help you plan a relaxing trip to the coast this weekend, "
    "with a scenic drive along the water and a few great places to stop for lunch. "
    "Let's start with the route, then pick a hotel near the beach for both nights."
)


async def _play(chunking: str) -> tuple[float, list[str], float]:
    """Run a turn and play its audio at 1x as it arrives; return (stall s, chunks, first)."""
    provider = TimedProvider()
    engine = TTSEngine({"timed": provider}, default_provider="timed", default_voice="v1")
    session = SpeechSession(engine, provider, "v1", PCM, chunking=chunking, normalize=False)
    session.start()
    started = time.monotonic()
    session.feed(PARAGRAPH)
    session.flush()
    play_start = first_audio = None
    played = stall = 0.0
    async for event in session.events():
        now = time.monotonic()
        if isinstance(event, Audio):
            if play_start is None:
                play_start, first_audio = now, now - started
            elif now > play_start + played + stall:
                stall += now - (play_start + played + stall)
            played += len(event.pcm) / 48_000
        elif isinstance(event, TurnDone):
            break
    await session.close()
    return stall, provider.chunks, first_audio


async def test_adaptive_chunking_starts_fast_and_never_stalls():
    stall, chunks, first_audio = await _play("adaptive")
    assert stall < 0.1, f"stalled {stall:.2f}s with chunks {chunks}"
    assert len(chunks[0]) <= FIRST_CHUNK_MAX_CHARS  # a short first chunk ...
    assert first_audio < 0.5  # ... so audio starts quickly
    assert "Hi!" not in chunks  # the tiny first sentence was merged, not spoken alone
    assert max(len(c) for c in chunks) > len(chunks[0])  # chunks grew


async def test_sentence_chunking_stalls_on_the_same_text():
    stall, chunks, _ = await _play("sentence")
    assert chunks[0] == "Hi!"
    assert stall > 0.3  # the long second sentence takes longer than "Hi!" plays


def test_latency_model_learns_the_real_cost():
    model = LatencyModel(prior=(1.0, 0.05))
    assert model.estimate() == pytest.approx((1.0, 0.05))
    for chars in [20, 60, 100, 40, 80] * 4:
        model.update(chars, 0.3 + 0.01 * chars)
    a, b = model.estimate()
    assert a == pytest.approx(0.3, abs=0.05) and b == pytest.approx(0.01, abs=0.002)


async def test_adaptive_session_numbers_turns_and_finishes_short_ones():
    session = _session(FakeStreamProvider(), chunking="adaptive")
    for expected in range(3):
        session.feed("Hi.")  # shorter than the minimum: spoken when the turn ends
        assert session.flush() == expected
        events = await _collect_until(session, TurnDone)
        assert events[-1] == TurnDone(expected)
        assert [e.boundary for e in events if isinstance(e, SentenceStarted)] == ["end"]
    await session.close()


async def test_adaptive_session_speaks_before_the_turn_ends():
    provider = FakeStreamProvider()
    session = _session(provider, chunking="adaptive")
    first = "This first sentence is long enough to speak now."
    session.feed(first + " And then")
    await _collect_until(session, Audio)
    # Nothing is queued for playback yet, so the first chunk is short.
    assert first.startswith(provider.started[0])
    assert len(provider.started[0]) <= FIRST_CHUNK_MAX_CHARS
    session.feed(" a second one.")
    session.flush()
    await _collect_until(session, TurnDone)
    spoken = " ".join(provider.started)
    assert spoken.split() == (first + " And then a second one.").split()
    await session.close()


async def test_adaptive_cancel_stops_speech_and_drops_the_turn():
    provider = FakeStreamProvider(chunks=50, delay=0.02)
    session = _session(provider, chunking="adaptive")
    session.feed("The first sentence is being spoken right now. A second one is waiting.")
    session.flush()
    await _collect_until(session, Audio)
    await session.cancel()
    events = await _collect_until(session, Cancelled)
    assert not any(isinstance(e, TurnDone) for e in events)
    assert len(provider.started) == 1 and provider.started[0] not in provider.completed

    provider.chunks, provider.delay = 1, 0.0
    session.feed("A fresh turn after the cancel.")
    turn = session.flush()
    assert (await _collect_until(session, TurnDone))[-1] == TurnDone(turn)
    await session.close()


async def test_adaptive_failure_ends_only_its_turn():
    provider = FakeStreamProvider(fail_on="Bad")
    session = _session(provider, chunking="adaptive")
    session.feed("This Bad sentence fails to synthesize. The rest is skipped.")
    session.flush()
    failed = (await _collect_until(session, TurnFailed))[-1]
    assert failed.turn == 0
    session.feed("The next turn works fine again.")
    turn = session.flush()
    assert (await _collect_until(session, TurnDone))[-1] == TurnDone(turn)
    assert not any("rest is skipped" in s for s in provider.started)
    await session.close()


async def test_adaptive_chunks_stay_above_break_even_on_a_slow_machine():
    # Each chunk costs 0.2 s + 4 ms/char but speaks only 6 ms/char: anything under
    # 100 chars takes longer to make than to play. Small chunks would stall forever.
    provider = TimedProvider(a=0.2, b=0.004, ms_per_char=6)
    engine = TTSEngine({"timed": provider}, default_provider="timed", default_voice="v1")
    session = SpeechSession(engine, provider, "v1", PCM, chunking="adaptive",
                            normalize=False)
    session._timing.audio_per_char = 0.006  # this voice speaks faster than the prior
    session.start()
    session.feed(PARAGRAPH)
    session.flush()
    await _collect_until(session, TurnDone, timeout=20)
    first, *rest = provider.chunks
    assert len(first) <= FIRST_CHUNK_MAX_CHARS  # the listener is waiting: latency first
    # After that, chunks stay above break-even so the buffer can catch up.
    assert all(len(chunk) >= 100 for chunk in rest[:-1]), provider.chunks
    await session.close()


async def test_sessions_share_what_they_learn_about_a_provider():
    provider = FakeStreamProvider()
    engine = TTSEngine({"fake": provider}, default_provider="fake", default_voice="v1")
    first = SpeechSession(engine, provider, "v1", PCM, chunking="adaptive")
    first._timing.latency.update(50, 9.0)  # pretend it learned the CPU is very slow
    second = SpeechSession(engine, provider, "v1", PCM, chunking="adaptive")
    assert second._timing is first._timing
    faster = SpeechSession(engine, provider, "v1", SynthOpts(format="pcm", speed=1.5),
                           chunking="adaptive")
    assert faster._timing is not first._timing  # another speed speaks at another rate
