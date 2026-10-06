"""Incremental speech session: text arrives in pieces, audio goes out sentence by sentence.

This is the engine side of a real-time chatbot voice. The client (a chatbot backend)
forwards LLM tokens as they are generated, and speech starts as soon as the first
sentence is complete, while the LLM is still writing the rest. It is transport-neutral:
the WebSocket route (``app/api/routes/realtime.py``) only translates messages.

Model:

- ``feed(text)`` adds text to the current *turn*; complete sentences are queued.
- ``flush()`` ends the turn: leftover text is queued, and a ``TurnDone`` follows the
  turn's last audio.
- ``cancel()`` is barge-in: queued sentences, buffered text and audio not yet sent
  are dropped, the sentence being synthesized is stopped (its provider stream is
  closed, which stops Kokoro's worker), and a ``Cancelled`` event is emitted. It ends
  every pending turn; the next ``feed`` starts a new one.

A single speaker task synthesizes queued sentences in order, starting the next one as
soon as the previous has been synthesized. It doesn't wait for playback, so the
client's buffer grows while audio is produced faster than real time. A sentence that
fails (e.g. provider timeout) emits ``TurnFailed`` and skips the rest of that turn, but
the session keeps working for later turns.

Every turn ends with exactly one of ``TurnDone``, ``TurnFailed`` or ``Cancelled``.

Chunking modes:

- ``sentence`` / ``fast_start`` ("push"): text is cut into chunks as soon as it arrives
  (``fast_start`` lets the first chunk end at a comma), and chunks queue up for the
  speaker.
- ``adaptive`` ("pull"): text just accumulates per turn, and the speaker cuts the next
  chunk only when it is ready to synthesize it. The chunk length is chosen so that it
  is synthesized before the audio already queued for playback runs out. It comes from
  a latency model ``a + b * chars`` (seeded by the provider's ``latency_prior``,
  refined after every synthesized chunk) and a playback clock. So the first chunk is
  short (fast first audio), later chunks grow, and once enough audio is queued whole
  sentences are used again (best prosody, cache friendly). A chunk is never shorter
  than ``MIN_CHUNK_CHARS``, nor than the break-even length ``a / (d - b)`` (``d`` =
  seconds of speech per char, also learned): each chunk costs a fixed ~0.6 s with
  Kokoro, so on a slow or busy CPU short chunks take longer to make than to play, and
  only longer chunks can rebuild the buffer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal

from .chunker import SentenceChunker
from .engine import InvalidRequest, TTSEngine, shape_chunk
from .providers.base import SynthOpts, TTSProvider
from .textnorm import normalize_for_speech
from .timing import LatencyModel, SpeechTiming  # noqa: F401 - LatencyModel re-exported

log = logging.getLogger("app.session")

Chunking = Literal["sentence", "fast_start", "adaptive"]

BYTES_PER_SECOND = 24_000 * 2  # canonical PCM: 24 kHz, 16-bit, mono
MIN_CHUNK_CHARS = 20  # below this, a chunk takes longer to make than to play
# Aim: first audio this long after the first chunk is ready. The first chunk is usually
# ready before the first sentence is complete, so the wait a listener notices is shorter.
# A first chunk that is too short backfires: its audio barely covers the next chunk's
# synthesis, so chunks stay small (and cut mid-phrase) for a long time.
FIRST_CHUNK_TARGET_S = 1.5
FIRST_CHUNK_MAX_CHARS = 50
SAFETY = 0.85  # use only this share of the queued audio as synthesis time budget
SUSTAIN_MARGIN = 1.25  # chunks at least this much above the break-even length
# When synthesis can't keep up at any size (a throttled CPU), bigger chunks no longer
# help, they only make each pause longer: cap the break-even floor.
MAX_FLOOR_CHARS = 120


# --- events (session -> client) ---------------------------------------------------


@dataclass(slots=True)
class SentenceStarted:
    index: int  # running count across the session
    turn: int
    chars: int
    cached: bool
    text: str = ""  # what is being spoken (after normalization)
    boundary: str = "sentence"  # how the chunk ended: sentence|clause|line|word|end


@dataclass(slots=True)
class Audio:
    pcm: bytes


@dataclass(slots=True)
class TurnDone:
    turn: int


@dataclass(slots=True)
class TurnFailed:
    turn: int
    error: Exception


@dataclass(slots=True)
class Cancelled:
    pass


Event = SentenceStarted | Audio | TurnDone | TurnFailed | Cancelled


# --- work items (feed/flush -> speaker) -------------------------------------------


@dataclass(slots=True)
class _Sentence:
    generation: int
    turn: int
    text: str
    boundary: str = "sentence"


@dataclass(slots=True)
class _TurnEnd:
    generation: int
    turn: int


@dataclass(slots=True)
class _InputTurn:
    """Adaptive mode: one turn's text, cut into chunks on demand."""

    generation: int
    turn: int
    chunker: SentenceChunker
    final: bool = False
    failed: bool = False
    spoken: bool = False  # has a chunk of this turn been taken yet?


class SpeechSession:
    def __init__(
        self,
        engine: TTSEngine,
        provider: TTSProvider,
        voice: str,
        opts: SynthOpts,
        *,
        chunking: Chunking = "fast_start",
        use_cache: bool = True,
        normalize: bool = True,
        shape_audio: bool | None = None,
        max_turn_chars: int = 10_000,
        max_sentence_length: int = 200,
    ) -> None:
        if opts.format != "pcm":
            raise ValueError("SpeechSession streams raw PCM; pass SynthOpts(format='pcm').")
        self._engine = engine
        self._provider = provider
        self._voice = voice
        self._opts = opts
        self._use_cache = use_cache
        self._normalize = normalize
        self._shape_audio = engine.shape_audio if shape_audio is None else shape_audio
        self._max_turn_chars = max_turn_chars
        self._max_sentence_length = max_sentence_length
        self._chunking = chunking
        self._chunker = SentenceChunker(
            max_length=max_sentence_length, fast_start=chunking == "fast_start"
        )
        # Adaptive mode: turns waiting to be spoken, and what the speaker plans with.
        self._turns: deque[_InputTurn] = deque()
        self._wakeup = asyncio.Event()
        # Shared with other sessions on this provider/speed: learning carries over.
        self._timing: SpeechTiming = engine.timing(provider, opts.speed)
        self._play_start: float | None = None  # when the current run of audio started
        self._played_s = 0.0  # audio emitted since then, in seconds

        self._work: asyncio.Queue[_Sentence | _TurnEnd] = asyncio.Queue()
        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._generation = 0  # bumped by cancel(); older work is dropped
        self._turn = 0
        self._turn_chars = 0
        self._sentence_index = 0
        self._failed_turns: set[int] = set()
        self._speaker: asyncio.Task | None = None
        self._current: asyncio.Task | None = None  # the sentence being synthesized

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def chunking(self) -> Chunking:
        return self._chunking

    def start(self) -> None:
        if self._speaker is None:
            loop = self._adaptive_loop if self._chunking == "adaptive" else self._speak_loop
            self._speaker = asyncio.create_task(loop())

    async def close(self) -> None:
        """Stop everything (the client left). Safe to call more than once."""
        for task in (self._current, self._speaker):
            if task is not None and not task.done():
                task.cancel()
        for task in (self._current, self._speaker):
            if task is not None:
                await asyncio.wait({task})

    # --- commands ----------------------------------------------------------------

    def feed(self, text: str) -> None:
        """Add text to the current turn; complete sentences start synthesizing."""
        if self._turn_chars + len(text) > self._max_turn_chars:
            raise InvalidRequest(
                f"Turn is longer than {self._max_turn_chars} characters; send 'flush' "
                f"to end it and start a new one."
            )
        self._turn_chars += len(text)
        if self._chunking == "adaptive":
            self._input_turn().chunker.append(text)
            self._wakeup.set()
            return
        for chunk in self._chunker.feed_chunks(text):
            self._queue_sentence(chunk.text, chunk.boundary)

    def flush(self) -> int:
        """End the current turn; returns its number."""
        if self._chunking == "adaptive":
            turn = self._input_turn()
            turn.final = True
            self._next_turn()
            self._wakeup.set()
            return turn.turn
        tail = self._chunker.flush_chunk()
        if tail:
            self._queue_sentence(tail.text, tail.boundary)
        turn = self._turn
        self._work.put_nowait(_TurnEnd(self._generation, turn))
        self._next_turn()
        return turn

    async def cancel(self) -> None:
        """Barge-in: stop speaking now and drop everything pending."""
        self._generation += 1
        self._chunker.reset()
        self._turns.clear()
        self._wakeup.set()
        _drain(self._work)
        # Audio already produced but not yet sent must not play after the cancel.
        kept = [e for e in _drain(self._events) if not isinstance(e, SentenceStarted | Audio)]
        for event in kept:
            self._events.put_nowait(event)
        current = self._current
        if current is not None and not current.done():
            current.cancel()  # closes the provider stream via aclosing()
            await asyncio.wait({current})
        self._next_turn()
        self._events.put_nowait(Cancelled())
        log.info("speech cancelled")

    async def events(self) -> AsyncIterator[Event]:
        while True:
            yield await self._events.get()

    # --- internals ---------------------------------------------------------------

    def _queue_sentence(self, text: str, boundary: str = "sentence") -> None:
        if self._normalize:
            text = normalize_for_speech(text)
            if not text:
                return  # e.g. a chunk that was only markdown or emoji
        self._work.put_nowait(_Sentence(self._generation, self._turn, text, boundary))

    def _next_turn(self) -> None:
        self._turn += 1
        self._turn_chars = 0

    def _input_turn(self) -> _InputTurn:
        """Adaptive mode: the turn that new text goes into (opened on first use)."""
        if self._turns and not self._turns[-1].final:
            return self._turns[-1]
        turn = _InputTurn(
            self._generation, self._turn, SentenceChunker(max_length=self._max_sentence_length)
        )
        self._turns.append(turn)
        return turn

    def _budget(self, first_of_turn: bool) -> tuple[int | None, int]:
        """Adaptive mode: ``(max_chars, min_chars)`` for the next chunk.

        A turn's first chunk, with nothing playing, is sized for latency only: the
        listener is waiting for it. After that, ``max_chars`` is the longest chunk that
        is ready before playback runs dry (``None``: there's audio enough to take whole
        sentences), and ``min_chars`` is normally ``MIN_CHUNK_CHARS``. When even that
        would fall behind (a slow or busy CPU), the chunk is sized for efficiency
        instead: between the break-even length and 1.5x of it. Then small chunks would
        stall forever, while bigger ones stall once and catch up.
        """
        a, b = self._timing.latency.estimate()
        # A chunk shorter than `sustain` takes longer to make than to play, so a run of
        # them can never build up a buffer (on a busy CPU that can be ~70 chars). Never
        # go below it, even if that means one stall now: small chunks would stall forever.
        spare = self._timing.audio_per_char - b
        sustain = a / spare if spare > 1e-4 else float(self._max_sentence_length)
        floor = min(max(MIN_CHUNK_CHARS, SUSTAIN_MARGIN * sustain), MAX_FLOOR_CHARS)
        now = time.monotonic()
        ahead = 0.0
        if self._play_start is not None:
            ahead = self._play_start + self._played_s - now
        if ahead <= 0 and first_of_turn:  # the listener is waiting: latency first
            chars = (FIRST_CHUNK_TARGET_S - a) / b
            return int(min(max(chars, MIN_CHUNK_CHARS), FIRST_CHUNK_MAX_CHARS)), MIN_CHUNK_CHARS
        chars = (SAFETY * ahead - a) / b
        lowest = MIN_CHUNK_CHARS
        if chars < floor:  # a chunk that fits the buffer would lose ground anyway
            lowest, chars = int(floor), 1.5 * floor
        if chars >= self._max_sentence_length:
            return None, min(lowest, self._max_sentence_length)
        return int(chars), lowest

    async def _adaptive_loop(self) -> None:
        while True:
            self._wakeup.clear()
            if not self._turns:
                await self._wakeup.wait()
                continue
            turn = self._turns[0]
            if turn.generation != self._generation:
                self._turns.popleft()
                continue
            if turn.failed:  # TurnFailed was sent; drop the rest of this turn
                if turn.final:
                    self._turns.popleft()
                else:
                    turn.chunker.reset()
                    await self._wakeup.wait()
                continue
            budget, lowest = self._budget(first_of_turn=not turn.spoken)
            chunk = turn.chunker.take(budget, final=turn.final, min_chars=lowest)
            if chunk is None:
                if turn.final:  # all of it has been spoken
                    self._turns.popleft()
                    self._events.put_nowait(TurnDone(turn.turn))
                else:
                    await self._wakeup.wait()  # more text is coming
                continue
            text = normalize_for_speech(chunk.text) if self._normalize else chunk.text
            if not text:
                continue
            log.debug("chunk: budget=%s..%s chars=%d boundary=%s", lowest, budget,
                      len(text), chunk.boundary)
            turn.spoken = True
            item = _Sentence(turn.generation, turn.turn, text, chunk.boundary)
            if not await self._run(item):
                turn.failed = True

    async def _speak_loop(self) -> None:
        while True:
            item = await self._work.get()
            if item.generation != self._generation:
                continue  # cancelled before it started
            if isinstance(item, _TurnEnd):
                if item.turn in self._failed_turns:
                    self._failed_turns.discard(item.turn)  # TurnFailed already sent
                else:
                    self._events.put_nowait(TurnDone(item.turn))
                continue
            if item.turn in self._failed_turns:
                continue  # an earlier sentence of this turn failed
            await self._run(item)

    async def _run(self, item: _Sentence) -> bool:
        """Speak one chunk; False if it failed (``TurnFailed`` has been sent)."""
        self._current = asyncio.create_task(self._speak(item))
        await asyncio.wait({self._current})  # doesn't raise if the sentence was cancelled
        task, self._current = self._current, None
        if task.cancelled():
            return True  # cancel() already reported it
        error = task.exception()
        if error is not None and item.generation == self._generation:
            log.warning("turn %d failed: %s", item.turn, error)
            self._failed_turns.add(item.turn)
            self._events.put_nowait(TurnFailed(item.turn, error))
            return False
        return True

    async def _speak(self, item: _Sentence) -> None:
        started = time.monotonic()
        audio = await self._engine.open_sentence(
            self._provider, self._voice, self._opts, item.text, self._use_cache
        )
        self._events.put_nowait(
            SentenceStarted(
                self._sentence_index,
                item.turn,
                len(item.text),
                audio.cached,
                item.text,
                item.boundary,
            )
        )
        self._sentence_index += 1
        source = audio.chunks
        if self._shape_audio:
            source = shape_chunk(source, item.boundary)  # trims edges, adds the pause
        first = True
        chunk_audio_s = 0.0
        async with aclosing(source) as chunks:
            async for pcm in chunks:
                now = time.monotonic()
                if first and not audio.cached:
                    self._timing.latency.update(len(item.text), now - started)
                first = False
                # Playback clock: a client plays audio as it arrives, at 1x. If it ran
                # dry before this chunk, a new run of audio starts now.
                if self._play_start is None or now >= self._play_start + self._played_s:
                    self._play_start, self._played_s = now, 0.0
                self._played_s += len(pcm) / BYTES_PER_SECOND
                chunk_audio_s += len(pcm) / BYTES_PER_SECOND
                self._events.put_nowait(Audio(pcm))
        if len(item.text) >= MIN_CHUNK_CHARS // 2:  # how long this voice speaks per char
            self._timing.record_audio(len(item.text), chunk_audio_s)


def _drain(queue: asyncio.Queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items
