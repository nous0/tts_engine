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
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass

from .chunker import SentenceChunker
from .engine import InvalidRequest, TTSEngine
from .providers.base import SynthOpts, TTSProvider

log = logging.getLogger("app.session")


# --- events (session -> client) ---------------------------------------------------


@dataclass(slots=True)
class SentenceStarted:
    index: int  # running count across the session
    turn: int
    chars: int
    cached: bool


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


@dataclass(slots=True)
class _TurnEnd:
    generation: int
    turn: int


class SpeechSession:
    def __init__(
        self,
        engine: TTSEngine,
        provider: TTSProvider,
        voice: str,
        opts: SynthOpts,
        *,
        fast_start: bool = True,
        use_cache: bool = True,
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
        self._max_turn_chars = max_turn_chars
        self._chunker = SentenceChunker(max_length=max_sentence_length, fast_start=fast_start)

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

    def start(self) -> None:
        if self._speaker is None:
            self._speaker = asyncio.create_task(self._speak_loop())

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
        for sentence in self._chunker.feed(text):
            self._queue_sentence(sentence)

    def flush(self) -> int:
        """End the current turn; returns its number."""
        tail = self._chunker.flush()
        if tail:
            self._queue_sentence(tail)
        turn = self._turn
        self._work.put_nowait(_TurnEnd(self._generation, turn))
        self._next_turn()
        return turn

    async def cancel(self) -> None:
        """Barge-in: stop speaking now and drop everything pending."""
        self._generation += 1
        self._chunker.reset()
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

    def _queue_sentence(self, text: str) -> None:
        self._work.put_nowait(_Sentence(self._generation, self._turn, text))

    def _next_turn(self) -> None:
        self._turn += 1
        self._turn_chars = 0

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

            self._current = asyncio.create_task(self._speak(item))
            await asyncio.wait({self._current})  # doesn't raise if the sentence was cancelled
            task, self._current = self._current, None
            if task.cancelled():
                continue  # cancel() already reported it
            error = task.exception()
            if error is not None and item.generation == self._generation:
                log.warning("turn %d failed: %s", item.turn, error)
                self._failed_turns.add(item.turn)
                self._events.put_nowait(TurnFailed(item.turn, error))

    async def _speak(self, item: _Sentence) -> None:
        audio = await self._engine.open_sentence(
            self._provider, self._voice, self._opts, item.text, self._use_cache
        )
        self._events.put_nowait(
            SentenceStarted(self._sentence_index, item.turn, len(item.text), audio.cached)
        )
        self._sentence_index += 1
        async with aclosing(audio.chunks) as chunks:
            async for pcm in chunks:
                self._events.put_nowait(Audio(pcm))


def _drain(queue: asyncio.Queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items
