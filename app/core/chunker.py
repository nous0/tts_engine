"""Sentence chunker for incremental chatbot streaming.

Text from an LLM arrives token-by-token, but TTS providers synthesize best on
complete utterances. ``SentenceChunker`` buffers incoming text and flushes a
chunk whenever it hits a sentence boundary (``.`` ``?`` ``!`` or a newline) or
a configurable maximum length, so each flushed piece can be synthesized and
streamed while later text is still arriving.

A terminator only ends a sentence when whitespace (or the end of the turn) follows it,
so ``3.5``, ``example.com`` and ``$4.99`` stay whole. It doesn't end one after common
abbreviations (``Dr.``, ``e.g.``), initials (``U.S.``, ``J. K.``) or a list number at
the start of a chunk (``1.``). ``...`` and ``?!`` count as one terminator, and closing
quotes, brackets and markdown emphasis stay on the chunk. When a terminator is the last
character received so far, the chunker waits for the next one, so the result doesn't
depend on how the text was split into tokens: streaming token by token gives the same
chunks as splitting the whole text at once.

With ``fast_start``, the first chunk of a turn may also end at a clause break
(``,`` ``;`` ``:``) once it is long enough, so the first audio comes sooner. Only the
first chunk does this: later chunks keep whole sentences, which sound better.

``feed``/``flush`` cut chunks as soon as the text allows ("push"). ``append``/``take``
instead let the caller decide *when* to cut and how long a chunk may be ("pull"): the
real-time scheduler (``app/core/session.py``) takes the next chunk only when it is
ready to synthesize it, with a length that fits the audio still queued for playback.
``take`` prefers, in order: whole sentences (several, if they fit), a clause break,
a word boundary before a conjunction, any word boundary.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable
from typing import Literal, NamedTuple

# Characters that end a sentence. The terminator is kept on the flushed chunk.
_TERMINATORS = ".!?\u2026"
# Whitespace that should force a flush without being part of the chunk.
_FLUSH_WHITESPACE = "\n\r"
# Clause breaks a fast-start first chunk may end on.
_CLAUSE_BREAKS = ",;:"
# Kept on a chunk right after its terminator: quotes, brackets, markdown emphasis.
_CLOSERS = "\"'\u201d\u2019)]}*_`"
# A "." after these words doesn't end a sentence (matched lowercase, inner dots kept).
_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr st vs etc e.g i.e approx inc ltd dept fig a.m p.m".split()
)
# A cut just before one of these words keeps the phrasing natural ("..., | and then").
_CONJUNCTIONS = frozenset(
    "and but or so because which that who whom whose when while where why how what if "
    "as like after before since until unless although though whether with for to of in "
    "on at from about into than then over under through across around during without "
    "within between".split()
)
# "U.S", "J", "e.g"-style initials: single letters joined by dots.
_INITIALS = re.compile(r"^(?:[A-Za-z]\.)*[A-Za-z]$")
# Returned by _sentence_end when the next character decides.
_WAIT = "wait"

# How a chunk ended; decides the pause after it when chunks are joined.
Boundary = Literal["sentence", "clause", "line", "word", "end"]


class Chunk(NamedTuple):
    text: str
    boundary: Boundary  # sentence end, clause break, newline, length cut, or end of turn


class SentenceChunker:
    """Buffer text and yield sentence-ish chunks.

    Usage:

        chunker = SentenceChunker()
        for token in llm_tokens:
            for sentence in chunker.feed(token):
                await synthesize(sentence)
        tail = chunker.flush()
        if tail:
            await synthesize(tail)
    """

    def __init__(
        self,
        max_length: int = 200,
        terminators: str = _TERMINATORS,
        fast_start: bool = False,
        clause_min_chars: int = 24,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self._max_length = max_length
        self._terminators = set(terminators)
        self._fast_start = fast_start
        self._clause_min_chars = clause_min_chars
        self._emitted = False  # has this turn produced a chunk yet?
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Append ``text`` to the buffer and return any complete chunks."""
        return [chunk.text for chunk in self.feed_chunks(text)]

    def feed_chunks(self, text: str) -> list[Chunk]:
        """Like ``feed``, also saying how each chunk ended."""
        out = self._feed(text)
        if out:
            self._emitted = True
        return out

    def _feed(self, text: str) -> list[Chunk]:
        out: list[Chunk] = []
        self._buffer += text

        while True:
            # Drop leading spaces/tabs so a chunk never starts with whitespace.
            self._buffer = self._buffer.lstrip(" \t")
            buf = self._buffer
            if not buf:
                return out

            newline = _first_index(buf, _FLUSH_WHITESPACE)
            limit = newline if newline != -1 else len(buf)
            end = _sentence_end(buf, self._terminators, limit)
            # The earliest candidate cut, whatever its kind.
            if isinstance(end, tuple):
                first_cut = end[0]
            elif newline != -1:
                first_cut = newline
            else:
                first_cut = None

            # 0) Never emit a chunk longer than max_length, however it would end:
            #    the same text must give the same chunks streamed or all at once.
            if len(buf) > self._max_length and (
                first_cut is None or first_cut > self._max_length
            ):
                cut = self._max_length
                space_idx = buf.rfind(" ", 0, cut)
                if space_idx > 0:
                    cut = space_idx
                out.append(Chunk(buf[:cut].rstrip(), "word"))
                self._buffer = buf[cut:].lstrip()
                continue

            # 1) Fast start: the turn's first chunk may end at a clause break that
            #    comes before the first sentence end or newline.
            if self._fast_start and not self._emitted and not out:
                clause_idx = self._first_clause_break()
                if clause_idx != -1 and (first_cut is None or clause_idx < first_cut):
                    out.append(Chunk(buf[: clause_idx + 1], "clause"))
                    self._buffer = buf[clause_idx + 1 :]
                    continue

            # 2) A sentence end before any newline.
            if isinstance(end, tuple):
                cut, resume = end
                out.append(Chunk(buf[:cut], "sentence"))
                self._buffer = buf[resume:]
                continue

            # 3) A newline ends the chunk; it isn't part of it.
            if newline != -1:
                chunk = buf[:newline].rstrip()
                rest = newline + 1
                while rest < len(buf) and buf[rest] in _FLUSH_WHITESPACE:
                    rest += 1
                self._buffer = buf[rest:]
                if chunk:
                    out.append(Chunk(chunk, "line"))
                continue

            # A terminator is the last thing received (end is _WAIT), or there is no
            # boundary yet: either way, wait for more text.
            return out

    def append(self, text: str) -> None:
        """Add text without cutting anything; chunks come out of ``take``."""
        self._buffer += text

    def take(self, max_chars: int | None, final: bool, min_chars: int = 20) -> Chunk | None:
        """Cut the next chunk from the buffered text, or ``None`` to wait for more.

        ``max_chars`` caps the chunk (``None`` means only ``max_length``); ``final``
        says no more text will come in this turn. Whole sentences are preferred, and
        several are merged when they fit. A sentence shorter than ``min_chars`` waits
        to be merged with what follows, since a tiny chunk costs nearly as much to
        synthesize as a long one. A sentence is only cut inside when it doesn't fit,
        at the best boundary: a clause break, then a word before a conjunction, then
        any word.
        """
        buf = self._buffer.lstrip(" \t\r\n")
        self._buffer = buf
        if not buf:
            return None
        limit = self._max_length if max_chars is None else min(max_chars, self._max_length)
        ends = _ends(buf, self._terminators)
        decided = ends[-1][1] if ends else 0
        if final and buf[decided:].strip():
            # End of the turn: whatever is left (e.g. "Done." waiting on what follows)
            # is a complete piece too.
            ends.append((len(buf.rstrip()), len(buf), "end"))
        fitting = [end for end in ends if end[0] <= limit]

        if fitting and fitting[-1][0] >= min_chars:
            return self._cut(*fitting[-1])
        if len(buf) <= limit:
            if final:
                return self._cut(len(buf.rstrip()), len(buf), "end")
            return None  # the sentence isn't finished (or is too short): wait
        # Too long to wait any more. Whole sentences that aren't tiny still beat a cut
        # mid-sentence; otherwise cut inside, after any tiny leading sentences ("Hi!").
        if fitting and fitting[-1][0] >= min_chars // 2:
            return self._cut(*fitting[-1])
        # Search from half the budget (not from min_chars): with a tight budget, the
        # window [min_chars, limit] can be too narrow to hold a single space.
        tiny_end = fitting[-1][0] if fitting else 0
        inner = _inner_cut(buf, max(min(min_chars, limit // 2), tiny_end + 1), limit)
        if inner is not None:
            return self._cut(inner[0], inner[0], inner[1])
        # No usable boundary within the budget. Never split a word (a URL, a long
        # number): go past the budget to the next space, within max_length. And never
        # speak a tiny leading sentence ("Hi!") alone if a word boundary follows it.
        space = buf.rfind(" ", 0, limit + 1)
        if fitting and space <= tiny_end:
            return self._cut(*fitting[-1])
        if space <= 0:
            space = buf.find(" ", limit)
            if not 0 < space <= self._max_length:
                if not final and len(buf) <= self._max_length:
                    return None  # the word is still arriving
                space = min(len(buf), self._max_length)
        return self._cut(space, space, "word")

    def _cut(self, cut: int, resume: int, boundary: Boundary) -> Chunk:
        text = self._buffer[:cut].strip()
        self._buffer = self._buffer[resume:].lstrip(" \t")
        return Chunk(text, boundary)

    def flush(self) -> str | None:
        """End the turn: return any remaining buffered text, or ``None`` if empty.

        The next text starts a new turn (which fast-starts again).
        """
        chunk = self.flush_chunk()
        return chunk.text if chunk else None

    def flush_chunk(self) -> Chunk | None:
        """Like ``flush``; the tail's boundary is ``"end"``."""
        tail = self._buffer.strip()
        self.reset()
        return Chunk(tail, "end") if tail else None

    def reset(self) -> None:
        """Drop buffered text and start a new turn (e.g. when speech is cancelled)."""
        self._buffer = ""
        self._emitted = False

    def _first_clause_break(self) -> int:
        """Index of the first clause break that can end a fast-start chunk, or -1.

        The chunk must be at least ``clause_min_chars`` long, and like a sentence end
        the break must be followed by whitespace, so ``1,000``, ``10:30`` and
        ``https://`` aren't split. When the break is the last character received so
        far, it waits to see what follows.
        """
        buf = self._buffer
        for i, ch in enumerate(buf):
            if ch not in _CLAUSE_BREAKS or i + 1 < self._clause_min_chars:
                continue
            if i + 1 >= len(buf):
                return -1  # can't tell yet what follows
            if not buf[i + 1].isspace():
                continue  # "1,000", "10:30", "https://"
            return i
        return -1

    @property
    def pending(self) -> str:
        """Text currently buffered but not yet flushed."""
        return self._buffer


def split_sentences(text: str, max_length: int = 200, fast_start: bool = False) -> list[str]:
    """Split a complete ``text`` into sentence-ish chunks (stateless helper)."""
    return [chunk.text for chunk in split_chunks(text, max_length, fast_start)]


def split_chunks(text: str, max_length: int = 200, fast_start: bool = False) -> list[Chunk]:
    """Like ``split_sentences``, also saying how each chunk ended."""
    chunker = SentenceChunker(max_length=max_length, fast_start=fast_start)
    chunks = chunker.feed_chunks(text)
    tail = chunker.flush_chunk()
    if tail:
        chunks.append(tail)
    return chunks


async def iter_sentences(chunks: Iterable[str] | AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield complete sentences from a stream of text chunks.

    Accepts either a plain iterable (e.g. a list of LLM tokens) or an async
    iterator, so the engine can drive synthesis from either source.
    """
    chunker = SentenceChunker()
    if hasattr(chunks, "__aiter__"):
        async for piece in chunks:  # type: ignore[union-attr]
            for sentence in chunker.feed(piece):
                yield sentence
    else:
        for piece in chunks:  # type: ignore[union-attr]
            for sentence in chunker.feed(piece):
                yield sentence
    tail = chunker.flush()
    if tail:
        yield tail


def _sentence_end(buf: str, terminators: set[str], limit: int) -> tuple[int, int] | str | None:
    """First sentence end in ``buf[:limit]``.

    Returns ``(cut, resume)``: the chunk is ``buf[:cut]`` and the rest starts at
    ``resume``. Returns ``_WAIT`` when a terminator is the last character received, so
    what follows decides, and ``None`` when there is no sentence end.
    """
    i = 0
    while i < limit:
        if buf[i] not in terminators:
            i += 1
            continue
        start = i
        while i < len(buf) and buf[i] in terminators:  # "...", "?!"
            i += 1
        while i < len(buf) and buf[i] in _CLOSERS:
            i += 1
        if i >= len(buf):
            return _WAIT
        if not buf[i].isspace():
            continue  # "3.5", "example.com", "e.g.,": not a sentence end
        if buf[start:i].rstrip(_CLOSERS) == "." and not _ends_sentence(buf, start):
            continue  # "Dr. Smith", "U.S. army", "1. First"
        resume = i
        while resume < len(buf) and buf[resume] in " \t":
            resume += 1
        return i, resume
    return None


def _ends(buf: str, terminators: set[str]) -> list[tuple[int, int, Boundary]]:
    """Every decided sentence end and line break in ``buf``, in order.

    Each is ``(cut, resume, boundary)``. A terminator that is the last character is
    left out (what follows decides; at the end of a turn the caller adds it).
    """
    ends: list[tuple[int, int, Boundary]] = []
    pos = 0
    while pos < len(buf):
        sub = buf[pos:]
        newline = _first_index(sub, _FLUSH_WHITESPACE)
        limit = newline if newline != -1 else len(sub)
        end = _sentence_end(sub, terminators, limit)
        if isinstance(end, tuple):
            ends.append((pos + end[0], pos + end[1], "sentence"))
            pos += end[1]
            continue
        if newline == -1:
            break
        rest = newline + 1
        while rest < len(sub) and sub[rest] in _FLUSH_WHITESPACE:
            rest += 1
        if sub[:newline].strip():
            ends.append((pos + newline, pos + rest, "line"))
        pos += rest
    return ends


def _inner_cut(buf: str, lo: int, hi: int) -> tuple[int, Boundary] | None:
    """Best place to cut inside a sentence, between ``lo`` and ``hi`` characters.

    A clause break beats a word before a conjunction, which beats any word boundary;
    a cut in the upper half of the range is preferred over a better-ranked one that
    would leave a much shorter chunk.
    """
    hi = min(hi, len(buf) - 1)
    clause = conj = word = None
    for i in range(max(lo, 1), hi + 1):
        if not buf[i].isspace() or buf[i - 1].isspace():
            continue
        word = i
        if buf[i - 1] in _CLAUSE_BREAKS:
            clause = i
        nxt = buf[i + 1 :].split(None, 1)
        if nxt and nxt[0].strip("\"'(*_").lower() in _CONJUNCTIONS:
            conj = i
    candidates: list[tuple[int | None, Boundary]] = [
        (clause, "clause"),
        (conj, "word"),
        (word, "word"),
    ]
    for cut, boundary in candidates:
        if cut is not None and cut >= max(lo, hi // 2):
            return cut, boundary
    for cut, boundary in candidates:
        if cut is not None:
            return cut, boundary
    return None


def _ends_sentence(buf: str, dot: int) -> bool:
    """Whether the single ``.`` at ``buf[dot]`` ends a sentence."""
    start = dot
    while start > 0 and not buf[start - 1].isspace():
        start -= 1
    word = buf[start:dot].lstrip("\"'([{*_`")
    if not word:
        return True
    if word.lower() in _ABBREVIATIONS:
        return False
    if _INITIALS.match(word) and ("." in word or word.isupper()):
        return False  # "U.S." or a lone initial like "J."
    if word.isdigit() and not buf[:start].strip():
        return False  # "1." numbering a list item at the start of a chunk
    return True


def _first_index(s: str, chars: str) -> int:
    for i, c in enumerate(s):
        if c in chars:
            return i
    return -1
