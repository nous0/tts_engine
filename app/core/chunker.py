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
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable

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
# "U.S", "J", "e.g"-style initials: single letters joined by dots.
_INITIALS = re.compile(r"^(?:[A-Za-z]\.)*[A-Za-z]$")
# Returned by _sentence_end when the next character decides.
_WAIT = "wait"


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
        out = self._feed(text)
        if out:
            self._emitted = True
        return out

    def _feed(self, text: str) -> list[str]:
        out: list[str] = []
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
                out.append(buf[:cut].rstrip())
                self._buffer = buf[cut:].lstrip()
                continue

            # 1) Fast start: the turn's first chunk may end at a clause break that
            #    comes before the first sentence end or newline.
            if self._fast_start and not self._emitted and not out:
                clause_idx = self._first_clause_break()
                if clause_idx != -1 and (first_cut is None or clause_idx < first_cut):
                    out.append(buf[: clause_idx + 1])
                    self._buffer = buf[clause_idx + 1 :]
                    continue

            # 2) A sentence end before any newline.
            if isinstance(end, tuple):
                cut, resume = end
                out.append(buf[:cut])
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
                    out.append(chunk)
                continue

            # A terminator is the last thing received (end is _WAIT), or there is no
            # boundary yet: either way, wait for more text.
            return out

    def flush(self) -> str | None:
        """End the turn: return any remaining buffered text, or ``None`` if empty.

        The next text starts a new turn (which fast-starts again).
        """
        tail = self._buffer.strip()
        self.reset()
        return tail or None

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
    chunker = SentenceChunker(max_length=max_length, fast_start=fast_start)
    chunks = chunker.feed(text)
    tail = chunker.flush()
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
