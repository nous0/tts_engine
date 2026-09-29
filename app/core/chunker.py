"""Sentence chunker for incremental chatbot streaming.

Text from an LLM arrives token-by-token, but TTS providers synthesize best on
complete utterances. ``SentenceChunker`` buffers incoming text and flushes a
chunk whenever it hits a sentence boundary (``.`` ``?`` ``!`` or a newline) or
a configurable maximum length, so each flushed piece can be synthesized and
streamed while later text is still arriving.

With ``fast_start``, the first chunk of a turn may also end at a clause break
(``,`` ``;`` ``:``) once it is long enough, so the first audio comes sooner. Only the
first chunk does this: later chunks keep whole sentences, which sound better.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

# Characters that end a sentence. The terminator is kept on the flushed chunk.
_TERMINATORS = ".!?"
# Whitespace that should force a flush without being part of the chunk.
_FLUSH_WHITESPACE = "\n\r"
# Clause breaks a fast-start first chunk may end on.
_CLAUSE_BREAKS = ",;:"


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
            if not self._buffer:
                return out

            # 1) Flush on hard whitespace (newlines) — strip them off the chunk.
            ws_idx = _first_index(self._buffer, _FLUSH_WHITESPACE)
            if ws_idx != -1:
                chunk = self._buffer[:ws_idx]
                # Skip the newline and any consecutive whitespace after it.
                end = ws_idx + 1
                while end < len(self._buffer) and self._buffer[end] in _FLUSH_WHITESPACE:
                    end += 1
                self._buffer = self._buffer[end:]
                if chunk:
                    out.append(chunk)
                continue

            term_idx = _first_index(self._buffer, "".join(self._terminators))

            # 1b) Fast start: the turn's first chunk may end at a clause break that
            #     comes before the first sentence end.
            if self._fast_start and not self._emitted and not out:
                clause_idx = self._first_clause_break()
                if clause_idx != -1 and (term_idx == -1 or clause_idx < term_idx):
                    out.append(self._buffer[: clause_idx + 1])
                    self._buffer = self._buffer[clause_idx + 1 :]
                    continue

            # 2) Flush on a sentence terminator (keep the terminator + any
            #    trailing closing quotes on the chunk; drop spaces after it).
            if term_idx != -1:
                cut = term_idx + 1
                while cut < len(self._buffer) and self._buffer[cut] in "\"'":
                    cut += 1
                chunk = self._buffer[:cut]
                end = cut
                while end < len(self._buffer) and self._buffer[end] in " \t":
                    end += 1
                self._buffer = self._buffer[end:]
                if chunk:
                    out.append(chunk)
                continue

            # 3) Flush when the buffer exceeds the max length, breaking on a word
            #    boundary if possible so we don't split a word mid-stream.
            if len(self._buffer) > self._max_length:
                cut = self._max_length
                space_idx = self._buffer.rfind(" ", 0, cut)
                if space_idx > 0:
                    cut = space_idx
                out.append(self._buffer[:cut].rstrip())
                self._buffer = self._buffer[cut:].lstrip()
                continue

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

        The chunk must be at least ``clause_min_chars`` long, and the break must be
        followed by text we've already seen, so ``1,000`` isn't split when ``,`` is the
        last character received so far.
        """
        buf = self._buffer
        for i, ch in enumerate(buf):
            if ch not in _CLAUSE_BREAKS or i + 1 < self._clause_min_chars:
                continue
            if i + 1 >= len(buf):
                return -1  # can't tell yet what follows
            if buf[i - 1].isdigit() and buf[i + 1].isdigit():
                continue  # a number like 1,000 or a time like 10:30
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


def _first_index(s: str, chars: str) -> int:
    for i, c in enumerate(s):
        if c in chars:
            return i
    return -1
