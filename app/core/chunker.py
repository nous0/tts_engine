"""Sentence chunker for incremental chatbot streaming.

Text from an LLM arrives token-by-token, but TTS providers synthesize best on
complete utterances. ``SentenceChunker`` buffers incoming text and flushes a
chunk whenever it hits a sentence boundary (``.`` ``?`` ``!`` or a newline) or
a configurable maximum length, so each flushed piece can be synthesized and
streamed while later text is still arriving.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

# Characters that end a sentence. The terminator is kept on the flushed chunk.
_TERMINATORS = ".!?"
# Whitespace that should force a flush without being part of the chunk.
_FLUSH_WHITESPACE = "\n\r"


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

    def __init__(self, max_length: int = 200, terminators: str = _TERMINATORS) -> None:
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self._max_length = max_length
        self._terminators = set(terminators)
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Append ``text`` to the buffer and return any complete chunks."""
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

            # 2) Flush on a sentence terminator (keep the terminator + any
            #    trailing closing quotes on the chunk; drop spaces after it).
            term_idx = _first_index(self._buffer, "".join(self._terminators))
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
        """Return any remaining buffered text, or ``None`` if empty."""
        tail = self._buffer.strip()
        self._buffer = ""
        return tail or None

    @property
    def pending(self) -> str:
        """Text currently buffered but not yet flushed."""
        return self._buffer


def split_sentences(text: str, max_length: int = 200) -> list[str]:
    """Split a complete ``text`` into sentence-ish chunks (stateless helper)."""
    chunker = SentenceChunker(max_length=max_length)
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
