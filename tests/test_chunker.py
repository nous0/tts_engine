"""Tests for the sentence chunker."""

from __future__ import annotations

import pytest

from app.core.chunker import (
    SentenceChunker,
    iter_sentences,
    split_sentences,
)


def test_single_sentence_stays_buffered_until_flush():
    chunker = SentenceChunker()
    assert chunker.feed("Hello world") == []
    assert chunker.flush() == "Hello world"
    assert chunker.flush() is None


def test_flushes_on_period():
    chunker = SentenceChunker()
    out = chunker.feed("Hello world. How are you?")
    assert out == ["Hello world.", "How are you?"]
    assert chunker.flush() is None


def test_keeps_terminator_and_trailing_quotes():
    chunker = SentenceChunker()
    out = chunker.feed('She said "hi." Then left.')
    assert out == ['She said "hi."', "Then left."]


def test_flushes_on_question_and_exclamation():
    chunker = SentenceChunker()
    out = chunker.feed("Really? Yes! OK.")
    assert out == ["Really?", "Yes!", "OK."]


def test_flushes_on_newline_and_strips_whitespace():
    chunker = SentenceChunker()
    out = chunker.feed("First line\n\n  Second line\n")
    assert out == ["First line", "Second line"]
    assert chunker.flush() is None


def test_incremental_token_feed_assembles_sentences():
    chunker = SentenceChunker()
    received: list[str] = []
    for token in "Hello there. How are you?".split(" "):
        received.extend(chunker.feed(token + " "))
    tail = chunker.flush()
    if tail:
        received.append(tail)
    assert received == ["Hello there.", "How are you?"]


def test_max_length_flushes_on_word_boundary():
    chunker = SentenceChunker(max_length=20)
    text = "abcdefghijklmnopqrstuvwxyz"
    out = chunker.feed(text)
    # No spaces -> falls back to hard cut at max_length.
    assert out == ["abcdefghijklmnopqrst"]
    assert chunker.flush() == "uvwxyz"


def test_max_length_breaks_on_space_when_possible():
    chunker = SentenceChunker(max_length=15)
    out = chunker.feed("alpha beta gamma delta epsilon")
    tail = chunker.flush()
    if tail:
        out.append(tail)
    # Should never split a word; each chunk <= max_length and on word boundaries.
    assert all(len(c) <= 15 for c in out)
    assert "".join(c + " " for c in out).strip() == "alpha beta gamma delta epsilon"


def test_empty_feed_yields_nothing():
    chunker = SentenceChunker()
    assert chunker.feed("") == []
    assert chunker.flush() is None


def test_whitespace_only_feed_yields_nothing():
    chunker = SentenceChunker()
    assert chunker.feed("   \n  ") == []
    assert chunker.flush() is None


def test_invalid_max_length_raises():
    with pytest.raises(ValueError):
        SentenceChunker(max_length=0)


def test_split_sentences_helper():
    assert split_sentences("A. B. C.") == ["A.", "B.", "C."]
    assert split_sentences("No terminator here") == ["No terminator here"]


async def test_iter_sentences_from_sync_iterable():
    out = [s async for s in iter_sentences(["Hello ", "world. ", "Bye!"])]
    assert out == ["Hello world.", "Bye!"]


async def test_iter_sentences_from_async_iterable():
    async def gen():
        yield "Hello "
        yield "world."
        yield " Bye!"

    out = [s async for s in iter_sentences(gen())]
    assert out == ["Hello world.", "Bye!"]


async def test_iter_sentences_emits_tail():
    out = [s async for s in iter_sentences(["just some words"])]
    assert out == ["just some words"]
