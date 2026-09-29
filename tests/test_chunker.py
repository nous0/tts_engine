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


LONG_FIRST = (
    "When you think about how speech has changed over the years, the biggest shift "
    "is direct audio generation. That sounds natural."
)


def test_fast_start_ends_the_first_chunk_at_a_clause_break():
    assert split_sentences(LONG_FIRST, fast_start=True) == [
        "When you think about how speech has changed over the years,",
        "the biggest shift is direct audio generation.",
        "That sounds natural.",
    ]


def test_fast_start_is_off_by_default():
    assert split_sentences(LONG_FIRST)[0].endswith("direct audio generation.")


def test_fast_start_only_affects_the_first_chunk():
    chunks = split_sentences(
        "Well then, here is a first sentence that is long enough to split, "
        "yes. Then later, a second one, with commas, stays whole.",
        fast_start=True,
    )
    assert chunks[-1] == "Then later, a second one, with commas, stays whole."


def test_fast_start_ignores_breaks_too_early_in_the_chunk():
    assert split_sentences("Sure, I can help with that. Ok.", fast_start=True) == [
        "Sure, I can help with that.",
        "Ok.",
    ]


def test_fast_start_prefers_a_sentence_end_that_comes_first():
    text = "This first sentence is short enough. But this one, being long, has a comma."
    assert split_sentences(text, fast_start=True)[0] == "This first sentence is short enough."


def test_fast_start_never_splits_numbers():
    chunks = split_sentences("The total comes to exactly 1,000 dollars, as agreed.",
                             fast_start=True)
    assert chunks[0] == "The total comes to exactly 1,000 dollars,"


def test_fast_start_waits_to_see_what_follows_a_streamed_comma():
    chunker = SentenceChunker(fast_start=True)
    assert chunker.feed("The total price comes to 1") == []
    assert chunker.feed(",") == []  # could be "1,000": wait for the next token
    assert chunker.feed("000 dollars, which") == ["The total price comes to 1,000 dollars,"]
    assert chunker.feed(" is fine.") == ["which is fine."]


def test_flush_and_reset_start_a_new_fast_start_turn():
    chunker = SentenceChunker(fast_start=True)
    first_turn = "Here is a first clause long enough, and more text"
    assert chunker.feed(first_turn) == ["Here is a first clause long enough,"]
    assert chunker.flush() == "and more text"
    assert chunker.feed(first_turn) == ["Here is a first clause long enough,"]
    chunker.reset()
    assert chunker.pending == ""
    assert chunker.feed(first_turn) == ["Here is a first clause long enough,"]
