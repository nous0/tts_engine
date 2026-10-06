"""Tests for the sentence chunker."""

from __future__ import annotations

import random

import pytest

from app.core.chunker import (
    Chunk,
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
    # The last "?" could still be followed by more text, so it waits for the flush.
    assert out == ["Hello world."]
    assert chunker.flush() == "How are you?"
    assert chunker.flush() is None


def test_keeps_terminator_and_trailing_quotes():
    chunker = SentenceChunker()
    out = chunker.feed('She said "hi." Then left. ')
    assert out == ['She said "hi."', "Then left."]


def test_flushes_on_question_and_exclamation():
    chunker = SentenceChunker()
    out = chunker.feed("Really? Yes! OK. ")
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
    assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]
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
    assert chunker.feed(" is fine.") == []  # waits: "fine." might be "fine.com"
    assert chunker.feed(" Ok") == ["which is fine."]


def test_flush_and_reset_start_a_new_fast_start_turn():
    chunker = SentenceChunker(fast_start=True)
    first_turn = "Here is a first clause long enough, and more text"
    assert chunker.feed(first_turn) == ["Here is a first clause long enough,"]
    assert chunker.flush() == "and more text"
    assert chunker.feed(first_turn) == ["Here is a first clause long enough,"]
    chunker.reset()
    assert chunker.pending == ""
    assert chunker.feed(first_turn) == ["Here is a first clause long enough,"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The latency is 3.5 seconds on average.", ["The latency is 3.5 seconds on average."]),
        ("It costs $4.99 per month.", ["It costs $4.99 per month."]),
        ("See example.com for details. Thanks!", ["See example.com for details.", "Thanks!"]),
        ("Dr. Smith met Mr. Jones.", ["Dr. Smith met Mr. Jones."]),
        ("We met in the U.S. last week.", ["We met in the U.S. last week."]),
        ("Use a tool, e.g. a hammer. Done.", ["Use a tool, e.g. a hammer.", "Done."]),
        ("J. K. Rowling wrote it.", ["J. K. Rowling wrote it."]),
        ("Wait... really? Yes.", ["Wait...", "really?", "Yes."]),
        ("Really?! No way.", ["Really?!", "No way."]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
        ("**Note.** This matters.", ["**Note.**", "This matters."]),
        ("1. First item\n2. Second item", ["1. First item", "2. Second item"]),
        ("I counted to 3. Then I stopped.", ["I counted to 3.", "Then I stopped."]),
        ("Line one. Line two.\nNext line", ["Line one.", "Line two.", "Next line"]),
        ("We open at 9 a.m. every day.", ["We open at 9 a.m. every day."]),
        ("It ends here … then more.", ["It ends here …", "then more."]),
    ],
)
def test_sentence_boundaries(text, expected):
    assert split_sentences(text) == expected


def test_a_terminator_at_the_end_waits_for_the_next_character():
    chunker = SentenceChunker()
    assert chunker.feed("It is 3.") == []  # could become 3.5
    assert chunker.feed("5 seconds.") == []
    assert chunker.feed(" Next") == ["It is 3.5 seconds."]


def test_long_sentences_split_the_same_way_streamed_or_whole():
    text = "word " * 60 + "end."
    whole = split_sentences(text, max_length=50)
    assert all(len(chunk) <= 50 for chunk in whole)
    chunker = SentenceChunker(max_length=50)
    streamed = []
    for word in text.split(" "):
        streamed += chunker.feed(word + " ")
    tail = chunker.flush()
    streamed += [tail] if tail else []
    assert streamed == whole


# Text that exercises every rule: decimals, abbreviations, initials, URLs, ellipses,
# quotes, markdown, lists, newlines, numbers with commas, long sentences.
INVARIANT_CORPUS = [
    "Hello! How are you today? I'm fine, thanks.",
    "The latency is 3.5 seconds, which is fine. Next we test 1,000 requests.",
    "Dr. Smith met Mr. Jones in the U.S. last week. They talked for 2.5 hours.",
    "See example.com or https://docs.example.org/a?b=1 for details. Thanks!",
    "Wait... really?! Yes. \"Quoted.\" (Parenthetical.) Done",
    "**Bold.** _Italic._ `code.` Then plain text follows here.",
    "1. First item\n2. Second item\n- bullet one\n- bullet two",
    "Use a tool, e.g. a hammer, i.e. something heavy. Then stop.",
    "When you think about how speech synthesis has changed over the last few years, "
    "the biggest shift is direct audio generation. That sounds natural.",
    "It costs $4.99 per month at 10:30 a.m. every day, approx. forever.",
    "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.",
    "A sentence without any terminator at all but quite long nonetheless and it keeps "
    "going on and on and on without stopping for a very long time indeed my friend",
    "Short. " * 10 + "A final longer sentence that wraps it all up nicely.",
    "Line one\n\nLine two after a blank line.\r\nWindows line three.",
    "Numbers like 1,000,000 and 3.14159 and 2.0, plus times like 12:45, stay whole.",
    "J. K. Rowling and George R. R. Martin wrote books. Both sold well.",
    "What?! No... I mean, yes. Okay?",
    "Emoji \U0001F389 and symbols ❤ sit inside text. Then a sentence.",
    "He said \"stop.\" She said 'go!' They left.",
    "Mixed: first clause, second clause; third clause: and the end.",
]


@pytest.mark.parametrize("fast_start", [False, True])
@pytest.mark.parametrize("index", range(len(INVARIANT_CORPUS)))
def test_streaming_token_by_token_matches_the_whole_text(index, fast_start):
    text = INVARIANT_CORPUS[index]
    expected = split_sentences(text, max_length=80, fast_start=fast_start)
    # Nothing is lost or duplicated: the chunks hold exactly the original words.
    assert " ".join(expected).split() == text.split()

    rng = random.Random(index)
    for _ in range(50):
        chunker = SentenceChunker(max_length=80, fast_start=fast_start)
        streamed: list[str] = []
        pos = 0
        while pos < len(text):
            size = rng.randint(1, 8)
            streamed += chunker.feed(text[pos : pos + size])
            pos += size
        tail = chunker.flush()
        if tail:
            streamed.append(tail)
        assert streamed == expected, f"token split changed the chunks for {text!r}"


# --- pull mode: append / take ------------------------------------------------------


def _pull(text: str, budgets: list[int | None], final: bool = True) -> list[Chunk]:
    chunker = SentenceChunker()
    chunker.append(text)
    out: list[Chunk] = []
    for budget in budgets + [None] * 20:
        chunk = chunker.take(budget, final=final)
        if chunk is None:
            break
        out.append(chunk)
    return out


def test_take_waits_for_an_unfinished_sentence():
    chunker = SentenceChunker()
    chunker.append("Hello there, how are")
    assert chunker.take(40, final=False) is None
    chunker.append(" you today?")
    assert chunker.take(40, final=True) == Chunk("Hello there, how are you today?", "end")


def test_take_merges_whole_sentences_that_fit():
    # (With the whole turn fitting the budget, it would all be one chunk.)
    chunks = _pull("One sentence here. Another one here. A third sentence.", [40])
    assert chunks[0] == Chunk("One sentence here. Another one here.", "sentence")


def test_take_merges_a_tiny_first_sentence_instead_of_speaking_it_alone():
    chunker = SentenceChunker()
    chunker.append("Hi! I can")
    assert chunker.take(40, final=False) is None  # "Hi!" alone is too short: wait
    chunker.append(" help you plan a relaxing trip to the coast this weekend.")
    chunk = chunker.take(40, final=False)
    assert chunk.text.startswith("Hi! I can help") and len(chunk.text) <= 40


def test_take_cuts_inside_a_long_sentence_at_the_best_boundary():
    clause = _pull("When you think about how speech has changed over the years, the rest.",
                   [65])[0]
    assert clause == Chunk("When you think about how speech has changed over the years,",
                           "clause")
    conj = _pull("The weather is sunny with a light breeze that makes it perfect today.",
                 [50])[0]
    assert conj.text == "The weather is sunny with a light breeze" and conj.boundary == "word"


def test_take_respects_the_budget_and_minimum():
    text = "word " * 40 + "end."
    for budget in (20, 33, 57):
        chunk = _pull(text, [budget])[0]
        # The minimum is soft: with a budget right at it, the nearest word wins.
        assert 15 <= len(chunk.text) <= budget


def test_take_without_a_budget_uses_whole_sentences():
    chunks = _pull("First sentence, with a comma. Second sentence.", [None])
    assert chunks == [Chunk("First sentence, with a comma. Second sentence.", "end")]


@pytest.mark.parametrize("index", range(len(INVARIANT_CORPUS)))
def test_take_never_loses_or_duplicates_text(index):
    text = INVARIANT_CORPUS[index]
    rng = random.Random(index)
    for _ in range(30):
        chunker = SentenceChunker(max_length=80)
        taken: list[str] = []
        pos = 0
        while pos < len(text):  # text arrives in random pieces, takes interleave
            size = rng.randint(1, 12)
            chunker.append(text[pos : pos + size])
            pos += size
            chunk = chunker.take(rng.choice([20, 30, 45, 80, None]), final=False)
            if chunk:
                taken.append(chunk.text)
                assert len(chunk.text) <= 80
        while (chunk := chunker.take(rng.choice([20, 45, None]), final=True)) is not None:
            taken.append(chunk.text)
        assert " ".join(taken).split() == text.split()


def test_take_never_splits_a_word_even_past_the_budget():
    chunks = _pull("https://docs.example.org/a?b=1 for details.", [20])
    assert chunks[0].text == "https://docs.example.org/a?b=1"
