"""Tests for making chatbot text speakable."""

from __future__ import annotations

import pytest

from app.core.textnorm import normalize_for_speech


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Bold** and _italics_ here.", "Bold and italics here."),
        ("***Both*** and __dunder__ too.", "Both and dunder too."),
        ("# A heading", "A heading"),
        ("- a bullet point", "a bullet point"),
        ("* another bullet", "another bullet"),
        ("1. First item", "First item"),
        ("> quoted text", "quoted text"),
        ("Use `pip install` now.", "Use pip install now."),
        ("See [the docs](https://example.com/x) now.", "See the docs now."),
        ("Visit https://example.com/path?a=1 today.", "Visit link today."),
        ("Go to www.example.com for more.", "Go to link for more."),
        ("Great job! \U0001F389\U0001F44D", "Great job!"),
        ("I ❤️ this", "I this"),
        ("snake_case_names stay intact.", "snake_case_names stay intact."),
        ("Price: $4.99, approx. 3.5 hours.", "Price: $4.99, approx. 3.5 hours."),
        ("  lots   of\tspace  ", "lots of space"),
    ],
)
def test_normalize_for_speech(text, expected):
    assert normalize_for_speech(text) == expected


def test_a_chunk_of_only_markup_or_emoji_becomes_empty():
    assert normalize_for_speech("**") == ""
    assert normalize_for_speech("\U0001F600\U0001F600") == ""
